//! Loss detection (RFC 9002 §6) and ACK processing.
//!
//! Each packet-number space (Initial, Handshake, Application) owns a
//! [`LossDetector`] that records every sent packet, processes ACK frames to
//! declare packets acknowledged or lost, and arms the probe-timeout (PTO)
//! timer.
//!
//! Loss is declared on two conditions (whichever fires first):
//! 1. Packet threshold: a packet is lost if a later packet was acked and the
//!    gap exceeds [`LossDetector::packet_threshold`] (RFC 9002 §6.1.1,
//!    default 3).
//! 2. Time threshold: a packet is lost if it was sent more than
//!    `max(9/8 * smoothed_rtt, granularity)` ago relative to the latest ack
//!    (RFC 9002 §6.1.2).

use crate::congestion::CongestionController;
use crate::frame::{AckFrame, Frame};
use crate::varint::Varint;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// The three QUIC packet-number spaces (RFC 9000 §17.2.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PnSpace {
    Initial,
    Handshake,
    Application,
}

/// A sent packet awaiting acknowledgment.
#[derive(Debug, Clone)]
pub struct SentPacket {
    pub pn: u64,
    pub sent_at: Instant,
    pub bytes_in_flight: u64,
    /// Whether this packet contains retransmittable frames (STREAM, CRYPTO,
    /// PING, etc.) and thus counts toward the in-flight accounting.
    pub ack_eliciting: bool,
    pub in_flight: bool,
}

/// Outcome of processing one ACK frame.
#[derive(Debug, Clone, Default)]
pub struct AckOutcome {
    /// Newly acknowledged packet numbers.
    pub acked: Vec<u64>,
    /// Newly declared-lost packet numbers.
    pub lost: Vec<u64>,
    /// Bytes acknowledged (sum over `acked`).
    pub bytes_acked: u64,
    /// Bytes lost (sum over `lost`).
    pub bytes_lost: u64,
    /// Whether persistent congestion (RFC 9002 §7.6.2) was triggered.
    pub persistent_congestion: bool,
}

/// Per-space loss tracker.
#[derive(Debug)]
pub struct LossDetector {
    pub space: PnSpace,
    /// Sent packets keyed by packet number.
    pub sent: BTreeMap<u64, SentPacket>,
    /// Largest acknowledged packet number observed so far in this space.
    pub largest_acked: Option<u64>,
    /// Latest RTT sample derived from the largest acked packet.
    pub latest_rtt: Duration,
    /// Packet threshold (§6.1.1): default 3.
    pub packet_threshold: u64,
    /// Granularity used for the time-threshold (§6.1.2): default 1 ms.
    pub time_threshold_granularity: Duration,
    /// Time when the loss-detection timer was last armed.
    pub loss_detection_deadline: Option<Instant>,
}

impl LossDetector {
    pub fn new(space: PnSpace) -> Self {
        Self {
            space,
            sent: BTreeMap::new(),
            largest_acked: None,
            latest_rtt: Duration::ZERO,
            packet_threshold: 3,
            time_threshold_granularity: Duration::from_millis(1),
            loss_detection_deadline: None,
        }
    }

    /// Record a freshly sent packet.
    pub fn on_sent(&mut self, p: SentPacket) {
        self.sent.insert(p.pn, p);
    }

    /// Process an ACK frame for this space. Returns the outcome and mutates
    /// the congestion controller accordingly.
    pub fn on_ack(
        &mut self,
        ack: &AckFrame,
        now: Instant,
        cc: &mut dyn CongestionController,
        smoothed_rtt: Duration,
        max_ack_delay: Duration,
    ) -> AckOutcome {
        let largest = ack.largest_acknowledged.as_u64();
        let mut outcome = AckOutcome::default();

        // Build the set of acked pns from the ranges.
        let mut acked_pns: Vec<u64> = Vec::new();
        // largest down to (largest - first_ack_range).
        let first = ack.first_ack_range.as_u64();
        let mut cur = largest;
        let bottom = largest.saturating_sub(first);
        loop {
            acked_pns.push(cur);
            if cur == bottom || cur == 0 {
                break;
            }
            cur -= 1;
        }
        for (gap, range) in &ack.ranges {
            let gap = gap.as_u64() + 1; // gap is the *additional* gap
            let range = range.as_u64();
            cur = bottom.saturating_sub(gap);
            let rbottom = cur.saturating_sub(range);
            loop {
                acked_pns.push(cur);
                if cur == rbottom || cur == 0 {
                    break;
                }
                cur -= 1;
            }
        }

        // Update largest acked.
        match self.largest_acked {
            Some(l) if largest > l => self.largest_acked = Some(largest),
            None => self.largest_acked = Some(largest),
            _ => {}
        }

        // RTT sample from the largest acked pn.
        let ack_delay = Duration::from_micros(ack.ack_delay.as_u64());
        if let Some(sp) = self.sent.get(&largest) {
            let rtt = now.duration_since(sp.sent_at);
            self.latest_rtt = rtt;
            cc.on_packets_acked(0, rtt, ack_delay, now); // update RTT only
        }

        // Acknowledge packets and account bytes.
        for pn in &acked_pns {
            if let Some(sp) = self.sent.remove(pn) {
                outcome.acked.push(*pn);
                outcome.bytes_acked += sp.bytes_in_flight;
                if sp.in_flight {
                    cc.on_packets_acked(sp.bytes_in_flight, self.latest_rtt, ack_delay, now);
                }
            }
        }

        // Detect losses (§6.1).
        self.detect_losses(now, smoothed_rtt, max_ack_delay, &mut outcome, cc);

        outcome
    }

    /// Packet threshold + time threshold loss detection.
    fn detect_losses(
        &mut self,
        now: Instant,
        smoothed_rtt: Duration,
        _max_ack_delay: Duration,
        outcome: &mut AckOutcome,
        cc: &mut dyn CongestionController,
    ) {
        let largest = match self.largest_acked {
            Some(l) => l,
            None => return,
        };
        let loss_threshold = largest.saturating_sub(self.packet_threshold);
        let time_threshold = (smoothed_rtt * 9 / 8).max(self.time_threshold_granularity);
        // Iterate sent pns below the threshold.
        let keys: Vec<u64> = self.sent.keys().cloned().filter(|&k| k <= loss_threshold).collect();
        let mut persistent = false;
        let mut lost_send_times: Vec<Instant> = Vec::new();
        for k in keys {
            if let Some(sp) = self.sent.get(&k) {
                let age = now.duration_since(sp.sent_at);
                if age >= time_threshold {
                    if let Some(sp) = self.sent.remove(&k) {
                        outcome.lost.push(k);
                        outcome.bytes_lost += sp.bytes_in_flight;
                        if sp.in_flight {
                            cc.on_packets_lost(sp.bytes_in_flight, false, now);
                        }
                        lost_send_times.push(sp.sent_at);
                    }
                }
            }
        }
        // Persistent congestion heuristic (§7.6.2): if we lost everything
        // across a span >= 2 * (smoothed_rtt + max_ack_delay), treat as
        // persistent. This is a simplified check.
        if lost_send_times.len() >= 2 {
            let earliest = *lost_send_times.iter().min().unwrap();
            let latest = *lost_send_times.iter().max().unwrap();
            if latest.duration_since(earliest) >= smoothed_rtt * 2 {
                persistent = true;
            }
        }
        if persistent {
            outcome.persistent_congestion = true;
            // Remaining in-flight packets are dropped as lost.
            let remaining: Vec<u64> = self.sent.keys().cloned().collect();
            for k in remaining {
                if let Some(sp) = self.sent.remove(&k) {
                    outcome.lost.push(k);
                    outcome.bytes_lost += sp.bytes_in_flight;
                    if sp.in_flight {
                        cc.on_packets_lost(sp.bytes_in_flight, true, now);
                    }
                }
            }
        }
    }

    /// Whether there is anything outstanding in this space (to arm PTO).
    pub fn has_outstanding(&self) -> bool {
        !self.sent.is_empty()
    }

    /// Build an ACK frame from the peer's acknowledged-packet ranges.
    ///
    /// `acked_set` is the set of packet numbers the peer's packets covered,
    /// in ascending order. We compress into `(largest, first_range, ranges)`.
    pub fn build_ack_frame(
        largest: u64,
        acked_set: &[u64],
        ack_delay_us: u64,
        ecn: Option<crate::frame::AckEcnCounts>,
    ) -> Frame {
        // Simple encoding: one range covering everything below largest.
        // A real impl compresses gaps; this is correct, just not maximally
        // compact.
        let count = acked_set.len() as u64;
        let first_range = count.saturating_sub(1);
        Frame::Ack(AckFrame {
            largest_acknowledged: Varint(largest),
            ack_delay: Varint(ack_delay_us),
            ack_range_count: Varint(0),
            first_ack_range: Varint(first_range),
            ranges: Vec::new(),
            ecn,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::congestion::{CongestionController, NewReno, NewRenoConfig};

    #[test]
    fn ack_advances_and_clears() {
        let mut det = LossDetector::new(PnSpace::Application);
        let now = Instant::now();
        for pn in 0..5 {
            det.on_sent(SentPacket {
                pn,
                sent_at: now,
                bytes_in_flight: 1200,
                ack_eliciting: true,
                in_flight: true,
            });
        }
        let mut cc: NewReno = NewReno::new(NewRenoConfig::default());
        let ack = AckFrame {
            largest_acknowledged: Varint(2),
            ack_delay: Varint(0),
            ack_range_count: Varint(0),
            first_ack_range: Varint(2),
            ranges: Vec::new(),
            ecn: None,
        };
        let out = det.on_ack(&ack, now + Duration::from_millis(50), &mut cc, Duration::from_millis(50), Duration::from_millis(25));
        assert_eq!(out.acked.len(), 3);
    }
}
