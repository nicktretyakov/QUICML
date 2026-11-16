//! Congestion control (RFC 9002).
//!
//! This module defines a [`CongestionController`] trait that the connection
//! drives on every acknowledged, lost or ECN-CE-marked packet. The trait is
//! deliberately ML-friendly: an implementation can be a classic Reno/Cubic
//! controller, a learned policy (see `quic-ml::rl_congestion`), or anything
//! in between, and the connection code does not change.
//!
//! The reference implementation [`NewReno`] follows the pseudocode in RFC 9002
//! Appendix B and is the default controller for this crate.

use std::time::{Duration, Instant};

/// Configuration for [`NewReno`].
#[derive(Debug, Clone, Copy)]
pub struct NewRenoConfig {
    /// Initial congestion window in bytes (RFC 9002 §7.2: min(10 * max_datagram_size, max(2*max_datagram_size, 14720))).
    pub initial_window: u64,
    /// Minimum congestion window (RFC 9002 §7.2: 2 * max_datagram_size).
    pub minimum_window: u64,
    /// Loss reduction factor applied on congestion event (Reno: 0.5).
    pub loss_reduction_factor: f64,
    /// Persistent congestion duration multiplier (RFC 9002 §7.6.2: 2 * (smoothed_rtt + max_ack_delay) ... handled in loss detection).
    pub persistent_congestion_multiplier: u32,
    /// Initial smooth RTT guess before any samples.
    pub initial_rtt: Duration,
}

impl Default for NewRenoConfig {
    fn default() -> Self {
        // Assume a 1200-byte max datagram size (conservative, common for QUIC).
        let mss: u64 = 1200;
        Self {
            initial_window: 10 * mss,
            minimum_window: 2 * mss,
            loss_reduction_factor: 0.5,
            persistent_congestion_multiplier: 3,
            initial_rtt: Duration::from_millis(333),
        }
    }
}

/// Statistics reported by a congestion controller for telemetry and ML input.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CongestionStats {
    pub bytes_in_flight: u64,
    pub congestion_window: u64,
    pub packets_acknowledged_total: u64,
    pub packets_lost_total: u64,
    pub congestion_events: u64,
    /// Smoothed RTT in microseconds, or 0 if no samples yet.
    pub smoothed_rtt_us: u64,
    /// RTT variance in microseconds.
    pub rttvar_us: u64,
    /// Minimum RTT observed, microseconds.
    pub min_rtt_us: u64,
}

/// Trait implemented by every congestion controller, classical or learned.
///
/// The connection calls these methods from the loss-detection timer and the
/// ACK-processing path. Implementations must be cheap (single-digit
/// microsecond) because they run in the data path.
pub trait CongestionController: Send + Sync {
    /// Largest amount of unacknowledged data the connection may have in
    /// flight at once, in bytes.
    fn window(&self) -> u64;

    /// Current in-flight bytes (sent, not yet acked or declared lost).
    fn bytes_in_flight(&self) -> u64;

    /// Whether the controller is in a state that forbids sending new
    /// application data (e.g. during recovery). Probes are still allowed.
    fn is_congestion_limited(&self) -> bool {
        self.bytes_in_flight() >= self.window()
    }

    /// Called when `bytes` of new application data are placed on the wire.
    fn on_packet_sent(&mut self, bytes: u64, now: Instant);

    /// Called when packets are acknowledged. `bytes_ack` is the newly
    /// acknowledged payload bytes; `latest_rtt` is the RTT sample for the
    /// largest acknowledged packet; `ack_delay` is the peer-reported ack
    /// delay to subtract from the RTT sample.
    fn on_packets_acked(
        &mut self,
        bytes_ack: u64,
        latest_rtt: Duration,
        ack_delay: Duration,
        now: Instant,
    );

    /// Called when `bytes` of payload are declared lost (after the loss
    /// detection timer or PTO probes).
    fn on_packets_lost(&mut self, bytes: u64, persistent_congestion: bool, now: Instant);

    /// Called when an ECN-CE mark is observed in an ACK frame.
    fn on_ecn_ce(&mut self, ce_count: u64, now: Instant);

    /// Called once per ACK to update the PTO probe deadline if needed.
    /// Returns the duration until the next probe should fire, or `None` if
    /// no probe is pending.
    fn pto_timeout(&self, now: Instant, max_ack_delay: Duration) -> Option<Duration>;

    /// Snapshot of internal stats for telemetry / ML feature extraction.
    fn stats(&self) -> CongestionStats;
}

/// Reno congestion control as specified in RFC 9002 Appendix B.
pub struct NewReno {
    cfg: NewRenoConfig,
    /// Congestion window in bytes.
    cwnd: u64,
    bytes_in_flight: u64,
    /// Slow-start threshold. When cwnd < ssthresh we are in slow start.
    ssthresh: u64,
    /// Earliest packet sent in the current recovery episode; `None` if not
    /// in recovery.
    recovery_start: Option<Instant>,
    /// RTT estimator (RFC 9002 §5).
    smoothed_rtt: Duration,
    rttvar: Duration,
    min_rtt: Duration,
    /// Timestamp of the latest RTT sample's packet send time (for min_rtt).
    last_send_time: Option<Instant>,
    counters: CongestionStats,
}

impl NewReno {
    pub fn new(cfg: NewRenoConfig) -> Self {
        let cwnd = cfg.initial_window;
        Self {
            cfg,
            cwnd,
            bytes_in_flight: 0,
            ssthresh: u64::MAX,
            recovery_start: None,
            smoothed_rtt: cfg.initial_rtt,
            rttvar: cfg.initial_rtt / 2,
            min_rtt: Duration::MAX,
            last_send_time: None,
            counters: CongestionStats {
                congestion_window: cwnd,
                ..Default::default()
            },
        }
    }

    /// Whether a packet sent at `sent_time` should still contribute to the
    /// current recovery episode (RFC 9002 §7.3.2).
    fn in_recovery(&self, sent_time: Instant) -> bool {
        match self.recovery_start {
            Some(start) => sent_time <= start,
            None => false,
        }
    }

    fn update_rtt(&mut self, latest_rtt: Duration, ack_delay: Duration) {
        // RFC 9002 §5.3 — ignore ack_delay for the first sample, and for
        // samples that would push min_rtt below its current value.
        if self.smoothed_rtt == self.cfg.initial_rtt && self.min_rtt == Duration::MAX {
            // first sample
            self.smoothed_rtt = latest_rtt;
            self.rttvar = latest_rtt / 2;
            self.min_rtt = latest_rtt;
        } else {
            let adjusted = if latest_rtt > self.min_rtt + ack_delay {
                latest_rtt - ack_delay
            } else {
                latest_rtt
            };
            let s = self.smoothed_rtt;
            let v = self.rttvar;
            self.smoothed_rtt = (s * 7 + adjusted) / 8;
            let diff = if s > adjusted { s - adjusted } else { adjusted - s };
            self.rttvar = (v * 3 + diff) / 4;
            if latest_rtt < self.min_rtt {
                self.min_rtt = latest_rtt;
            }
        }
        self.counters.smoothed_rtt_us = self.smoothed_rtt.as_micros() as u64;
        self.counters.rttvar_us = self.rttvar.as_micros() as u64;
        self.counters.min_rtt_us = self.min_rtt.as_micros() as u64;
    }
}

impl CongestionController for NewReno {
    fn window(&self) -> u64 {
        self.cwnd
    }
    fn bytes_in_flight(&self) -> u64 {
        self.bytes_in_flight
    }
    fn is_congestion_limited(&self) -> bool {
        self.bytes_in_flight >= self.cwnd
    }

    fn on_packet_sent(&mut self, bytes: u64, _now: Instant) {
        self.bytes_in_flight += bytes;
        self.counters.bytes_in_flight = self.bytes_in_flight;
    }

    fn on_packets_acked(
        &mut self,
        bytes_ack: u64,
        latest_rtt: Duration,
        ack_delay: Duration,
        _now: Instant,
    ) {
        self.update_rtt(latest_rtt, ack_delay);
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(bytes_ack);
        // Increase cwnd per Reno:
        //  - slow start: +ack bytes (cwnd doubles per RTT)
        //  - congestion avoidance: + MSS*MSS/cwnd per ack (approx 1 MSS per RTT)
        if self.bytes_in_flight <= self.cwnd {
            let acked_send_time = self.last_send_time;
            let _ = acked_send_time;
            if self.cwnd < self.ssthresh {
                // slow start
                self.cwnd += bytes_ack;
            } else {
                // congestion avoidance (RFC 9002 Appendix B.5)
                let mss = self.cfg.minimum_window / 2;
                self.cwnd += (mss * bytes_ack) / self.cwnd.max(1);
            }
        }
        self.counters.bytes_in_flight = self.bytes_in_flight;
        self.counters.congestion_window = self.cwnd;
        self.counters.packets_acknowledged_total += 1;
    }

    fn on_packets_lost(&mut self, bytes: u64, persistent_congestion: bool, now: Instant) {
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(bytes);
        self.counters.bytes_in_flight = self.bytes_in_flight;
        self.counters.packets_lost_total += 1;
        // Collapsing to minimum window on persistent congestion (RFC 9002 §7.6.2).
        if persistent_congestion {
            self.cwnd = self.cfg.minimum_window;
            self.ssthresh = self.cwnd;
            self.recovery_start = Some(now);
            self.counters.congestion_window = self.cwnd;
            self.counters.congestion_events += 1;
            return;
        }
        // Enter recovery only if not already in it.
        if self.recovery_start.is_none() {
            self.recovery_start = Some(now);
            self.ssthresh = (self.cwnd as f64 * self.cfg.loss_reduction_factor) as u64;
            self.ssthresh = self.ssthresh.max(self.cfg.minimum_window);
            self.cwnd = self.ssthresh;
        }
        self.counters.congestion_window = self.cwnd;
        self.counters.congestion_events += 1;
    }

    fn on_ecn_ce(&mut self, _ce_count: u64, now: Instant) {
        // Treat ECN-CE like a loss for congestion response (RFC 9002 §7.5).
        if self.recovery_start.is_none() {
            self.recovery_start = Some(now);
            self.ssthresh = (self.cwnd as f64 * self.cfg.loss_reduction_factor) as u64;
            self.ssthresh = self.ssthresh.max(self.cfg.minimum_window);
            self.cwnd = self.ssthresh;
        }
        self.counters.congestion_events += 1;
        self.counters.congestion_window = self.cwnd;
    }

    fn pto_timeout(&self, now: Instant, max_ack_delay: Duration) -> Option<Duration> {
        // PTO = smoothed_rtt + max(4*rttvar, granularity) + max_ack_delay
        // (RFC 9002 §6.2.1). Simplified: ignore the 1ms granularity.
        if self.smoothed_rtt == self.cfg.initial_rtt && self.min_rtt == Duration::MAX {
            // No samples: arm PTO at the initial RTT + ack_delay.
            return Some(self.cfg.initial_rtt + max_ack_delay);
        }
        let pto = self.smoothed_rtt
            + self.rttvar * 4
            + max_ack_delay;
        let _ = now;
        Some(pto)
    }

    fn stats(&self) -> CongestionStats {
        self.counters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slow_start_grows_window() {
        let mut cc = NewReno::new(NewRenoConfig::default());
        let now = Instant::now();
        cc.on_packet_sent(1200, now);
        cc.on_packets_acked(1200, Duration::from_millis(50), Duration::ZERO, now);
        assert!(cc.window() > NewRenoConfig::default().initial_window);
    }

    #[test]
    fn loss_halves_window() {
        let mut cc = NewReno::new(NewRenoConfig::default());
        let now = Instant::now();
        cc.on_packet_sent(12000, now);
        let before = cc.window();
        cc.on_packets_lost(12000, false, now);
        assert!(cc.window() < before);
    }
}
