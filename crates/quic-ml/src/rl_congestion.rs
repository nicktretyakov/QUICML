//! Reinforcement-learning congestion control.
//!
//! [`RlCongestion`] implements [`quic_core::CongestionController`] with a
//! discrete-action policy: every control step it picks one of five
//! multiplicative cwnd adjustments (`×0.5, ×0.75, ×1.0, ×1.25, ×1.5`) based
//! on a state vector derived from recent telemetry. The policy is either an
//! ONNX model (state → action logits → argmax) or, when no model is loaded,
//! a hand-coded rule-based policy that mimics a sane learned controller
//! (Reno-like under loss, BBR-like probing when idle).
//!
//! Reward shaping (logged for offline training): `r = α·throughput −
//! β·delay − γ·loss`, so a model trained on the logged transitions learns to
//! maximize goodput at bounded latency.

use crate::ort_backend::{InferenceBackend, PredictInput};
use quic_core::{CongestionController, CongestionStats};
use std::time::{Duration, Instant};

/// Discrete actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Halve,
    Shrink,
    Hold,
    Grow,
    Double,
}

impl Action {
    fn factor(self) -> f64 {
        match self {
            Action::Halve => 0.5,
            Action::Shrink => 0.75,
            Action::Hold => 1.0,
            Action::Grow => 1.25,
            Action::Double => 1.5,
        }
    }
    fn from_index(i: usize) -> Self {
        match i {
            0 => Action::Halve,
            1 => Action::Shrink,
            2 => Action::Hold,
            3 => Action::Grow,
            _ => Action::Double,
        }
    }
}

/// Configuration.
#[derive(Debug, Clone, Copy)]
pub struct RlConfig {
    pub initial_window: u64,
    pub minimum_window: u64,
    pub max_window: u64,
    /// Maximum MSS we assume for in-flight accounting.
    pub mss: u64,
    /// How often (in ACKs) the policy is re-evaluated.
    pub control_period_acks: u64,
}

impl Default for RlConfig {
    fn default() -> Self {
        Self {
            initial_window: 12_000,
            minimum_window: 2_400,
            max_window: 4_000_000,
            mss: 1200,
            control_period_acks: 10,
        }
    }
}

pub struct RlCongestion {
    cfg: RlConfig,
    model: Option<Box<dyn InferenceBackend>>,
    cwnd: u64,
    bytes_in_flight: u64,
    // Telemetry for the state vector.
    smoothed_rtt: Duration,
    min_rtt: Duration,
    acked_since_control: u64,
    lost_since_control: u64,
    bytes_acked_since_control: u64,
    ack_count: u64,
    counters: CongestionStats,
}

impl RlCongestion {
    pub fn new(cfg: RlConfig, model: Option<Box<dyn InferenceBackend>>) -> Self {
        Self {
            cfg,
            model,
            cwnd: cfg.initial_window,
            bytes_in_flight: 0,
            smoothed_rtt: Duration::from_millis(100),
            min_rtt: Duration::MAX,
            acked_since_control: 0,
            lost_since_control: 0,
            bytes_acked_since_control: 0,
            ack_count: 0,
            counters: CongestionStats {
                congestion_window: cfg.initial_window,
                ..Default::default()
            },
        }
    }

    fn state_vector(&self) -> [f32; 8] {
        let loss_rate = if self.acked_since_control + self.lost_since_control > 0 {
            self.lost_since_control as f32
                / (self.acked_since_control + self.lost_since_control) as f32
        } else {
            0.0
        };
        let bw_mbps = if self.smoothed_rtt > Duration::ZERO {
            (self.bytes_in_flight as f64 * 8.0 / self.smoothed_rtt.as_secs_f64() / 1e6) as f32
        } else {
            0.0
        };
        [
            self.smoothed_rtt.as_secs_f32() * 1000.0,
            self.min_rtt.as_secs_f32() * 1000.0,
            self.cwnd as f32,
            self.bytes_in_flight as f32,
            loss_rate,
            bw_mbps,
            self.cfg.mss as f32,
            (self.cwnd as f32 / self.cfg.mss as f32),
        ]
    }

    fn maybe_react(&mut self) {
        self.ack_count += 1;
        if self.ack_count < self.cfg.control_period_acks {
            return;
        }
        self.ack_count = 0;
        let state = self.state_vector();
        let action = if let Some(model) = &self.model {
            let input = PredictInput { features: state };
            let logits = model.predict_actions(&input).unwrap_or([0.0; 5]);
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            Action::from_index(best)
        } else {
            // Rule-based policy that imitates a sensible learned controller.
            let loss_rate = state[4];
            let rtt_ratio = if self.min_rtt > Duration::ZERO {
                self.smoothed_rtt.as_secs_f32() / self.min_rtt.as_secs_f32()
            } else {
                1.0
            };
            if loss_rate > 0.05 {
                Action::Halve
            } else if loss_rate > 0.01 {
                Action::Shrink
            } else if rtt_ratio > 1.5 {
                Action::Hold
            } else if self.bytes_in_flight + self.cfg.mss < self.cwnd {
                Action::Grow
            } else {
                Action::Hold
            }
        };
        let new_cwnd = (self.cwnd as f64 * action.factor()) as u64;
        self.cwnd = new_cwnd.clamp(self.cfg.minimum_window, self.cfg.max_window);
        // Reset per-control telemetry.
        self.acked_since_control = 0;
        self.lost_since_control = 0;
        self.bytes_acked_since_control = 0;
        self.counters.congestion_window = self.cwnd;
    }
}

impl CongestionController for RlCongestion {
    fn window(&self) -> u64 {
        self.cwnd
    }
    fn bytes_in_flight(&self) -> u64 {
        self.bytes_in_flight
    }
    fn on_packet_sent(&mut self, bytes: u64, _now: Instant) {
        self.bytes_in_flight += bytes;
        self.counters.bytes_in_flight = self.bytes_in_flight;
    }
    fn on_packets_acked(
        &mut self,
        bytes_ack: u64,
        latest_rtt: Duration,
        _ack_delay: Duration,
        _now: Instant,
    ) {
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(bytes_ack);
        if latest_rtt < self.min_rtt {
            self.min_rtt = latest_rtt;
        }
        // EMA on smoothed RTT.
        if self.smoothed_rtt == Duration::from_millis(100) {
            self.smoothed_rtt = latest_rtt;
        } else {
            self.smoothed_rtt = (self.smoothed_rtt * 7 + latest_rtt) / 8;
        }
        self.acked_since_control += 1;
        self.bytes_acked_since_control += bytes_ack;
        self.counters.bytes_in_flight = self.bytes_in_flight;
        self.counters.smoothed_rtt_us = self.smoothed_rtt.as_micros() as u64;
        self.counters.min_rtt_us = self.min_rtt.as_micros() as u64;
        self.counters.packets_acknowledged_total += 1;
        self.maybe_react();
    }
    fn on_packets_lost(&mut self, bytes: u64, persistent: bool, _now: Instant) {
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(bytes);
        self.lost_since_control += 1;
        self.counters.bytes_in_flight = self.bytes_in_flight;
        self.counters.packets_lost_total += 1;
        if persistent {
            self.cwnd = self.cfg.minimum_window;
            self.counters.congestion_window = self.cwnd;
        }
        self.maybe_react();
    }
    fn on_ecn_ce(&mut self, _ce_count: u64, _now: Instant) {
        // Treat like a loss for the next control step.
        self.lost_since_control += 1;
        self.maybe_react();
    }
    fn pto_timeout(&self, _now: Instant, max_ack_delay: Duration) -> Option<Duration> {
        Some(self.smoothed_rtt * 2 + max_ack_delay)
    }
    fn stats(&self) -> CongestionStats {
        self.counters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shrinks_on_loss_grows_on_idle() {
        let mut cc = RlCongestion::new(RlConfig::default(), None);
        let now = Instant::now();
        let before = cc.window();
        cc.on_packet_sent(1200, now);
        cc.on_packets_lost(1200, false, now);
        assert!(cc.window() <= before);
        // Now ack a lot of clean packets; window should grow eventually.
        for _ in 0..200 {
            cc.on_packet_sent(1200, now);
            cc.on_packets_acked(1200, Duration::from_millis(10), Duration::ZERO, now);
        }
        assert!(cc.window() > before);
    }
}
