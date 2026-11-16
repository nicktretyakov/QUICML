//! MTU prediction (RFC 8899 DPLPMTUD augmented with an ML predictor).
//!
//! Standard DPLPMTUD probes upward in fixed steps and backs off on loss. An
//! ML model trained on path features (loss rate per size, RTT, hop count
//! inferred from TTL) can skip most probes by predicting the largest size
//! likely to succeed. This module exposes [`MtuPredictor`] which:
//!
//! 1. Predicts a candidate MTU from features via ONNX (or a heuristic).
//! 2. Tracks confirmation from real probe successes / failures.
//! 3. Exposes the current *confirmed* MTU for the packetizer.

use crate::ort_backend::{InferenceBackend, PredictInput};

/// Per-path features for MTU prediction.
#[derive(Debug, Clone, Copy, Default)]
pub struct MtuFeatures {
    /// Loss rate observed at 1200-byte packets.
    pub loss_rate_1200: f32,
    /// Loss rate at 1400 bytes.
    pub loss_rate_1400: f32,
    /// Smoothed RTT in ms.
    pub srtt_ms: f32,
    /// TTL observed (rough hop proxy).
    pub ttl: f32,
    /// IPv4 vs IPv6 (1.0 = v4, 0.0 = v6).
    pub ipv4: f32,
    /// Last confirmed MTU.
    pub last_mtu: f32,
}

impl MtuFeatures {
    fn as_array(&self) -> [f32; 6] {
        [
            self.loss_rate_1200,
            self.loss_rate_1400,
            self.srtt_ms,
            self.ttl,
            self.ipv4,
            self.last_mtu,
        ]
    }
}

pub struct MtuPredictor {
    model: Option<Box<dyn InferenceBackend>>,
    /// Currently confirmed MTU (largest size known to reach the peer).
    confirmed: u16,
    /// Candidate we are currently probing (0 = none).
    probing: u16,
    /// Minimum MTU we ever return (RFC 9000: 1200).
    floor: u16,
    /// Hard ceiling.
    ceiling: u16,
}

impl MtuPredictor {
    pub fn new(model: Option<Box<dyn InferenceBackend>>) -> Self {
        Self {
            model,
            confirmed: 1200,
            probing: 0,
            floor: 1200,
            ceiling: 1454,
        }
    }

    /// Predict the next probe target. Returns the MTU to try, or `None` if we
    /// are already at the ceiling.
    pub fn predict_probe(&mut self, f: MtuFeatures) -> Option<u16> {
        if self.confirmed >= self.ceiling {
            return None;
        }
        let candidate: u16 = if let Some(model) = &self.model {
            let input = PredictInput {
                features: pad6_to_8(f.as_array()),
            };
            let raw = model.predict_one(&input).unwrap_or(self.confirmed as f32);
            raw.round().clamp(self.floor as f32, self.ceiling as f32) as u16
        } else {
            // Heuristic: step by 70 bytes above confirmed unless loss is high.
            let step = if f.loss_rate_1400 > 0.05 { 0 } else { 70 };
            (self.confirmed + step).min(self.ceiling)
        };
        self.probing = candidate;
        Some(candidate)
    }

    /// Record a successful probe at the given size.
    pub fn on_probe_success(&mut self, size: u16) {
        if size > self.confirmed {
            self.confirmed = size;
        }
        self.probing = 0;
    }

    /// Record a failed probe; back off to the confirmed size.
    pub fn on_probe_failure(&mut self, _size: u16) {
        self.probing = 0;
    }

    pub fn confirmed_mtu(&self) -> u16 {
        self.confirmed
    }
}

fn pad6_to_8(a: [f32; 6]) -> [f32; 8] {
    let mut out = [0.0; 8];
    out[..6].copy_from_slice(&a);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_advances_when_no_loss() {
        let mut p = MtuPredictor::new(None);
        let f = MtuFeatures {
            loss_rate_1200: 0.0,
            loss_rate_1400: 0.0,
            srtt_ms: 30.0,
            ttl: 64.0,
            ipv4: 1.0,
            last_mtu: 1200.0,
        };
        let target = p.predict_probe(f).unwrap();
        assert!(target > 1200);
        p.on_probe_success(target);
        assert_eq!(p.confirmed_mtu(), target);
    }

    #[test]
    fn ceiling_caps_growth() {
        let mut p = MtuPredictor::new(None);
        p.confirmed = p.ceiling;
        assert!(p.predict_probe(MtuFeatures::default()).is_none());
    }
}
