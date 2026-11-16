//! Anomaly / attack detection for QUIC traffic.
//!
//! Runs an ONNX classifier (isolation-forest or autoencoder-reconstruction
//! style) over a rolling window of per-connection statistics to flag
//! anomalous patterns that suggest amplification attacks, scanners, or
//! side-channel probing. Falls back to a deterministic EWMA + threshold
//! heuristic when no model is loaded.
//!
//! Features used (per connection, sampled every `window`):
//!   0. packet rate (pps)
//!   1. byte rate (Bps)
//!   2. fraction of initial packets (handshake flood signal)
//!   3. average inter-arrival time (us)
//!   4. stddev of inter-arrival (us)
//!   5. unique DCID count in window (CID rotation attack)
//!   6. ratio of ack-eliciting to non-ack-eliciting
//!   7. handshake completion rate

use crate::ort_backend::{InferenceBackend, PredictInput};

/// One sampled feature vector for a connection.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrafficFeatures {
    pub pps: f32,
    pub bps: f32,
    pub init_frac: f32,
    pub inter_arrival_us: f32,
    pub inter_arrival_std_us: f32,
    pub unique_dcids: f32,
    pub ack_eliciting_ratio: f32,
    pub handshake_completion: f32,
}

impl TrafficFeatures {
    pub fn as_array(&self) -> [f32; 8] {
        [
            self.pps,
            self.bps,
            self.init_frac,
            self.inter_arrival_us,
            self.inter_arrival_std_us,
            self.unique_dcids,
            self.ack_eliciting_ratio,
            self.handshake_completion,
        ]
    }
}

/// Verdict returned by the detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Traffic looks normal.
    Benign,
    /// Suspicious; reduce rate / require address validation.
    Suspicious,
    /// Very likely an attack; close connection and add rate limit.
    Malicious,
}

/// Configuration.
#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    /// Score above which we mark Suspicious.
    pub suspicious_threshold: f32,
    /// Score above which we mark Malicious.
    pub malicious_threshold: f32,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            suspicious_threshold: 0.6,
            malicious_threshold: 0.85,
        }
    }
}

/// The detector. Owns an optional ONNX model; if absent, uses EWMA heuristics.
pub struct AnomalyDetector {
    cfg: AnomalyConfig,
    model: Option<Box<dyn InferenceBackend>>,
    /// EWMA-smoothed feature vector for the heuristic fallback.
    ewma: [f32; 8],
    alpha: f32,
    /// Baseline "normal" vector for the heuristic distance.
    baseline: [f32; 8],
}

impl AnomalyDetector {
    pub fn new(cfg: AnomalyConfig, model: Option<Box<dyn InferenceBackend>>) -> Self {
        Self {
            cfg,
            model,
            ewma: [0.0; 8],
            alpha: 0.2,
            baseline: [
                50.0, 20_000.0, 0.02, 20_000.0, 5_000.0, 1.0, 0.5, 0.95,
            ],
        }
    }

    /// Update with the latest features and return a verdict plus a raw score
    /// in `[0, 1]`.
    pub fn observe(&mut self, f: TrafficFeatures) -> (Verdict, f32) {
        let arr = f.as_array();
        // Update EWMA.
        for i in 0..8 {
            self.ewma[i] = self.alpha * arr[i] + (1.0 - self.alpha) * self.ewma[i];
        }
        let score = if let Some(model) = &self.model {
            // ONNX model takes a [1, 8] f32 tensor and returns an anomaly
            // score in [0, 1].
            let input = PredictInput {
                features: self.ewma,
            };
            let out = model.predict_one(&input).unwrap_or(0.0);
            out
        } else {
            // Heuristic: normalized distance from baseline, max-pooled.
            let mut max_ratio = 0.0f32;
            for i in 0..8 {
                let b = self.baseline[i].abs().max(1e-3);
                let r = (self.ewma[i] - self.baseline[i]).abs() / b;
                if r > max_ratio {
                    max_ratio = r;
                }
            }
            // Saturate at 5x baseline deviation.
            (max_ratio / 5.0).min(1.0)
        };
        let verdict = if score >= self.cfg.malicious_threshold {
            Verdict::Malicious
        } else if score >= self.cfg.suspicious_threshold {
            Verdict::Suspicious
        } else {
            Verdict::Benign
        };
        (verdict, score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benign_baseline_low_score() {
        let mut d = AnomalyDetector::new(AnomalyConfig::default(), None);
        let f = TrafficFeatures {
            pps: 50.0,
            bps: 20_000.0,
            init_frac: 0.02,
            inter_arrival_us: 20_000.0,
            inter_arrival_std_us: 5_000.0,
            unique_dcids: 1.0,
            ack_eliciting_ratio: 0.5,
            handshake_completion: 0.95,
        };
        // Warm up the EWMA.
        for _ in 0..20 {
            d.observe(f);
        }
        let (v, s) = d.observe(f);
        assert_eq!(v, Verdict::Benign);
        assert!(s < 0.6);
    }

    #[test]
    fn flood_flags_malicious() {
        let mut d = AnomalyDetector::new(AnomalyConfig::default(), None);
        let flood = TrafficFeatures {
            pps: 50_000.0,
            bps: 5_000_000.0,
            init_frac: 1.0,
            inter_arrival_us: 10.0,
            inter_arrival_std_us: 1.0,
            unique_dcids: 10_000.0,
            ack_eliciting_ratio: 0.0,
            handshake_completion: 0.0,
        };
        for _ in 0..20 {
            d.observe(flood);
        }
        let (v, s) = d.observe(flood);
        assert_eq!(v, Verdict::Malicious);
        assert!(s > 0.85);
    }
}
