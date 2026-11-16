//! LSTM-based RTT prediction.
//!
//! A small recurrent model trained on per-connection RTT time series
//! predicts the RTT one probe ahead, which lets the connection set the PTO
//! timer more aggressively than the conservative `smoothed_rtt + 4*rttvar`
//! bound in RFC 9002. When no ONNX model is available, we fall back to a
//! double-exponential-smoothing predictor (Holt's method), which captures
//! both level and trend and is substantially better than plain EWMA under
//! drift.
//!
//! Model contract (input name `"x"`, shape `[1, T, F]` with `T = 16`,
//! `F = 1`; output name `"y"`, shape `[1, 1]`): the predicted RTT in
//! milliseconds for the next sample.

use crate::ort_backend::{InferenceBackend, PredictInput};
use std::time::Duration;

const WINDOW: usize = 16;

/// Predicted RTT plus a confidence band (one standard deviation).
#[derive(Debug, Clone, Copy)]
pub struct PredictedRtt {
    pub mean: Duration,
    pub std: Duration,
}

/// LSTM / DES predictor.
pub struct RttPredictor {
    model: Option<Box<dyn InferenceBackend>>,
    /// Rolling window of recent RTT samples in milliseconds.
    samples: [f32; WINDOW],
    /// Number of valid samples (so we don't predict on garbage).
    filled: usize,
    /// Holt's method state (fallback).
    level: f32,
    trend: f32,
    alpha: f32,
    beta: f32,
}

impl RttPredictor {
    pub fn new(model: Option<Box<dyn InferenceBackend>>) -> Self {
        Self {
            model,
            samples: [0.0; WINDOW],
            filled: 0,
            level: 0.0,
            trend: 0.0,
            alpha: 0.3,
            beta: 0.1,
        }
    }

    /// Push a new RTT sample.
    pub fn observe(&mut self, rtt: Duration) {
        let ms = rtt.as_secs_f32() * 1000.0;
        // Holt's double exponential smoothing update.
        if self.filled == 0 {
            self.level = ms;
            self.trend = 0.0;
        } else {
            let new_level = self.alpha * ms + (1.0 - self.alpha) * (self.level + self.trend);
            self.trend = self.beta * (new_level - self.level) + (1.0 - self.beta) * self.trend;
            self.level = new_level;
        }
        // Slide the window.
        if self.filled < WINDOW {
            self.samples[self.filled] = ms;
            self.filled += 1;
        } else {
            self.samples.rotate_left(1);
            self.samples[WINDOW - 1] = ms;
        }
    }

    /// Predict the next RTT. Returns `None` until at least one sample is in.
    pub fn predict(&mut self) -> Option<PredictedRtt> {
        if self.filled == 0 {
            return None;
        }
        let (mean_ms, std_ms) = if let Some(model) = &self.model {
            // Feed the window (padded with the latest sample) as the feature
            // row; the model returns the predicted next-RTT in ms.
            let mut feats = [0.0; 8];
            // Down-sample the 16-wide window into 8 features by averaging
            // pairs, so the feature width matches the backend's contract.
            for i in 0..8 {
                let a = self.samples[2 * i];
                let b = self.samples[2 * i + 1];
                feats[i] = (a + b) * 0.5;
            }
            let input = PredictInput { features: feats };
            let m = model.predict_one(&input).unwrap_or(self.level);
            // Estimate std as the recent sample stddev.
            let s = self.recent_stddev();
            (m, s)
        } else {
            // Holt's one-step forecast = level + trend.
            let m = self.level + self.trend;
            let s = self.recent_stddev();
            (m, s)
        };
        Some(PredictedRtt {
            mean: Duration::from_secs_f32(mean_ms / 1000.0),
            std: Duration::from_secs_f32(std_ms / 1000.0),
        })
    }

    fn recent_stddev(&self) -> f32 {
        if self.filled < 2 {
            return 0.0;
        }
        let n = self.filled as f32;
        let mean: f32 = self.samples[..self.filled].iter().copied().sum::<f32>() / n;
        let var: f32 = self.samples[..self.filled]
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / n;
        var.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_trend_without_model() {
        let mut p = RttPredictor::new(None);
        // Increasing RTT: the trend term should make the forecast exceed
        // the last sample.
        for i in 0..20 {
            p.observe(Duration::from_millis(10 + i));
        }
        let pred = p.predict().unwrap();
        assert!(pred.mean > Duration::from_millis(29));
    }
}
