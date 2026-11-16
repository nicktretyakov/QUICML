//! `quic-ml` — ML-driven enhancements for `quic-core`.
//!
//! This crate provides plug-in ML predictors that consume telemetry from a
//! running QUIC connection and either drive a control decision (congestion
//! control, scheduling, MTU) or raise a signal (anomaly / attack detection).
//! Every predictor has a trained-ONNX path and a deterministic fallback so
//! the crate is useful with or without model files.
//!
//! # Modules
//!
//! - [`ort_backend`] — the [`InferenceBackend`] trait + ONNX (`ort`) impl.
//! - [`cuda`] — GPU/CUDA abstraction trait with a CPU fallback.
//! - [`rtt_predictor`] — LSTM / double-exponential-smoothing RTT forecast.
//! - [`rl_congestion`] — RL congestion controller implementing
//!   `quic_core::CongestionController`.
//! - [`anomaly`] — anomaly / attack detection over traffic features.
//! - [`mtu_predictor`] — ML-guided path MTU probing.
//! - [`scheduler`] — ML-scored stream scheduling within the cwnd budget.

#![warn(missing_docs)]

pub mod anomaly;
pub mod cuda;
pub mod mtu_predictor;
pub mod ort_backend;
pub mod rtt_predictor;
pub mod rl_congestion;
pub mod scheduler;

pub use anomaly::{AnomalyConfig, AnomalyDetector, TrafficFeatures, Verdict};
pub use cuda::{best_available, CpuBackend, FeatureBatch, GpuBackend, NormStats};
pub use mtu_predictor::{MtuFeatures, MtuPredictor};
pub use ort_backend::{BackendError, InferenceBackend, NoBackend, PredictInput};
#[cfg(feature = "onnx")]
pub use ort_backend::OnnxBackend;
pub use rtt_predictor::{PredictedRtt, RttPredictor};
pub use rl_congestion::{RlConfig, RlCongestion};
pub use scheduler::{MlScheduler, ScheduledSend, SchedulerWeights, StreamStats};
