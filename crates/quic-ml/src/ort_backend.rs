//! ONNX Runtime backend.
//!
//! Every ML predictor in this crate talks to a model through the
//! [`InferenceBackend`] trait, so that the same predictor code runs against
//! either a real ONNX model ([`OnnxBackend`], feature `onnx`) or a
//! heuristic fallback ([`NoBackend`]) when no model file is present.
//!
//! Model contracts:
//! - **Regression** (RTT, MTU, anomaly score): input `features` shape
//!   `[1, 8]` f32, output `score` shape `[1, 1]` f32.
//! - **RL policy**: input `features` shape `[1, 8]` f32, output `logits`
//!   shape `[1, 5]` f32.

use thiserror::Error;

/// Input feature row for a single inference call.
#[derive(Debug, Clone, Copy)]
pub struct PredictInput {
    pub features: [f32; 8],
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("onnx backend not available (no model loaded)")]
    Unavailable,
    #[error("model io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ort inference error: {0}")]
    Inference(String),
}

/// Trait every model backend implements.
pub trait InferenceBackend: Send + Sync {
    /// Run a regression model and return the scalar output.
    fn predict_one(&self, input: &PredictInput) -> Result<f32, BackendError>;
    /// Run an RL policy model and return the per-action logits.
    fn predict_actions(&self, input: &PredictInput) -> Result<[f32; 5], BackendError>;
}

/// Fallback backend with no model — always returns `Unavailable`. Predictors
/// handle this by falling back to their heuristic path.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoBackend;

impl InferenceBackend for NoBackend {
    fn predict_one(&self, _input: &PredictInput) -> Result<f32, BackendError> {
        Err(BackendError::Unavailable)
    }
    fn predict_actions(&self, _input: &PredictInput) -> Result<[f32; 5], BackendError> {
        Err(BackendError::Unavailable)
    }
}

#[cfg(feature = "onnx")]
pub mod onnx {
    //! ONNX Runtime-backed implementation (ort 2.0.0-rc.x).

    use super::{BackendError, InferenceBackend, PredictInput};
    use ndarray::Array2;
    use ort::environment::Environment;
    use ort::session::builder::GraphOptimizationLevel;
    use ort::session::{Session, SessionOutputs};
    use ort::value::Tensor;
    use parking_lot::Mutex;
    use std::path::Path;

    /// A pair of ONNX sessions: one regression model and one optional RL
    /// policy model. If the policy model is `None`, `predict_actions` falls
    /// back to `predict_one` (treated as an index into the action table).
    /// I/O is positional (input #0, output #0) so we never touch the private
    /// `Session.inputs` / `Session.outputs` fields. The sessions are wrapped
    /// in a `Mutex` because `Session::run` requires `&mut self` while the
    /// [`InferenceBackend`] trait exposes `&self` — the lock is held only for
    /// the duration of one inference call.
    pub struct OnnxBackend {
        _env: Environment,
        reg: Mutex<Session>,
        policy: Option<Mutex<Session>>,
    }

    impl std::fmt::Debug for OnnxBackend {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("OnnxBackend")
                .field("has_policy", &self.policy.is_some())
                .finish()
        }
    }

    impl OnnxBackend {
        /// Load a regression model from `path`. The optional policy model is
        /// loaded from `policy_path` if supplied.
        pub fn load<P: AsRef<Path>>(
            path: P,
            policy_path: Option<P>,
        ) -> Result<Self, BackendError> {
            let env = ort::init()
                .with_name("quic-ml")
                .build()
                .map_err(|e| BackendError::Inference(format!("env: {e}")))?;

            let reg = Session::builder(&env)
                .map_err(|e| BackendError::Inference(format!("builder: {e}")))?
                .with_optimization_level(GraphOptimizationLevel::Level1)
                .map_err(|e| BackendError::Inference(format!("opt: {e}")))?
                .commit_from_file(path.as_ref())
                .map_err(|e| BackendError::Inference(format!("load reg: {e}")))?;

            let policy = match policy_path {
                Some(p) => Some(
                    Session::builder(&env)
                        .map_err(|e| BackendError::Inference(format!("builder: {e}")))?
                        .with_optimization_level(GraphOptimizationLevel::Level1)
                        .map_err(|e| BackendError::Inference(format!("opt: {e}")))?
                        .commit_from_file(p.as_ref())
                        .map_err(|e| BackendError::Inference(format!("load policy: {e}")))?,
                ),
                None => None,
            };
            Ok(Self {
                _env: env,
                reg: Mutex::new(reg),
                policy: policy.map(Mutex::new),
            })
        }

        fn run_regression(&self, features: [f32; 8]) -> Result<f32, BackendError> {
            let arr: Array2<f32> =
                Array2::from_shape_vec((1, 8), features.to_vec()).expect("8 features");
            let tensor: Tensor<f32> = Tensor::from_array(arr)
                .map_err(|e| BackendError::Inference(format!("value: {e}")))?;
            let mut reg = self.reg.lock();
            let outputs: SessionOutputs =
                reg.run(ort::inputs![tensor])
                    .map_err(|e| BackendError::Inference(format!("run: {e}")))?;
            let view = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| BackendError::Inference(format!("extract: {e}")))?;
            view.iter()
                .next()
                .copied()
                .ok_or_else(|| BackendError::Inference("empty output".into()))
        }
    }

    impl InferenceBackend for OnnxBackend {
        fn predict_one(&self, input: &PredictInput) -> Result<f32, BackendError> {
            self.run_regression(input.features)
        }

        fn predict_actions(&self, input: &PredictInput) -> Result<[f32; 5], BackendError> {
            let session = match &self.policy {
                Some(s) => s,
                None => {
                    let s = self.predict_one(input)?;
                    let mut out = [s / 5.0; 5];
                    out[2] = s;
                    return Ok(out);
                }
            };
            let arr: Array2<f32> =
                Array2::from_shape_vec((1, 8), input.features.to_vec()).expect("8 features");
            let tensor: Tensor<f32> = Tensor::from_array(arr)
                .map_err(|e| BackendError::Inference(format!("value: {e}")))?;
            let mut session = session.lock();
            let outputs: SessionOutputs =
                session
                    .run(ort::inputs![tensor])
                    .map_err(|e| BackendError::Inference(format!("run: {e}")))?;
            let view = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| BackendError::Inference(format!("extract: {e}")))?;
            let mut logits = [0.0f32; 5];
            for (i, v) in view.iter().take(5).enumerate() {
                logits[i] = *v;
            }
            Ok(logits)
        }
    }
}

#[cfg(feature = "onnx")]
pub use onnx::OnnxBackend;
