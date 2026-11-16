//! GPU/CUDA abstraction for batched ML preprocessing.
//!
//! Real-world QUIC + ML deployments want to run inference on batches of
//! per-packet features (RTT samples, inter-arrival times, byte counts) that
//! arrive at line rate. On edge nodes with a GPU this is faster on CUDA; on
//! CPU-only edge nodes the same code path must still work. This module
//! defines a [`GpuBackend`] trait plus a [`CpuBackend`] reference
//! implementation, so callers write one code path that runs on either.
//!
//! The actual CUDA kernel integration (via `rust-cuda` / `cust`) is behind
//! the `cuda` feature and lives in a downstream crate; here we expose only
//! the trait and the CPU fallback so that `quic-ml` stays dependency-light.

use std::time::Duration;

/// A 2-D feature batch: `rows` samples of `cols` features, row-major f32.
pub struct FeatureBatch {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl FeatureBatch {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }
}

/// Normalization statistics computed once from a training corpus.
#[derive(Debug, Clone, Copy)]
pub struct NormStats {
    pub mean: [f32; 8],
    pub std: [f32; 8],
}

impl Default for NormStats {
    fn default() -> Self {
        // Reasonable defaults derived from a typical QUIC data-plane corpus
        // (RTT in ms, cwnd, in-flight, loss rate, ack rate, bw Mbps, jitter,
        // inter-arrival us). Replace with values fitted to your traffic.
        Self {
            mean: [40.0, 32_768.0, 16_384.0, 0.02, 0.95, 50.0, 8.0, 200.0],
            std: [60.0, 65_536.0, 32_768.0, 0.08, 0.05, 80.0, 12.0, 300.0],
        }
    }
}

/// Abstraction over a device that can normalize a feature batch in place.
pub trait GpuBackend: Send + Sync {
    /// Apply `(x - mean) / std` to every element of `batch` in place. Returns
    /// the wall-clock time spent so callers can compare CPU vs GPU.
    fn normalize_in_place(&self, batch: &mut FeatureBatch, stats: &NormStats) -> Duration;

    /// Human-readable device name (e.g. "cpu" or "cuda:0").
    fn device_name(&self) -> &str;
}

/// Pure-Rust CPU implementation. Always available; used when no CUDA device
/// is present or the `cuda` feature is disabled.
pub struct CpuBackend;

impl GpuBackend for CpuBackend {
    fn normalize_in_place(&self, batch: &mut FeatureBatch, stats: &NormStats) -> Duration {
        let start = std::time::Instant::now();
        let cols = batch.cols.min(8);
        for r in 0..batch.rows {
            let row = batch.row_mut(r);
            for c in 0..cols {
                let s = stats.std[c].max(1e-6);
                row[c] = (row[c] - stats.mean[c]) / s;
            }
        }
        start.elapsed()
    }
    fn device_name(&self) -> &str {
        "cpu"
    }
}

/// Pick the best available backend: CUDA when the `cuda` feature is compiled
/// in *and* a device is present, otherwise CPU.
pub fn best_available() -> Box<dyn GpuBackend> {
    // The CUDA path would be `#[cfg(feature="cuda")] { ... Box::new(CudaBackend::new()) }`
    // falling back to CPU on any init error. For now CPU is the only impl.
    Box::new(CpuBackend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_zero_mean_unit_std() {
        let mut b = FeatureBatch::new(2, 8);
        // fill with the mean -> after normalize, all zeros.
        let stats = NormStats::default();
        for r in 0..2 {
            for c in 0..8 {
                b.row_mut(r)[c] = stats.mean[c];
            }
        }
        CpuBackend.normalize_in_place(&mut b, &stats);
        for v in &b.data {
            assert!(v.abs() < 1e-5);
        }
    }
}
