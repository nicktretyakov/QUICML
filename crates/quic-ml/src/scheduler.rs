//! ML-driven packet scheduling.
//!
//! When many streams are ready to send and the congestion window only allows
//! a subset, the scheduler decides which bytes go first. Classic QUIC uses
//! round-robin or FIFO; an ML scheduler ranks streams by predicted
//! "urgency × importance" derived from recent behavior (stream age, bytes
//! already sent, observed app-level latency, priority hints).
//!
//! [`MlScheduler`] keeps per-stream statistics and produces an ordered list
//! of `(stream_id, bytes)` to send for the current congestion-window budget.

use std::collections::HashMap;

/// Per-stream scheduling statistics, updated on every send/ack.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamStats {
    pub bytes_sent: u64,
    pub bytes_acked: u64,
    pub bytes_lost: u64,
    /// Application-supplied priority hint in 0..=255 (higher = more urgent).
    pub priority: u8,
    /// Milliseconds since the stream was first written.
    pub age_ms: u32,
    /// Observed end-to-end latency of acked bytes (ms), EMA.
    pub observed_latency_ms: f32,
}

/// Scheduler entry returned by [`MlScheduler::schedule`].
#[derive(Debug, Clone, Copy)]
pub struct ScheduledSend {
    pub stream_id: u64,
    pub bytes: usize,
}

pub struct MlScheduler {
    stats: HashMap<u64, StreamStats>,
    /// Weight vector for the linear scoring model. A trained ONNX model
    /// would replace this; we expose the weights so the caller can fit them
    /// offline and hot-swap.
    weights: SchedulerWeights,
    /// Whether to use a learned model instead of the linear scorer. Reserved
    /// for the ONNX backend; the linear scorer is the working default.
    use_model: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct SchedulerWeights {
    pub priority: f32,
    pub age: f32,
    pub loss_rate: f32,
    pub bytes_in_flight: f32,
}

impl Default for SchedulerWeights {
    fn default() -> Self {
        // Favor high priority, older streams, low-loss streams, and streams
        // with little already in flight.
        Self {
            priority: 2.0,
            age: 0.01,
            loss_rate: -3.0,
            bytes_in_flight: -0.0001,
        }
    }
}

impl MlScheduler {
    pub fn new(weights: SchedulerWeights, use_model: bool) -> Self {
        Self {
            stats: HashMap::new(),
            weights,
            use_model,
        }
    }

    pub fn observe(&mut self, stream_id: u64, update: impl FnOnce(&mut StreamStats)) {
        let s = self.stats.entry(stream_id).or_default();
        update(s);
    }

    /// Given a set of streams with pending bytes and a congestion-window
    /// budget, return the ordered send plan.
    pub fn schedule(
        &self,
        pending: &[(u64, usize)],
        budget_bytes: usize,
    ) -> Vec<ScheduledSend> {
        let _ = self.use_model; // ONNX path would call model.predict here.
        let mut scored: Vec<(f32, u64, usize)> = pending
            .iter()
            .map(|&(id, bytes)| {
                let s = self.stats.get(&id).copied().unwrap_or_default();
                let loss_rate = if s.bytes_sent > 0 {
                    s.bytes_lost as f32 / s.bytes_sent as f32
                } else {
                    0.0
                };
                let in_flight = s.bytes_sent.saturating_sub(s.bytes_acked);
                let score = self.weights.priority * s.priority as f32
                    + self.weights.age * s.age_ms as f32
                    + self.weights.loss_rate * loss_rate
                    + self.weights.bytes_in_flight * in_flight as f32;
                (score, id, bytes)
            })
            .collect();
        // Highest score first.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = Vec::new();
        let mut used = 0usize;
        for (_score, id, bytes) in scored {
            if used >= budget_bytes {
                break;
            }
            let take = bytes.min(budget_bytes - used);
            out.push(ScheduledSend {
                stream_id: id,
                bytes: take,
            });
            used += take;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn higher_priority_sends_first() {
        let mut sched = MlScheduler::new(SchedulerWeights::default(), false);
        sched.observe(0, |s| s.priority = 10);
        sched.observe(4, |s| s.priority = 200);
        let pending = vec![(0, 1000), (4, 1000)];
        let plan = sched.schedule(&pending, 1500);
        assert_eq!(plan[0].stream_id, 4);
    }

    #[test]
    fn budget_is_respected() {
        let sched = MlScheduler::new(SchedulerWeights::default(), false);
        let pending = vec![(0, 5000), (4, 5000)];
        let plan = sched.schedule(&pending, 1200);
        let total: usize = plan.iter().map(|s| s.bytes).sum();
        assert!(total <= 1200);
    }
}
