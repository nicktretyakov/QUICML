//! `quic-hybrid` — dynamic QUIC + WebRTC + gRPC transport router.
//!
//! An edge node serving heterogeneous traffic (low-latency media via WebRTC,
//! RPC fan-out via gRPC, bulk transfer via QUIC streams) must pick the best
//! transport for each flow as conditions change. [`HybridRouter`] scores
//! every candidate transport against live telemetry (RTT, loss, bandwidth,
//! flow size, latency requirement) using a linear model whose weights can be
//! trained offline, and emits a routing decision.
//!
//! This is a control-plane component: it does not itself speak WebRTC or
//! gRPC, it tells a deployment which transport to bind a new flow to.

#![warn(missing_docs)]

use parking_lot::Mutex;
use std::collections::HashMap;

/// Transport options the router chooses among.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transport {
    /// QUIC streams — good for bulk, multiplexed, ordered+unordered.
    Quic,
    /// WebRTC data channels — good for sub-second-latency media and P2P.
    WebRtc,
    /// gRPC over HTTP/2 — good for structured RPC, ecosystem integration.
    Grpc,
}

impl Transport {
    fn as_index(self) -> usize {
        match self {
            Transport::Quic => 0,
            Transport::WebRtc => 1,
            Transport::Grpc => 2,
        }
    }
}

/// Per-flow request characteristics used to score transports.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlowProfile {
    /// Estimated flow size in bytes (0 if unknown).
    pub size_bytes: f32,
    /// One-way latency requirement in ms (lower = more urgent).
    pub latency_req_ms: f32,
    /// Observed path RTT in ms.
    pub rtt_ms: f32,
    /// Observed loss rate [0,1].
    pub loss_rate: f32,
    /// Available bandwidth in Mbps.
    pub bw_mbps: f32,
    /// Whether the flow is peer-to-peer (favors WebRTC).
    pub is_p2p: f32,
    /// Whether the flow needs strict ordering (favors QUIC/gRPC).
    pub needs_ordering: f32,
}

/// Score weights for the linear model, one set per transport.
#[derive(Debug, Clone)]
pub struct RouterWeights {
    /// `[transport][feature]` — 3 transports × 7 features.
    pub w: [[f32; 7]; 3],
    pub bias: [f32; 3],
}

impl Default for RouterWeights {
    fn default() -> Self {
        // Hand-tuned priors; replace with weights fit on deployment telemetry.
        // Features: [size, latency_req, rtt, loss, bw, is_p2p, needs_ordering]
        Self {
            w: [
                // QUIC: likes big flows, ordering, bandwidth.
                [0.0001, -0.02, -0.01, -2.0, 0.5, -1.0, 1.5],
                // WebRTC: likes p2p, low latency req, tolerates loss.
                [-0.0001, 0.05, -0.005, 0.5, 0.2, 3.0, -1.0],
                // gRPC: likes ordering, moderate sizes, low p2p.
                [0.00005, -0.01, -0.008, -3.0, 0.3, -2.0, 2.0],
            ],
            bias: [0.2, 0.0, 0.1],
        }
    }
}

/// A routing decision.
#[derive(Debug, Clone, Copy)]
pub struct RoutingDecision {
    pub transport: Transport,
    pub scores: [f32; 3],
}

/// The router.
pub struct HybridRouter {
    weights: RouterWeights,
    /// Optional ML override: when set, the linear scorer is replaced by a
    /// learned policy. Kept behind a trait object so the crate stays
    /// dependency-light.
    learned: Mutex<Option<Box<dyn LearnedRouter>>>,
    /// Per-transport current load (0..1), used to avoid overloading one.
    load: Mutex<HashMap<Transport, f32>>,
}

/// A learned router policy (e.g. an ONNX model mapping features → logits).
/// Implement this and call [`HybridRouter::set_learned`] to hot-swap.
pub trait LearnedRouter: Send + Sync {
    /// Returns raw per-transport scores; the argmax wins.
    fn score(&self, flow: &FlowProfile) -> [f32; 3];
}

impl HybridRouter {
    /// Construct with the given weights and no learned policy.
    pub fn new(weights: RouterWeights) -> Self {
        Self {
            weights,
            learned: Mutex::new(None),
            load: Mutex::new(HashMap::new()),
        }
    }

    /// Install a learned router policy.
    pub fn set_learned(&self, policy: Box<dyn LearnedRouter>) {
        *self.learned.lock() = Some(policy);
    }

    /// Report a transport's current load (0..1) so the router can balance.
    pub fn report_load(&self, t: Transport, load: f32) {
        self.load.lock().insert(t, load.clamp(0.0, 1.0));
    }

    /// Score every transport and pick the best, applying a load penalty.
    pub fn route(&self, flow: &FlowProfile) -> RoutingDecision {
        let mut scores = if let Some(policy) = self.learned.lock().as_ref() {
            policy.score(flow)
        } else {
            let f = [
                flow.size_bytes,
                flow.latency_req_ms,
                flow.rtt_ms,
                flow.loss_rate,
                flow.bw_mbps,
                flow.is_p2p,
                flow.needs_ordering,
            ];
            let mut s = self.weights.bias;
            for t in 0..3 {
                for (i, &fv) in f.iter().enumerate() {
                    s[t] += self.weights.w[t][i] * fv;
                }
            }
            s
        };
        // Load gating: a saturated transport is de-scored proportionally to
        // its load, so a 100%-loaded transport is effectively excluded
        // regardless of how well the features otherwise fit.
        let loads = self.load.lock();
        for t in [Transport::Quic, Transport::WebRtc, Transport::Grpc] {
            if let Some(&l) = loads.get(&t) {
                scores[t.as_index()] *= 1.0 - l.clamp(0.0, 1.0);
            }
        }
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in scores.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        RoutingDecision {
            transport: match best {
                0 => Transport::Quic,
                1 => Transport::WebRtc,
                _ => Transport::Grpc,
            },
            scores,
        }
    }
}

/// Simple rule-based learned router used in tests and as a fallback example.
pub struct RuleRouter;
impl LearnedRouter for RuleRouter {
    fn score(&self, flow: &FlowProfile) -> [f32; 3] {
        let mut s = [0.0; 3];
        if flow.is_p2p > 0.5 {
            s[1] += 5.0;
        }
        if flow.latency_req_ms < 50.0 {
            s[1] += 3.0;
        }
        if flow.size_bytes > 1_000_000.0 {
            s[0] += 4.0;
        }
        if flow.needs_ordering > 0.5 && flow.size_bytes < 100_000.0 {
            s[2] += 3.0;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bulk_flow_prefers_quic() {
        let r = HybridRouter::new(RouterWeights::default());
        let d = r.route(&FlowProfile {
            size_bytes: 50_000_000.0,
            latency_req_ms: 500.0,
            rtt_ms: 40.0,
            loss_rate: 0.0,
            bw_mbps: 100.0,
            is_p2p: 0.0,
            needs_ordering: 1.0,
        });
        assert_eq!(d.transport, Transport::Quic);
    }

    #[test]
    fn p2p_low_latency_prefers_webrtc() {
        let r = HybridRouter::new(RouterWeights::default());
        r.set_learned(Box::new(RuleRouter));
        let d = r.route(&FlowProfile {
            size_bytes: 1000.0,
            latency_req_ms: 20.0,
            rtt_ms: 30.0,
            loss_rate: 0.01,
            bw_mbps: 5.0,
            is_p2p: 1.0,
            needs_ordering: 0.0,
        });
        assert_eq!(d.transport, Transport::WebRtc);
    }

    #[test]
    fn load_penalty_steers_away_from_saturated() {
        let r = HybridRouter::new(RouterWeights::default());
        r.report_load(Transport::Quic, 1.0);
        let d = r.route(&FlowProfile {
            size_bytes: 50_000_000.0,
            latency_req_ms: 500.0,
            rtt_ms: 40.0,
            loss_rate: 0.0,
            bw_mbps: 100.0,
            is_p2p: 0.0,
            needs_ordering: 1.0,
        });
        // With QUIC saturated, even a bulk flow should not pick QUIC.
        assert_ne!(d.transport, Transport::Quic);
    }
}
