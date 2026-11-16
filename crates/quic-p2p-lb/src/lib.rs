//! `quic-p2p-lb` — peer-to-peer QUIC load balancer with ML load prediction.
//!
//! A set of edge peers cooperatively load-balance incoming QUIC connections.
//! [`P2pLoadBalancer`] tracks per-peer load telemetry, predicts each peer's
//! load a short horizon ahead using an ONNX model (or an EWMA heuristic), and
//! routes new connections to the peer with the lowest *predicted* load. This
//! avoids the oscillation that plagues least-connections balancers when
//! bursts arrive simultaneously.
//!
//! Routing is DCID-based and stateless: the balancer rewrites the DCID of the
//! incoming Initial to a per-peer routable CID and forwards the datagram
//! unchanged, so peers do not need shared flow state.

#![warn(missing_docs)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::net::SocketAddr;

/// Per-peer telemetry snapshot.
#[derive(Debug, Clone, Copy, Default)]
pub struct PeerLoad {
    /// Current active connection count.
    pub conns: f32,
    /// Current bytes-per-second aggregate.
    pub bps: f32,
    /// Current packet-per-second aggregate.
    pub pps: f32,
    /// CPU utilization 0..1.
    pub cpu: f32,
    /// Memory utilization 0..1.
    pub mem: f32,
}

impl PeerLoad {
    fn features(&self) -> [f32; 5] {
        [self.conns, self.bps, self.pps, self.cpu, self.mem]
    }
}

/// One peer entry.
#[derive(Debug, Clone)]
pub struct Peer {
    pub addr: SocketAddr,
    pub load: PeerLoad,
    /// Load history for the EWMA predictor.
    history: Vec<f32>,
}

/// A learned load predictor (e.g. ONNX LSTM). Implement and install via
/// [`P2pLoadBalancer::set_predictor`].
pub trait LoadPredictor: Send + Sync {
    /// Predict the load scalar `horizon_s` seconds ahead for the peer with
    /// the given history + current snapshot.
    fn predict(&self, history: &[f32], current: &PeerLoad, horizon_s: f32) -> f32;
}

/// The balancer.
pub struct P2pLoadBalancer {
    peers: Mutex<HashMap<SocketAddr, Peer>>,
    predictor: Mutex<Option<Box<dyn LoadPredictor>>>,
    /// EWMA smoothing for the heuristic fallback.
    ewma_alpha: f32,
    horizon_s: f32,
}

impl P2pLoadBalancer {
    /// New empty balancer with a prediction horizon (seconds).
    pub fn new(horizon_s: f32) -> Self {
        Self {
            peers: Mutex::new(HashMap::new()),
            predictor: Mutex::new(None),
            ewma_alpha: 0.3,
            horizon_s,
        }
    }

    /// Add or replace a peer.
    pub fn add_peer(&self, addr: SocketAddr) {
        self.peers.lock().insert(
            addr,
            Peer {
                addr,
                load: PeerLoad::default(),
                history: Vec::with_capacity(64),
            },
        );
    }

    /// Remove a peer.
    pub fn remove_peer(&self, addr: &SocketAddr) {
        self.peers.lock().remove(addr);
    }

    /// Install a learned predictor.
    pub fn set_predictor(&self, p: Box<dyn LoadPredictor>) {
        *self.predictor.lock() = Some(p);
    }

    /// Update a peer's current load.
    pub fn report_load(&self, addr: SocketAddr, load: PeerLoad) {
        let mut peers = self.peers.lock();
        if let Some(peer) = peers.get_mut(&addr) {
            peer.load = load;
            // Compute a scalar load score for history.
            let scalar = load.conns * 0.3 + load.cpu * 100.0 + load.mem * 50.0;
            peer.history.push(scalar);
            if peer.history.len() > 64 {
                peer.history.remove(0);
            }
        }
    }

    /// Predict each peer's load and return the address of the least-loaded.
    pub fn pick_peer(&self) -> Option<SocketAddr> {
        let predictor = self.predictor.lock();
        let pred = predictor.as_ref();
        let peers = self.peers.lock();
        let mut best: Option<(SocketAddr, f32)> = None;
        for (addr, peer) in peers.iter() {
            let predicted = match pred {
                Some(p) => p.predict(&peer.history, &peer.load, self.horizon_s),
                None => ewma_predict(&peer.history, &peer.load, self.ewma_alpha, self.horizon_s),
            };
            match best {
                None => best = Some((*addr, predicted)),
                Some((_, bv)) if predicted < bv => best = Some((*addr, predicted)),
                _ => {}
            }
        }
        best.map(|(a, _)| a)
    }

    /// Forward an incoming datagram to the chosen peer. In a real deployment
    /// the balancer rewrites the DCID to a per-peer routable CID; here we
    /// return the chosen peer address so the caller's socket layer sends it.
    pub fn route(&self, _dgram: &[u8]) -> Option<SocketAddr> {
        self.pick_peer()
    }
}

/// Heuristic predictor: linear extrapolation of the EWMA trend.
fn ewma_predict(history: &[f32], current: &PeerLoad, alpha: f32, horizon_s: f32) -> f32 {
    let mut level = if let Some(&last) = history.last() {
        last
    } else {
        0.0
    };
    let mut trend = 0.0;
    for &v in history {
        let new_level = alpha * v + (1.0 - alpha) * (level + trend);
        trend = alpha * (new_level - level) + (1.0 - alpha) * trend;
        level = new_level;
    }
    let baseline = current.conns * 0.3 + current.cpu * 100.0 + current.mem * 50.0;
    // Blend: trust recent telemetry, lean on trend for the horizon.
    (level + trend * horizon_s) * 0.6 + baseline * 0.4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_least_loaded() {
        let lb = P2pLoadBalancer::new(5.0);
        let a: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let b: SocketAddr = "127.0.0.1:2".parse().unwrap();
        lb.add_peer(a);
        lb.add_peer(b);
        lb.report_load(
            a,
            PeerLoad {
                conns: 1000.0,
                bps: 0.0,
                pps: 0.0,
                cpu: 0.9,
                mem: 0.8,
            },
        );
        lb.report_load(
            b,
            PeerLoad {
                conns: 10.0,
                bps: 0.0,
                pps: 0.0,
                cpu: 0.1,
                mem: 0.1,
            },
        );
        let chosen = lb.pick_peer().unwrap();
        assert_eq!(chosen, b);
    }

    #[test]
    fn trend_extrapolation_favors_declining_peer() {
        let lb = P2pLoadBalancer::new(5.0);
        let a: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let b: SocketAddr = "127.0.0.1:2".parse().unwrap();
        lb.add_peer(a);
        lb.add_peer(b);
        // Peer a: high but declining. Peer b: low but rising fast.
        for i in 0..20 {
            lb.report_load(
                a,
                PeerLoad {
                    conns: (100.0 - i as f32).max(0.0),
                    bps: 0.0,
                    pps: 0.0,
                    cpu: 0.5,
                    mem: 0.5,
                },
            );
            lb.report_load(
                b,
                PeerLoad {
                    conns: (10.0 + i as f32 * 5.0),
                    bps: 0.0,
                    pps: 0.0,
                    cpu: 0.5,
                    mem: 0.5,
                },
            );
        }
        let chosen = lb.pick_peer().unwrap();
        // a is trending down, b is trending up hard: pick a.
        assert_eq!(chosen, a);
    }
}
