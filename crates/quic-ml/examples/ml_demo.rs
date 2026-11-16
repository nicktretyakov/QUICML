//! Demonstration of every ML predictor in `quic-ml` running with the
//! deterministic fallback (no ONNX model files required).
//!
//! Run with: `cargo run --release --example ml_demo -p quic-ml`

use quic_ml::anomaly::{AnomalyConfig, AnomalyDetector, TrafficFeatures};
use quic_ml::mtu_predictor::{MtuFeatures, MtuPredictor};
use quic_ml::rl_congestion::{RlConfig, RlCongestion};
use quic_ml::rtt_predictor::RttPredictor;
use quic_ml::scheduler::{MlScheduler, SchedulerWeights};
use quic_core::CongestionController;
use std::time::{Duration, Instant};

fn main() {
    println!("=== quic-ml demo (fallback heuristics, no ONNX models) ===\n");

    // ---- LSTM RTT predictor (double-exponential smoothing fallback) ----
    let mut rtt = RttPredictor::new(None);
    for i in 0..30 {
        rtt.observe(Duration::from_millis(20 + i / 2));
    }
    let p = rtt.predict().unwrap();
    println!(
        "RTT predictor: forecast mean = {:?}, std = {:?}",
        p.mean, p.std
    );

    // ---- RL congestion control (rule-based fallback) ----
    let mut cc = RlCongestion::new(RlConfig::default(), None);
    let now = Instant::now();
    let initial = cc.window();
    for _ in 0..500 {
        cc.on_packet_sent(1200, now);
        cc.on_packets_acked(1200, Duration::from_millis(8), Duration::ZERO, now);
    }
    println!(
        "RL congestion: cwnd grew {} -> {} ({}x)",
        initial,
        cc.window(),
        cc.window() as f64 / initial as f64
    );
    // Inject a loss burst.
    let before_loss = cc.window();
    for _ in 0..20 {
        cc.on_packets_lost(1200, false, now);
    }
    println!(
        "RL congestion: after loss burst cwnd {} -> {} (reduced)",
        before_loss,
        cc.window()
    );

    // ---- Anomaly / attack detection ----
    let mut det = AnomalyDetector::new(AnomalyConfig::default(), None);
    println!("\nAnomaly detector over a flood:");
    for t in 0..15 {
        let f = TrafficFeatures {
            pps: 50_000.0 + t as f32 * 1000.0,
            bps: 6_000_000.0,
            init_frac: 1.0,
            inter_arrival_us: 10.0,
            inter_arrival_std_us: 1.0,
            unique_dcids: 5000.0,
            ack_eliciting_ratio: 0.0,
            handshake_completion: 0.0,
        };
        let (v, s) = det.observe(f);
        println!("  t={t:2} score={s:.3} verdict={v:?}");
    }

    // ---- MTU predictor ----
    let mut mtu = MtuPredictor::new(None);
    let f = MtuFeatures {
        loss_rate_1200: 0.0,
        loss_rate_1400: 0.0,
        srtt_ms: 20.0,
        ttl: 64.0,
        ipv4: 1.0,
        last_mtu: 1200.0,
    };
    println!("\nMTU predictor probing sequence:");
    loop {
        match mtu.predict_probe(f) {
            Some(target) => {
                println!("  probe {} -> success", target);
                mtu.on_probe_success(target);
                if target >= 1454 {
                    break;
                }
            }
            None => break,
        }
    }
    println!("  final confirmed MTU = {}", mtu.confirmed_mtu());

    // ---- Scheduler ----
    let mut sched = MlScheduler::new(SchedulerWeights::default(), false);
    sched.observe(0, |s| s.priority = 50);
    sched.observe(4, |s| s.priority = 200);
    sched.observe(8, |s| s.priority = 10);
    let pending = vec![(0, 5000), (4, 5000), (8, 5000)];
    let plan = sched.schedule(&pending, 6000);
    println!("\nScheduler plan (budget=6000 bytes):");
    for s in &plan {
        println!("  stream {} <- {} bytes", s.stream_id, s.bytes);
    }
}
