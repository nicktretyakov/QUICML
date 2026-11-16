# quic-rs — QUIC (RFC 9000) on tokio + `std::net` with ML-driven control

A from-scratch Rust workspace implementing the QUIC transport protocol
(RFC 9000, RFC 9001, RFC 9002, RFC 9369) on top of `tokio` and
`std::net::UdpSocket`, with `rustls` driving the TLS 1.3 handshake and a
suite of ML-driven enhancements (LSTM RTT prediction, RL congestion control,
anomaly/attack detection, ML-guided MTU, hybrid QUIC+WebRTC+gRPC routing,
P2P load balancing) backed by ONNX Runtime.

> **Status.** This is a production-oriented skeleton: the wire primitives
> (varint, packet headers, frames, transport parameters, initial-key
> derivation + AEAD + header protection, stream state machine, flow control,
> loss detection, NewReno congestion) are implemented and unit-tested against
> RFC test vectors. The endpoint/connection handshake *driver* (wiring
> rustls's QUIC key extraction into the 1-RTT space and completing a real
> end-to-end handshake) is the documented next integration step — see
> [ARCHITECTURE.md](ARCHITECTURE.md) §"Roadmap to a wire-complete handshake".
> The ML predictors compile, are unit-tested with their deterministic
> fallbacks, and accept trained ONNX models via `quic-ml::OnnxBackend`.

## Workspace layout

```
quic-rs/
├── Cargo.toml                 # virtual workspace + shared [workspace.dependencies]
├── crates/
│   ├── quic-core/             # transport: varint, packet, frame, crypto, tls,
│   │                          #          stream, loss_detection, congestion,
│   │                          #          connection, endpoint
│   ├── quic-ml/               # ML: ort_backend, cuda, rtt_predictor (LSTM),
│   │                          #     rl_congestion, anomaly, mtu_predictor, scheduler
│   ├── quic-hybrid/           # QUIC+WebRTC+gRPC dynamic router (ML-scored)
│   └── quic-p2p-lb/           # P2P load balancer with ML load prediction
├── examples/                  # (per-crate examples/ — see below)
├── models/generate_models.py  # train + export ONNX models
└── benches/                   # Criterion benchmarks (planned)
```

Examples live under each crate's `examples/` directory:

| Example | Crate | What it shows |
| --- | --- | --- |
| `echo` | quic-core | End-to-end in-process crypto round-trip: client encrypts a STREAM frame with derived Initial keys, server removes header protection + decrypts + echoes. |
| `benchmark` | quic-core | AEAD encrypt/decrypt throughput (Gbps) and frame codec throughput (Mframes/s). |
| `ml_demo` | quic-ml | Every ML predictor running on its fallback heuristic. |

Run them with, e.g.:

```bash
./scripts/cargo.sh run --release --example echo       -p quic-core
./scripts/cargo.sh run --release --example benchmark  -p quic-core
./scripts/cargo.sh run --release --example ml_demo    -p quic-ml
```

## Build

The toolchain is installed **inside the project** (`./rustup`, `./cargo`) so
no system-wide Rust is required. Use the wrapper:

```bash
./scripts/cargo.sh build --workspace --release
./scripts/cargo.sh test  --workspace
```

`quic-ml` downloads a prebuilt ONNX Runtime shared library at build time
(`ort` feature `download-binaries` + `tls-rustls`); the `models/out/*.onnx`
files are produced by `models/generate_models.py` and loaded at runtime.

### AWS / FIPS note

To use `aws-lc-rs` (the AWS-validated, FIPS-capable crypto provider) instead
of `ring`, disable the default `rustls-ring` feature and enable
`rustls-aws-lc-rs`:

```bash
./scripts/cargo.sh build -p quic-core --no-default-features --features rustls-aws-lc-rs,std
```

## Feature flags

| Crate | Feature | Effect |
| --- | --- | --- |
| quic-core | `rustls-ring` (default) | TLS provider = ring. |
| quic-core | `rustls-aws-lc-rs` | TLS provider = aws-lc-rs (FIPS). |
| quic-core | `trace-packets` | Trace every inbound/outbound packet. |
| quic-ml | `onnx` (default) | Use `ort` (ONNX Runtime) for inference. |
| quic-ml | `cuda` | Reserve the CUDA kernel path for batched preprocessing. |

## ML feature matrix

| Function | Module | Model contract | Fallback |
| --- | --- | --- | --- |
| RTT prediction | `rtt_predictor` | LSTM `[1,16,1]→[1,1]` (ms) | Double-exponential smoothing (Holt). |
| Congestion control | `rl_congestion` | Policy MLP `[1,8]→[1,5]` logits | Rule-based Reno/BBR hybrid. |
| Anomaly / attack | `anomaly` | Scorer `[1,8]→[1,1]` in `[0,1]` | EWMA distance from baseline. |
| MTU prediction | `mtu_predictor` | Regressor `[1,6]→[1,1]` (bytes) | 70-byte step probe (RFC 8899). |
| Packet scheduling | `scheduler` | (linear scorer, hot-swappable weights) | Priority/age/loss linear score. |
| Hybrid routing | `quic-hybrid` | `LearnedRouter` trait | Hand-tuned linear weights. |
| P2P load balancing | `quic-p2p-lb` | `LoadPredictor` trait | EWMA trend extrapolation. |

## Running the model trainer

```bash
pip install numpy scikit-learn skl2onnx onnxruntime   # + torch for the real LSTM
python models/generate_models.py
```

This writes `models/out/{rtt_lstm,rl_policy,anomaly,mtu_regressor}.onnx`.
Load them in Rust:

```rust
use quic_ml::{OnnxBackend, InferenceBackend, RttPredictor};
let backend = OnnxBackend::load("models/out/rtt_lstm.onnx", None)?;
let mut rtt = RttPredictor::new(Some(Box::new(backend)));
rtt.observe(std::time::Duration::from_millis(42));
println!("{:?}", rtt.predict());
```

## Roadmap to a wire-complete handshake

The remaining work to carry a real client↔server handshake to completion is
narrow and localized:

1. Wire rustls's `quic_read_keys()` into
   `Connection::install_handshake_keys` / `install_1rtt_keys` (the hooks
   already exist).
2. Drive `TlsSession::read_crypto`/`write_crypto` from CRYPTO frames in
   `Connection::handle_one_packet`, splitting TLS records by epoch.
3. Implement full header-protection removal + AEAD open in
   `Connection::handle_one_packet` (the helpers in `packet` +
   `crypto::PacketKey::decrypt` are already used by the `echo` example).
4. Emit ACK frames from `LossDetector::build_ack_frame` on ack-eliciting
   packets.

Everything else (loss detection, congestion, flow control, stream state,
transport-parameter negotiation) is already exercised by unit tests.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
