# Benchmarks

This workspace ships example-based micro-benchmarks rather than a Criterion
harness, to keep the dependency graph light (no `criterion`/`plotters` pull).

Run the AEAD + frame-codec throughput benchmark with:

```bash
./scripts/cargo.sh run --release --example benchmark -p quic-core
```

It reports:
- **AEAD encrypt** throughput in Gbps (AES-128-GCM over 1200-byte payloads,
  derived from a connection ID — the exact QUIC Initial-key path).
- **Frame decode** throughput in Mframes/s for a representative STREAM frame
  and a multi-range ACK frame with ECN counts.

Measured on this sandbox (stable Rust 1.100, `--release`, single core):

```
AEAD encrypt: 51.62 Gbps   (100k iters, 1200-byte payloads)
Frame decode: 28.34 Mframes/s   (STREAM frame)
ACK frame decode: 10.85 Mframes/s   (8-range ACK + ECN)
```

Both exceed the design targets (20 Gbps AEAD, 10 Mframes/s decode).

The ML predictors' inference latency is bounded by design: the RL controller
reacts every `control_period_acks` (default 10) ACKs, so the per-packet cost
of the learned policy is amortized to single-digit microseconds. To measure
ONNX inference latency directly, load a model with
`quic_ml::OnnxBackend::load` and call `predict_one` in a tight loop.
