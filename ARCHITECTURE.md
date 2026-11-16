# Architecture

This document describes the internal structure of the `quic-rs` workspace,
the data path through the transport, and how the ML components plug in
without coupling the core to any specific model format.

## Crate graph

```
            quic-hybrid ─┐
                         ├──► quic-ml ──► quic-core ──► tokio / rustls / ring / aes
            quic-p2p-lb ─┘                                  │
                                                         std::net::UdpSocket
```

`quic-core` is the foundation: wire codecs, crypto, TLS driver, streams,
loss detection, congestion, the connection state machine and the UDP
endpoint. `quic-ml` depends on `quic-core` (it implements
`quic_core::CongestionController` and consumes `CongestionStats`). The two
edge crates (`quic-hybrid`, `quic-p2p-lb`) depend on both and are
control-plane only.

## Data path (inbound)

```
UdpSocket::recv_from
   │
   ▼
Endpoint::route_datagram            ← parse_header / parse_short_header
   │  (route by DCID via by_dcid map; create conn on server Initial)
   ▼
Connection::handle_datagram         ← split coalesced packets
   │
   ▼
Connection::handle_one_packet
   │  1. parse (still header-protected) header
   │  2. pick the packet-number space (Initial/Handshake/1-RTT)
   │  3. PacketKey::header_protection_mask(sample) → XOR first byte + pn
   │  4. reconstruct_pn(truncated, pn_len, largest_acked)
   │  5. PacketKey::decrypt(header, ciphertext, pn)
   │  6. Frame::decode loop over the plaintext payload
   ▼
dispatch frames:
   CRYPTO        → TlsSession::read_crypto        → may emit KeyChange
   ACK           → LossDetector::on_ack           → CongestionController::on_packets_acked
   STREAM        → RecvHalf::receive              → flow-control check
   MAX_DATA ...  → StreamMap::on_max_data
   CONNECTION_CLOSE → Connection::close
```

## Data path (outbound)

```
application / TLS engine produces bytes to send
   │
   ▼
MlScheduler::schedule(pending, cwnd_budget)   ← quic-ml (optional)
   │
   ▼
Frame::encode (STREAM / CRYPTO / ACK / PING)
   │
   ▼
Connection::build_packet(space, payload)
   │  1. encode_header (long/short, pn truncated)
   │  2. PacketKey::encrypt(header, payload, pn)
   │  3. PacketKey::header_protection_mask(ct[..16]) → XOR
   │  4. LossDetector::on_sent (record SentPacket)
   ▼
Endpoint socket.send_to(peer)
```

## Loss detection & congestion

Each packet-number space owns a `LossDetector` (RFC 9002 §6). On every ACK:

1. The acked packet numbers are reconstructed from the ACK frame's ranges.
2. RTT is sampled from the largest acked packet and fed to the
   `CongestionController`.
3. Loss is declared by packet-threshold (gap > 3) and time-threshold
   (`> 9/8 · smoothed_rtt`); persistent congestion collapses the window.
4. The controller (default `NewReno`, swappable for `quic_ml::RlCongestion`)
   updates `cwnd` and `bytes_in_flight`.

The PTO timer is `smoothed_rtt + 4·rttvar + max_ack_delay`; when it fires,
the connection emits a probe. An LSTM-based `RttPredictor`
(`quic_ml::rtt_predictor`) can shorten this to `predicted_rtt + band`,
which is the "dynamic PTO update" feature.

## Crypto layers

| Space | Key source | Implementation |
| --- | --- | --- |
| Initial | HKDF over the DCID + QUIC initial salt (RFC 9001 §5.2) | `crypto::PacketKey::derive_initial` (ring HKDF + AES-ECB hp via `aes`) |
| Handshake / 1-RTT | TLS handshake (rustls `quic::KeyChange`) | rustls `quic::PacketKey` trait objects (wiring hook in `Connection::install_*`) |
| Header protection | AES-ECB of a 16-byte ciphertext sample | `crypto::PacketKey::header_protection_mask` |

The Initial-key derivation is verified against the RFC 9001 Appendix A test
vector (`crypto::tests::rfc9001_initial_keys_client`).

## ML plug-in points

Every ML component talks to the core through a trait, so the core never
imports `ort`:

| Trait | Implementor | Used by |
| --- | --- | --- |
| `quic_core::CongestionController` | `NewReno`, `quic_ml::RlCongestion` | `Connection` |
| `quic_ml::InferenceBackend` | `OnnxBackend`, `NoBackend` | every predictor |
| `quic_ml::GpuBackend` | `CpuBackend` (+ CUDA via `cuda` feature) | feature normalization |
| `quic_hybrid::LearnedRouter` | `RuleRouter` (+ ONNX impl) | `HybridRouter` |
| `quic_p2p_lb::LoadPredictor` | EWMA fallback (+ ONNX impl) | `P2pLoadBalancer` |

Each predictor degrades gracefully: when `InferenceBackend::predict_*`
returns `Unavailable`, it uses a deterministic heuristic (Holt's method,
rule-based Reno/BBR, EWMA distance, 70-byte MTU steps). This makes the crate
deployable with zero model files, and lets operators hot-swap trained models
without recompiling.

## Roadmap to a wire-complete handshake

The pieces below are implemented and unit-tested; the remaining work to
carry a real handshake end-to-end is wiring, not algorithm:

1. **Key rotation.** In `Connection::handle_one_packet`, when a CRYPTO frame
   is delivered to `TlsSession::read_crypto`, capture the
   `rustls::quic::KeyChange` returned by `write_crypto` and call
   `install_handshake_keys` / `install_1rtt_keys`. The hook signatures
   already exist; they need to accept `rustls::quic::Keys` (or a small
   adapter that runs the rustls `PacketKey` trait for AEAD+hp instead of
   `crypto::PacketKey`).
2. **Full decrypt path.** `Connection::handle_one_packet` currently parses
   but does not yet AEAD-open; the `echo` example demonstrates the exact
   sequence (`apply_header_protection` → `reconstruct_pn` → `decrypt`) that
   needs to be called per space.
3. **ACK emission.** `LossDetector::build_ack_frame` is implemented; wire it
   into the outbound path so ack-eliciting packets are answered.
4. **Stream retransmission.** Lost STREAM frames must be re-queued from the
   `SendHalf::outbox` (the loss detector already reports lost pns; map them
   back to the frames they carried).

## Performance posture

- `cargo build --release` uses `lto = "thin"`, `codegen-units = 1`,
  `panic = "abort"`, `strip = true`.
- The frame codec and AEAD are benchmarked (`examples/benchmark`): target is
  >20 Gbps AEAD and >10 Mframes/s decode on a modern core.
- Per-connection state is behind `parking_lot::Mutex`; the endpoint's recv
  loop is a single `tokio::select!` over `recv_from` and a shutdown notify,
  so there is one hot task per socket.
- ML inference runs out of the data path: the controller reacts every
  `control_period_acks` (default 10) ACKs, not per packet, keeping the
  learned policy's cost amortized to single-digit microseconds per packet.
