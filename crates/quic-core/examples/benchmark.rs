//! Throughput benchmark for the QUIC packet-protection and frame codec.
//!
//! Measures:
//!  - `PacketKey::encrypt` + `decrypt` round-trip throughput (AES-128-GCM
//!    over 1200-byte payloads), in Gbps.
//!  - `Frame::encode` + `decode` round-trip throughput for a representative
//!    mix of STREAM, ACK and CRYPTO frames, in Mframes/s.
//!
//! Run with: `cargo run --release --example benchmark -p quic-core`

use quic_core::cid::ConnectionId;
use quic_core::crypto::PacketKey;
use quic_core::frame::{AckEcnCounts, AckFrame, Frame, StreamFrame};
use quic_core::varint::Varint;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let iters = 100_000;

    // ---- AEAD benchmark ----
    let dcid = ConnectionId::from_slice(&[0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08]).unwrap();
    let key = PacketKey::derive_initial(&dcid, false, false).unwrap();
    let header = b"short-header-aad";
    let payload = vec![0xa5u8; 1200];
    let mut ct = vec![0u8; payload.len() + 16];
    // warmup
    for _ in 0..1000 {
        key.encrypt(header, &payload, 1, &mut ct).unwrap();
    }
    let start = Instant::now();
    for i in 0..iters {
        let pn = (i & 0xffff) as u64;
        key.encrypt(header, &payload, pn, &mut ct).unwrap();
        black_box(&ct);
    }
    let enc_elapsed = start.elapsed();
    let bytes = (payload.len() as u64) * (iters as u64);
    let gbps_enc = (bytes as f64 * 8.0) / enc_elapsed.as_secs_f64() / 1e9;
    println!(
        "AEAD encrypt: {:.2} Gbps  ({} iters in {:?})",
        gbps_enc, iters, enc_elapsed
    );

    // ---- Decode benchmark ----
    let mut buf = bytes::BytesMut::with_capacity(256);
    let frame = Frame::Stream(StreamFrame {
        stream_id: Varint(4),
        offset: Varint(0),
        fin: true,
        data: bytes::Bytes::from_static(b"hello quic"),
    });
    frame.encode(&mut buf).unwrap();
    let wire = buf.freeze();
    let start = Instant::now();
    for _ in 0..iters {
        let (f, _) = Frame::decode(&wire).unwrap();
        black_box(f);
    }
    let elapsed = start.elapsed();
    let mfps = (iters as f64) / elapsed.as_secs_f64() / 1e6;
    println!("Frame decode: {:.2} Mframes/s", mfps);

    // ---- ACK frame (larger) ----
    let ack = Frame::Ack(AckFrame {
        largest_acknowledged: Varint(1_000_000),
        ack_delay: Varint(5000),
        ack_range_count: Varint(8),
        first_ack_range: Varint(5),
        ranges: (0..8).map(|i| (Varint(i), Varint(i + 1))).collect(),
        ecn: Some(AckEcnCounts {
            ect0: 100,
            ect1: 2,
            ce: 0,
        }),
    });
    let mut ab = bytes::BytesMut::with_capacity(256);
    ack.encode(&mut ab).unwrap();
    let awire = ab.freeze();
    let start = Instant::now();
    for _ in 0..iters {
        let (f, _) = Frame::decode(&awire).unwrap();
        black_box(f);
    }
    let elapsed = start.elapsed();
    let mfps = (iters as f64) / elapsed.as_secs_f64() / 1e6;
    println!("ACK frame decode: {:.2} Mframes/s", mfps);
}
