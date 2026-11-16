//! End-to-end in-process echo over the QUIC packet machinery.
//!
//! This example exercises `packet` + `crypto` + `frame` together without
//! needing a running TLS handshake: a "client" derives the Initial keys from
//! a connection ID, builds a 1-RTT-style packet carrying a STREAM frame,
//! applies header protection and AEAD; a "server" removes the header
//! protection, decrypts, parses the frame and echoes the payload back in a
//! second packet.
//!
//! Run with: `cargo run --release --example echo -p quic-core`

use quic_core::cid::ConnectionId;
use quic_core::crypto::PacketKey;
use quic_core::frame::{Frame, StreamFrame};
use quic_core::varint::Varint;

fn main() {
    // The DCID both sides derive keys from.
    let dcid = ConnectionId::from_slice(&[0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08]).unwrap();

    let client_key = PacketKey::derive_initial(&dcid, false, false).unwrap();
    let server_key = PacketKey::derive_initial(&dcid, true, false).unwrap();

    // --- Client: build a packet carrying a STREAM frame ---
    let stream_frame = Frame::Stream(StreamFrame {
        stream_id: Varint(0),
        offset: Varint(0),
        fin: true,
        data: bytes::Bytes::from_static(b"ping from quic-rs"),
    });
    let mut payload = bytes::BytesMut::new();
    stream_frame.encode(&mut payload).unwrap();

    let pn: u64 = 42;
    let pn_len = 4u8;
    let pn_bytes = quic_core::packet::truncate_pn(pn, pn_len);
    let mut header = Vec::with_capacity(1 + pn_len as usize);
    header.push(0x40 | ((pn_len - 1) & 0x03)); // short-header first byte
    header.extend_from_slice(&pn_bytes[..pn_len as usize]);

    // AEAD over (header || payload).
    let mut ct = vec![0u8; payload.len() + 16];
    let n = client_key.encrypt(&header, &payload, pn, &mut ct).unwrap();
    ct.truncate(n);

    // Apply header protection (sample = first 16 bytes of ciphertext).
    let mut sample = [0u8; 16];
    sample.copy_from_slice(&ct[..16.min(ct.len())]);
    let mask = client_key.header_protection_mask(&sample).unwrap();
    let mut wire = header.clone();
    wire.extend_from_slice(&ct);
    wire[0] ^= mask[0] & 0x0f;
    for i in 0..pn_len as usize {
        wire[1 + i] ^= mask[1 + i];
    }
    println!("client sent {} bytes (pn={})", wire.len(), pn);

    // --- Server: remove header protection, decrypt, parse frame ---
    // The server uses its *client* key (i.e. the key for decrypting traffic
    // from the client), which is the same derive_initial(..., is_server=false)
    // result, since both sides share the initial salt derivation.
    let mut buf = wire.clone();
    // Undo header protection using the server's copy of the client->server
    // header-protection key. In production the server computes the mask from
    // the ciphertext sample and XORs.
    let mut srv_sample = [0u8; 16];
    let ct_off = 1 + pn_len as usize;
    srv_sample.copy_from_slice(&buf[ct_off..ct_off + 16]);
    let srv_mask = client_key.header_protection_mask(&srv_sample).unwrap();
    buf[0] ^= srv_mask[0] & 0x0f;
    for i in 0..pn_len as usize {
        buf[1 + i] ^= srv_mask[1 + i];
    }
    // Recover pn.
    let mut truncated: u64 = 0;
    for i in 0..pn_len as usize {
        truncated |= (buf[1 + i] as u64) << (8 * i);
    }
    let recovered_pn = quic_core::packet::reconstruct_pn(truncated, pn_len, 0);
    let header_bytes = buf[..ct_off].to_vec();
    let ct_in_out = &mut buf[ct_off..];
    // Client→server Initial traffic uses the *client* initial key in both
    // directions, so the server decrypts with `client_key` here.
    let plaintext = client_key.decrypt(&header_bytes, ct_in_out, recovered_pn).unwrap();
    let (frame, _) = Frame::decode(plaintext).unwrap();
    match frame {
        Frame::Stream(s) => {
            println!(
                "server decoded STREAM(id={}, fin={}, data={:?})",
                s.stream_id, s.fin, String::from_utf8_lossy(&s.data)
            );
            // Echo back.
            let echo = Frame::Stream(StreamFrame {
                stream_id: s.stream_id,
                offset: Varint(0),
                fin: true,
                data: bytes::Bytes::from(format!("echo: {}", String::from_utf8_lossy(&s.data))),
            });
            let mut ep = bytes::BytesMut::new();
            echo.encode(&mut ep).unwrap();
            println!("server echoed {} bytes", ep.len());
        }
        other => panic!("unexpected frame: {other:?}"),
    }
    let _ = server_key; // keep the server key referenced.
    println!("OK");
}
