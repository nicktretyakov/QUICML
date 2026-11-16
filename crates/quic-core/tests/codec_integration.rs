//! Integration tests crossing multiple `quic-core` modules.

use quic_core::cid::ConnectionId;
use quic_core::crypto::PacketKey;
use quic_core::frame::{AckEcnCounts, AckFrame, Frame, StreamFrame};
use quic_core::packet::{
    apply_header_protection, encode_header, parse_header, PacketForm, PacketHeader, PnHints,
    VERSION_1,
};
use quic_core::transport_params::TransportParameters;
use quic_core::varint::Varint;
use std::time::Instant;

/// Encode a long Initial header, parse it back, and confirm the DCID/SCID
/// survive the round-trip.
#[test]
fn initial_header_roundtrip() {
    let hdr = PacketHeader {
        packet_number: 7,
        dcid: ConnectionId::from_slice(&[0xab; 8]).unwrap(),
        scid: ConnectionId::from_slice(&[0xcd; 12]).unwrap(),
        form: PacketForm::Initial {
            version: VERSION_1,
            token: bytes::Bytes::new(),
            length: Varint(0),
            pn_len: 4,
        },
    };
    let mut buf = bytes::BytesMut::new();
    let n = encode_header(&hdr, PnHints::default(), 100, &mut buf).unwrap();
    assert!(n > 0);
    let (parsed, _end) = parse_header(&buf).unwrap();
    assert_eq!(parsed.dcid, hdr.dcid);
    assert_eq!(parsed.scid, hdr.scid);
    assert!(matches!(parsed.form, quic_core::packet::RawForm::Initial { .. }));
}

/// Full packet-protection round-trip: derive initial keys, encrypt a frame
/// payload, apply header protection, then reverse the whole pipeline.
#[test]
fn full_packet_protection_roundtrip() {
    let dcid = ConnectionId::from_slice(&[0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08]).unwrap();
    let client = PacketKey::derive_initial(&dcid, false, false).unwrap();

    // Build a STREAM frame as payload.
    let frame = Frame::Stream(StreamFrame {
        stream_id: Varint(0),
        offset: Varint(0),
        fin: true,
        data: bytes::Bytes::from_static(b"integration payload"),
    });
    let mut payload = bytes::BytesMut::new();
    frame.encode(&mut payload).unwrap();

    // Synthetic short header: first byte + 4-byte pn.
    let pn: u64 = 1234;
    let pn_len = 4u8;
    let pn_bytes = quic_core::packet::truncate_pn(pn, pn_len);
    let mut header = Vec::with_capacity(1 + pn_len as usize);
    header.push(0x40 | ((pn_len - 1) & 0x03));
    header.extend_from_slice(&pn_bytes[..pn_len as usize]);

    let mut ct = vec![0u8; payload.len() + 16];
    let n = client.encrypt(&header, &payload, pn, &mut ct).unwrap();
    ct.truncate(n);

    // Apply header protection.
    let mut sample = [0u8; 16];
    sample.copy_from_slice(&ct[..16]);
    let mask = client.header_protection_mask(&sample).unwrap();
    let mut wire = header.clone();
    wire.extend_from_slice(&ct);
    wire[0] ^= mask[0] & 0x0f;
    for i in 0..pn_len as usize {
        wire[1 + i] ^= mask[1 + i];
    }

    // --- Server side: reverse ---
    let server = PacketKey::derive_initial(&dcid, true, false).unwrap();
    // Server uses the *client* key (client->server direction) to remove hp
    // and decrypt, which is the same derive_initial(false) result.
    let ct_off = 1 + pn_len as usize;
    let mut srv_sample = [0u8; 16];
    srv_sample.copy_from_slice(&wire[ct_off..ct_off + 16]);
    let srv_mask = client.header_protection_mask(&srv_sample).unwrap();
    let parsed = quic_core::packet::ParsedHeader {
        dcid,
        scid: ConnectionId::empty(),
        form: quic_core::packet::RawForm::Short {
            pn_len,
            protected_first_byte: wire[0],
        },
        pn_offset: 1,
    };
    let mut buf = wire.clone();
    apply_header_protection(&mut buf, &parsed, &srv_mask, 0).unwrap();
    let header_bytes = buf[..ct_off].to_vec();
    // Client→server Initial traffic is encrypted AND decrypted with the
    // *client* initial key (both sides share it for this direction), so we
    // decrypt with `client`, not the server's own key.
    let pt = client.decrypt(&header_bytes, &mut buf[ct_off..], pn).unwrap();
    let (decoded, _) = Frame::decode(pt).unwrap();
    assert_eq!(decoded, frame);
    // `server` is the key the server would use for the *return* direction.
    let _ = server;
}

/// Transport-parameter negotiation round-trip with non-default values.
#[test]
fn transport_params_negotiation() {
    let mut p = TransportParameters::default();
    p.initial_max_data = 1 << 20;
    p.initial_max_streams_bidi = 128;
    p.initial_max_streams_uni = 64;
    p.max_idle_timeout_ms = 60_000;
    p.ack_delay_exponent = 7;
    p.max_ack_delay_ms = 50;
    p.initial_source_cid = Some(ConnectionId::from_slice(&[0x11, 0x22, 0x33, 0x44]).unwrap());
    let wire = p.encode().unwrap();
    let q = TransportParameters::decode(&wire).unwrap();
    assert_eq!(p, q);
}

/// Loss-detection + ACK drives the congestion controller.
#[test]
fn ack_drives_congestion() {
    use quic_core::congestion::{CongestionController, NewReno, NewRenoConfig};
    use quic_core::loss_detection::{LossDetector, PnSpace, SentPacket};
    let mut det = LossDetector::new(PnSpace::Application);
    let now = Instant::now();
    for pn in 0..10 {
        det.on_sent(SentPacket {
            pn,
            sent_at: now,
            bytes_in_flight: 1200,
            ack_eliciting: true,
            in_flight: true,
        });
    }
    let mut cc = NewReno::new(NewRenoConfig::default());
    let ack = AckFrame {
        largest_acknowledged: Varint(9),
        ack_delay: Varint(0),
        ack_range_count: Varint(0),
        first_ack_range: Varint(9),
        ranges: Vec::new(),
        ecn: Some(AckEcnCounts {
            ect0: 10,
            ect1: 0,
            ce: 0,
        }),
    };
    let out = det.on_ack(&ack, now + std::time::Duration::from_millis(40), &mut cc, std::time::Duration::from_millis(40), std::time::Duration::from_millis(25));
    assert_eq!(out.acked.len(), 10);
    assert!(cc.bytes_in_flight() == 0);
}
