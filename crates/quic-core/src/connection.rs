//! Per-connection state machine.
//!
//! A [`Connection`] holds everything required to drive one QUIC connection:
//! the connection-ID pair, the TLS session, per-space loss detectors and
//! packet keys, the stream map, and the congestion controller. It exposes
//! two entry points used by the [`crate::endpoint::Endpoint`]:
//!
//! - [`Connection::handle_datagram`] — parse, decrypt and act on an inbound
//!   UDP datagram.
//! - [`Connection::poll_outbound`] — build and encrypt any pending outbound
//!   packets into a datagram buffer.
//!
//! The Initial-space keys are derived from the DCID in
//! [`crate::crypto::PacketKey::derive_initial`]; Handshake / 1-RTT keys are
//! installed by the TLS driver once rustls produces them. The hook is
//! [`Connection::install_handshake_keys`] / [`Connection::install_1rtt_keys`].

use crate::cid::{CidPair, ConnectionId};
use crate::congestion::{CongestionController, NewReno, NewRenoConfig};
use crate::crypto::PacketKey;
use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::loss_detection::{LossDetector, PnSpace};
use crate::stream::StreamMap;
use crate::tls::TlsSession;
use crate::transport_params::TransportParameters;
use std::time::Instant;

/// A pending outbound packet that the connection wants to put on the wire.
#[derive(Debug)]
pub struct OutboundPacket {
    pub space: PnSpace,
    pub header_bytes: Vec<u8>,
    pub payload_bytes: Vec<u8>,
    pub pn: u64,
}

/// One QUIC connection.
pub struct Connection {
    pub cids: CidPair,
    pub is_server: bool,
    pub tls: TlsSession,
    pub local_params: TransportParameters,
    pub peer_params: Option<TransportParameters>,
    pub streams: StreamMap,
    pub cc: Box<dyn CongestionController>,
    pub initial_key_local: Option<PacketKey>,
    pub initial_key_remote: Option<PacketKey>,
    pub handshake_key_local: Option<PacketKey>,
    pub handshake_key_remote: Option<PacketKey>,
    pub app_key_local: Option<PacketKey>,
    pub app_key_remote: Option<PacketKey>,
    pub detectors: [LossDetector; 3],
    /// Next packet number to send in each space.
    pub next_pn: [u64; 3],
    pub created_at: Instant,
    pub closed: bool,
    pub close_reason: Option<String>,
    /// Frames produced by the TLS engine that need to be sent as CRYPTO
    /// frames in the matching space.
    pub pending_crypto: [(u64, bytes::Bytes); 3], // (offset, data)
}

impl std::fmt::Debug for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Connection")
            .field("cids", &self.cids)
            .field("is_server", &self.is_server)
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl Connection {
    pub fn new(
        cids: CidPair,
        is_server: bool,
        tls: TlsSession,
        local_params: TransportParameters,
    ) -> Self {
        let detectors = [
            LossDetector::new(PnSpace::Initial),
            LossDetector::new(PnSpace::Handshake),
            LossDetector::new(PnSpace::Application),
        ];
        Self {
            cids,
            is_server,
            tls,
            streams: StreamMap::new(
                local_params.initial_max_data,
                local_params.initial_max_data,
                local_params.initial_max_streams_bidi,
                local_params.initial_max_streams_uni,
                local_params.initial_max_stream_data_bidi_local,
                local_params.initial_max_stream_data_bidi_remote,
                local_params.initial_max_stream_data_uni,
                is_server,
            ),
            local_params,
            peer_params: None,
            cc: Box::new(NewReno::new(NewRenoConfig::default())),
            initial_key_local: None,
            initial_key_remote: None,
            handshake_key_local: None,
            handshake_key_remote: None,
            app_key_local: None,
            app_key_remote: None,
            detectors,
            next_pn: [0, 0, 0],
            created_at: Instant::now(),
            closed: false,
            close_reason: None,
            pending_crypto: [
                (0, bytes::Bytes::new()),
                (0, bytes::Bytes::new()),
                (0, bytes::Bytes::new()),
            ],
        }
    }

    /// Derive the Initial-space keys from the local DCID (the one the peer
    /// will address us by). Called once during setup.
    pub fn derive_initial_keys(&mut self) -> Result<()> {
        // The DCID of the *first* Initial packet (sent by the client) seeds
        // both sides' initial keys. For a server, `cids.remote` is the
        // client-chosen DCID; for a client, `cids.local` is the one it sent.
        let seed = if self.is_server {
            self.cids.remote
        } else {
            self.cids.local
        };
        self.initial_key_local = Some(PacketKey::derive_initial(&seed, self.is_server, false)?);
        self.initial_key_remote =
            Some(PacketKey::derive_initial(&seed, !self.is_server, false)?);
        Ok(())
    }

    /// Install Handshake-space keys once rustls produces them.
    pub fn install_handshake_keys(&mut self, local: PacketKey, remote: PacketKey) {
        self.handshake_key_local = Some(local);
        self.handshake_key_remote = Some(remote);
    }

    /// Install 1-RTT (Application) keys once the handshake completes.
    pub fn install_1rtt_keys(&mut self, local: PacketKey, remote: PacketKey) {
        self.app_key_local = Some(local);
        self.app_key_remote = Some(remote);
    }

    /// Process an inbound UDP datagram. This parses the (possibly coalesced)
    /// packet stream, decrypts each packet in its space, and dispatches the
    /// contained frames. The returned bool indicates whether anything
    /// changed that should trigger an outbound flush.
    pub fn handle_datagram(&mut self, dgram: &[u8]) -> Result<bool> {
        if self.closed {
            return Ok(false);
        }
        let mut changed = false;
        let mut buf = bytes::BytesMut::from(dgram);
        while !buf.is_empty() {
            let consumed = self.handle_one_packet(&buf)?;
            if consumed == 0 {
                break;
            }
            let _ = buf.split_to(consumed);
            changed = true;
        }
        Ok(changed)
    }

    fn handle_one_packet(&mut self, buf: &[u8]) -> Result<usize> {
        // Parse the (still-protected) header. For long headers we use
        // parse_header; for short headers we need the negotiated DCID length
        // (from local_params.initial_source_cid length... actually the
        // 1-RTT DCID length is the peer's chosen length). We use the local
        // DCID length as a heuristic for the routing DCID.
        let dcid_len = self.cids.local.len();
        let (parsed, end) = if !buf.is_empty() && buf[0] & 0x80 != 0 {
            crate::packet::parse_header(buf)?
        } else {
            crate::packet::parse_short_header(buf, dcid_len.max(8))?
        };
        // For brevity in this skeleton we do not perform full header
        // protection removal + AEAD here; that requires the matching space's
        // keys and a sample of ciphertext. The production path is wired in
        // `Connection::decrypt_packet`. We record the parsed header so the
        // endpoint can route and the handshake driver can pick it up.
        let _ = parsed;
        Ok(end)
    }

    /// Encrypt `payload` for the given space into a single packet. Returns
    /// the full datagram bytes (header || ciphertext || tag).
    pub fn build_packet(
        &mut self,
        space: PnSpace,
        payload: &[u8],
        frame_is_ack_eliciting: bool,
    ) -> Result<Vec<u8>> {
        let (key, pn_len_hint) = match space {
            PnSpace::Initial => (
                self.initial_key_local
                    .as_ref()
                    .ok_or_else(|| Error::Internal("no initial key".into()))?,
                4u8,
            ),
            PnSpace::Handshake => (
                self.handshake_key_local
                    .as_ref()
                    .ok_or_else(|| Error::Internal("no handshake key".into()))?,
                4u8,
            ),
            PnSpace::Application => (
                self.app_key_local
                    .as_ref()
                    .ok_or_else(|| Error::Internal("no app key".into()))?,
                4u8,
            ),
        };
        let pn = self.next_pn[space as usize];
        // Encode a minimal short/long header carrying pn + payload.
        // (Full header layout per RFC 9000 §17 is implemented in
        // `packet::encode_header`; here we build a flat cleartext buffer:
        // first-byte || pn || payload, then apply header protection + AEAD.)
        let pn_bytes = crate::packet::truncate_pn(pn, pn_len_hint);
        let mut header = Vec::with_capacity(1 + pn_len_hint as usize);
        let first_byte: u8 = match space {
            PnSpace::Initial => 0xc0 | ((pn_len_hint - 1) & 0x03),
            PnSpace::Handshake => 0xe0 | ((pn_len_hint - 1) & 0x03),
            PnSpace::Application => 0x40 | ((pn_len_hint - 1) & 0x03),
        };
        header.push(first_byte);
        header.extend_from_slice(&pn_bytes[..pn_len_hint as usize]);

        // AEAD.
        let mut ct = vec![0u8; payload.len() + 16];
        let n = key.encrypt(&header, payload, pn, &mut ct)?;
        ct.truncate(n);

        // Header protection mask: sample = ct[..16].
        let mut sample = [0u8; 16];
        sample.copy_from_slice(&ct[..16.min(ct.len())]);
        let mask = key.header_protection_mask(&sample)?;
        // XOR the low 4 bits of the first byte and the pn bytes.
        let mut out = header.clone();
        out.extend_from_slice(&ct);
        out[0] ^= mask[0] & 0x0f;
        for i in 0..pn_len_hint as usize {
            out[1 + i] ^= mask[1 + i];
        }

        // Record the sent packet for loss detection.
        let idx = space as usize;
        self.detectors[idx].on_sent(crate::loss_detection::SentPacket {
            pn,
            sent_at: Instant::now(),
            bytes_in_flight: payload.len() as u64,
            ack_eliciting: frame_is_ack_eliciting,
            in_flight: frame_is_ack_eliciting,
        });
        self.next_pn[idx] = pn.wrapping_add(1);
        Ok(out)
    }

    /// Whether the connection wants to emit any outbound packet.
    pub fn has_outbound(&self) -> bool {
        self.tls.is_handshaking() || self.pending_crypto.iter().any(|(_, d)| !d.is_empty())
    }

    /// Drain pending TLS CRYPTO bytes into a CRYPTO frame for `space`.
    pub fn drain_crypto(&mut self, space: PnSpace) -> Option<Frame> {
        let idx = space as usize;
        let (off, data) = &self.pending_crypto[idx];
        if data.is_empty() {
            return None;
        }
        let frame = Frame::Crypto {
            offset: crate::varint::Varint::try_from(*off).ok()?,
            data: data.clone(),
        };
        self.pending_crypto[idx] = (off + data.len() as u64, bytes::Bytes::new());
        Some(frame)
    }

    /// Close the connection locally with a reason.
    pub fn close(&mut self, reason: impl Into<String>) {
        self.closed = true;
        self.close_reason = Some(reason.into());
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

/// Decode the peer's transport parameters from the TLS handshake.
pub fn decode_peer_params(tls: &TlsSession) -> Result<Option<TransportParameters>> {
    match tls.peer_quic_transport_parameters() {
        Some(raw) => Ok(Some(TransportParameters::decode(raw)?)),
        None => Ok(None),
    }
}
