//! QUIC transport parameters (RFC 9000 §18).
//!
//! Transport parameters are exchanged in the TLS handshake (carried in a
//! QUIC-specific TLS extension) and govern almost every connection-wide
//! limit: idle timeout, flow-control budgets, stream limits, CID rotation
//! policy, etc. This module provides a typed [`TransportParameters`] struct
//! with sensible defaults, plus an encoder/decoder that round-trips the full
//! set including unknown (grease) parameters.

use crate::cid::ConnectionId;
use crate::error::{Result, TransportParamError};
use crate::varint::Varint;
use bytes::{Buf, BufMut};

/// Numeric identifiers of the transport parameters defined in RFC 9000 §18.
/// These are varints, but every defined value fits in one byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum TransportParameterId {
    OriginalDcid = 0x00,
    MaxIdleTimeout = 0x01,
    StatelessResetToken = 0x02,
    MaxUdpPayload = 0x03,
    InitialMaxData = 0x04,
    InitialMaxStreamDataBidiLocal = 0x05,
    InitialMaxStreamDataBidiRemote = 0x06,
    InitialMaxStreamDataUni = 0x07,
    InitialMaxStreamsBidi = 0x08,
    InitialMaxStreamsUni = 0x09,
    AckDelayExponent = 0x0a,
    MaxAckDelay = 0x0b,
    DisableActiveMigration = 0x0c,
    PreferredAddress = 0x0d,
    ActiveConnectionIdLimit = 0x0e,
    InitialSourceConnectionId = 0x0f,
    RetrySourceConnectionId = 0x10,
    /// Grease / unknown extension.
    Other(u64),
}

impl TransportParameterId {
    fn from_u64(v: u64) -> Self {
        match v {
            0x00 => Self::OriginalDcid,
            0x01 => Self::MaxIdleTimeout,
            0x02 => Self::StatelessResetToken,
            0x03 => Self::MaxUdpPayload,
            0x04 => Self::InitialMaxData,
            0x05 => Self::InitialMaxStreamDataBidiLocal,
            0x06 => Self::InitialMaxStreamDataBidiRemote,
            0x07 => Self::InitialMaxStreamDataUni,
            0x08 => Self::InitialMaxStreamsBidi,
            0x09 => Self::InitialMaxStreamsUni,
            0x0a => Self::AckDelayExponent,
            0x0b => Self::MaxAckDelay,
            0x0c => Self::DisableActiveMigration,
            0x0d => Self::PreferredAddress,
            0x0e => Self::ActiveConnectionIdLimit,
            0x0f => Self::InitialSourceConnectionId,
            0x10 => Self::RetrySourceConnectionId,
            other => Self::Other(other),
        }
    }
}

/// Defaults recommended by RFC 9000 §18.2 where applicable.
impl Default for TransportParameters {
    fn default() -> Self {
        Self {
            original_dcid: None,
            max_idle_timeout_ms: 0,
            stateless_reset_token: None,
            max_udp_payload: 65527,
            initial_max_data: 0,
            initial_max_stream_data_bidi_local: 0,
            initial_max_stream_data_bidi_remote: 0,
            initial_max_stream_data_uni: 0,
            initial_max_streams_bidi: 0,
            initial_max_streams_uni: 0,
            ack_delay_exponent: 3,
            max_ack_delay_ms: 25,
            disable_active_migration: false,
            preferred_address: None,
            active_connection_id_limit: 2,
            initial_source_cid: None,
            retry_source_cid: None,
            grease: Vec::new(),
        }
    }
}

/// Typed view of a peer's transport parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportParameters {
    pub original_dcid: Option<ConnectionId>,
    pub max_idle_timeout_ms: u64,
    pub stateless_reset_token: Option<[u8; 16]>,
    pub max_udp_payload: u64,
    pub initial_max_data: u64,
    pub initial_max_stream_data_bidi_local: u64,
    pub initial_max_stream_data_bidi_remote: u64,
    pub initial_max_stream_data_uni: u64,
    pub initial_max_streams_bidi: u64,
    pub initial_max_streams_uni: u64,
    pub ack_delay_exponent: u64,
    pub max_ack_delay_ms: u64,
    pub disable_active_migration: bool,
    pub preferred_address: Option<bytes::Bytes>,
    pub active_connection_id_limit: u64,
    pub initial_source_cid: Option<ConnectionId>,
    pub retry_source_cid: Option<ConnectionId>,
    /// Round-tripped unknown parameters (id, raw value bytes).
    pub grease: Vec<(u64, bytes::Bytes)>,
}

impl TransportParameters {
    /// Encode into the on-wire form used inside the TLS
    /// `quic_transport_parameters` extension.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut dst = bytes::BytesMut::new();
        let mut push = |id: u64, val: &[u8]| -> Result<()> {
            put_var(&mut dst, id)?;
            put_var(&mut dst, val.len() as u64)?;
            dst.put_slice(val);
            Ok(())
        };
        if let Some(d) = &self.original_dcid {
            push(0x00, d.as_bytes())?;
        }
        if self.max_idle_timeout_ms != 0 {
            let v = enc_var(self.max_idle_timeout_ms)?;
            push(0x01, &v)?;
        }
        if let Some(t) = &self.stateless_reset_token {
            push(0x02, t)?;
        }
        if self.max_udp_payload != 65527 {
            let v = enc_var(self.max_udp_payload)?;
            push(0x03, &v)?;
        }
        if self.initial_max_data != 0 {
            push(0x04, &enc_var(self.initial_max_data)?)?;
        }
        if self.initial_max_stream_data_bidi_local != 0 {
            push(0x05, &enc_var(self.initial_max_stream_data_bidi_local)?)?;
        }
        if self.initial_max_stream_data_bidi_remote != 0 {
            push(0x06, &enc_var(self.initial_max_stream_data_bidi_remote)?)?;
        }
        if self.initial_max_stream_data_uni != 0 {
            push(0x07, &enc_var(self.initial_max_stream_data_uni)?)?;
        }
        if self.initial_max_streams_bidi != 0 {
            push(0x08, &enc_var(self.initial_max_streams_bidi)?)?;
        }
        if self.initial_max_streams_uni != 0 {
            push(0x09, &enc_var(self.initial_max_streams_uni)?)?;
        }
        if self.ack_delay_exponent != 3 {
            push(0x0a, &enc_var(self.ack_delay_exponent)?)?;
        }
        if self.max_ack_delay_ms != 25 {
            push(0x0b, &enc_var(self.max_ack_delay_ms)?)?;
        }
        if self.disable_active_migration {
            push(0x0c, &[])?;
        }
        if let Some(pa) = &self.preferred_address {
            push(0x0d, pa)?;
        }
        if self.active_connection_id_limit != 2 {
            push(0x0e, &enc_var(self.active_connection_id_limit)?)?;
        }
        if let Some(c) = &self.initial_source_cid {
            push(0x0f, c.as_bytes())?;
        }
        if let Some(c) = &self.retry_source_cid {
            push(0x10, c.as_bytes())?;
        }
        for (id, val) in &self.grease {
            push(*id, val)?;
        }
        Ok(dst.to_vec())
    }

    /// Decode the wire form. Unknown parameters are preserved in `grease`.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        let mut p = Self::default();
        let mut seen = std::collections::HashSet::new();
        let mut cur = std::io::Cursor::new(buf);
        while cur.remaining() > 0 {
            let id = read_var(&mut cur)?;
            let len = read_var(&mut cur)? as usize;
            if cur.remaining() < len {
                return Err(TransportParamError::Malformed.into());
            }
            let pos = cur.position() as usize;
            let val = &cur.get_ref()[pos..pos + len];
            cur.advance(len);
            if !seen.insert(id) {
                return Err(TransportParamError::Duplicate(id).into());
            }
            match TransportParameterId::from_u64(id) {
                TransportParameterId::OriginalDcid => {
                    p.original_dcid = Some(ConnectionId::from_slice(val)?);
                }
                TransportParameterId::MaxIdleTimeout => p.max_idle_timeout_ms = decode_var(val)?,
                TransportParameterId::StatelessResetToken => {
                    let mut t = [0u8; 16];
                    if val.len() != 16 {
                        return Err(TransportParamError::InvalidValue(
                            TransportParameterId::StatelessResetToken,
                            "must be 16 bytes".into(),
                        )
                        .into());
                    }
                    t.copy_from_slice(val);
                    p.stateless_reset_token = Some(t);
                }
                TransportParameterId::MaxUdpPayload => p.max_udp_payload = decode_var(val)?,
                TransportParameterId::InitialMaxData => {
                    p.initial_max_data = decode_var(val)?
                }
                TransportParameterId::InitialMaxStreamDataBidiLocal => {
                    p.initial_max_stream_data_bidi_local = decode_var(val)?
                }
                TransportParameterId::InitialMaxStreamDataBidiRemote => {
                    p.initial_max_stream_data_bidi_remote = decode_var(val)?
                }
                TransportParameterId::InitialMaxStreamDataUni => {
                    p.initial_max_stream_data_uni = decode_var(val)?
                }
                TransportParameterId::InitialMaxStreamsBidi => {
                    p.initial_max_streams_bidi = decode_var(val)?
                }
                TransportParameterId::InitialMaxStreamsUni => {
                    p.initial_max_streams_uni = decode_var(val)?
                }
                TransportParameterId::AckDelayExponent => {
                    p.ack_delay_exponent = decode_var(val)?;
                    if p.ack_delay_exponent > 20 {
                        return Err(TransportParamError::InvalidValue(
                            TransportParameterId::AckDelayExponent,
                            "must be <= 20".into(),
                        )
                        .into());
                    }
                }
                TransportParameterId::MaxAckDelay => {
                    p.max_ack_delay_ms = decode_var(val)?
                }
                TransportParameterId::DisableActiveMigration => {
                    p.disable_active_migration = true
                }
                TransportParameterId::PreferredAddress => {
                    p.preferred_address = Some(bytes::Bytes::copy_from_slice(val))
                }
                TransportParameterId::ActiveConnectionIdLimit => {
                    p.active_connection_id_limit = decode_var(val)?
                }
                TransportParameterId::InitialSourceConnectionId => {
                    p.initial_source_cid = Some(ConnectionId::from_slice(val)?)
                }
                TransportParameterId::RetrySourceConnectionId => {
                    p.retry_source_cid = Some(ConnectionId::from_slice(val)?)
                }
                TransportParameterId::Other(_) => {
                    p.grease
                        .push((id, bytes::Bytes::copy_from_slice(val)));
                }
            }
        }
        Ok(p)
    }
}

fn read_var(cur: &mut std::io::Cursor<&[u8]>) -> Result<u64> {
    let pos = cur.position() as usize;
    let buf = cur.get_ref();
    if pos >= buf.len() {
        return Err(TransportParamError::Malformed.into());
    }
    let (v, used) = Varint::decode(&buf[pos..])?;
    cur.advance(used);
    Ok(v.as_u64())
}

fn decode_var(b: &[u8]) -> Result<u64> {
    if b.is_empty() {
        return Err(TransportParamError::Malformed.into());
    }
    let (v, used) = Varint::decode(b)?;
    if used != b.len() {
        return Err(TransportParamError::Malformed.into());
    }
    Ok(v.as_u64())
}

fn enc_var(v: u64) -> Result<Vec<u8>> {
    let var = Varint::try_from(v).map_err(|_| TransportParamError::Malformed)?;
    let mut tmp = [0u8; 8];
    let n = var.encode(&mut tmp)?;
    Ok(tmp[..n].to_vec())
}

fn put_var(dst: &mut bytes::BytesMut, v: u64) -> Result<()> {
    let var = Varint::try_from(v).map_err(|_| TransportParamError::Malformed)?;
    let mut tmp = [0u8; 8];
    let n = var.encode(&mut tmp)?;
    dst.put_slice(&tmp[..n]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_defaults() {
        let p = TransportParameters::default();
        let wire = p.encode().unwrap();
        // Defaults emit nothing except grease-free; should be empty.
        assert!(wire.is_empty());
        let q = TransportParameters::decode(&wire).unwrap();
        assert_eq!(p, q);
    }

    #[test]
    fn roundtrip_with_values() {
        let mut p = TransportParameters::default();
        p.initial_max_data = 1 << 20;
        p.initial_max_streams_bidi = 100;
        p.max_idle_timeout_ms = 30_000;
        p.ack_delay_exponent = 5;
        p.initial_source_cid = Some(ConnectionId::from_slice(&[1, 2, 3, 4]).unwrap());
        let wire = p.encode().unwrap();
        let q = TransportParameters::decode(&wire).unwrap();
        assert_eq!(p, q);
    }
}
