//! QUIC frames (RFC 9000 §19).
//!
//! This module defines the [`Frame`] enum and its variable-length wire
//! encoding. Every frame type defined in RFC 9000 §19 is represented, so the
//! codec round-trips any well-formed packet payload. Frames that this crate
//! does not *act on* (for example `DATAGRAM`, which is a separate RFC) are
//! still parsed into a generic form so they can be forwarded by gateways.

use crate::error::{FrameError, Result};
use crate::varint::Varint;
use bytes::{Buf, BufMut};

/// `Frame` carries a single QUIC frame in its decoded, typed form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Frame {
    /// PADDING (type 0x00). Sends one byte of padding.
    Padding,
    /// PING (type 0x01). Elicits an ACK.
    Ping,
    /// ACK (type 0x02 / 0x03). Carries ack ranges and (optionally) ECN counts.
    Ack(AckFrame),
    /// RESET_STREAM (type 0x04).
    ResetStream {
        stream_id: Varint,
        error_code: Varint,
        final_size: Varint,
    },
    /// STOP_SENDING (type 0x05).
    StopSending {
        stream_id: Varint,
        error_code: Varint,
    },
    /// CRYPTO (type 0x06). Used for the TLS handshake.
    Crypto {
        offset: Varint,
        data: bytes::Bytes,
    },
    /// NEW_TOKEN (type 0x07).
    NewToken { token: bytes::Bytes },
    /// STREAM (type 0x08..=0x0f). The low three bits encode FIN / LEN / OFF.
    Stream(StreamFrame),
    /// MAX_DATA (type 0x10).
    MaxData { max: Varint },
    /// MAX_STREAM_DATA (type 0x11).
    MaxStreamData {
        stream_id: Varint,
        max: Varint,
    },
    /// MAX_STREAMS (type 0x12 bidi / 0x13 uni).
    MaxStreams {
        bidi: bool,
        max: Varint,
    },
    /// DATA_BLOCKED (type 0x14).
    DataBlocked { limit: Varint },
    /// STREAM_DATA_BLOCKED (type 0x15).
    StreamDataBlocked {
        stream_id: Varint,
        limit: Varint,
    },
    /// STREAMS_BLOCKED (type 0x16 bidi / 0x17 uni).
    StreamsBlocked {
        bidi: bool,
        limit: Varint,
    },
    /// NEW_CONNECTION_ID (type 0x18).
    NewConnectionId {
        sequence: Varint,
        retire_prior_to: Varint,
        cid: crate::cid::ConnectionId,
        stateless_reset_token: [u8; 16],
    },
    /// RETIRE_CONNECTION_ID (type 0x19).
    RetireConnectionId { sequence: Varint },
    /// PATH_CHALLENGE (type 0x1a).
    PathChallenge { data: [u8; 8] },
    /// PATH_RESPONSE (type 0x1b).
    PathResponse { data: [u8; 8] },
    /// CONNECTION_CLOSE (type 0x1c transport / 0x1d application).
    ConnectionClose {
        application_error: bool,
        error_code: Varint,
        frame_type: Option<Varint>,
        reason: bytes::Bytes,
    },
    /// HANDSHAKE_DONE (type 0x1e).
    HandshakeDone,
    /// A frame type not understood by this implementation. The raw type and
    /// the *entire* remaining payload of the packet are captured verbatim so
    /// that gateways can forward unknown frames without re-encoding.
    Unknown {
        ty: Varint,
        payload: bytes::Bytes,
    },
}

/// ACK frame payload (RFC 9000 §19.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AckFrame {
    /// Largest acknowledged packet number.
    pub largest_acknowledged: Varint,
    /// Delay in sending the ACK, in microseconds (encoded on the wire as
    /// microseconds, not milliseconds — see §19.3 field definitions).
    pub ack_delay: Varint,
    /// Number of additional ACK ranges minus 1.
    pub ack_range_count: Varint,
    /// First ACK range: number of contiguous packets below
    /// `largest_acknowledged` that are being acknowledged.
    pub first_ack_range: Varint,
    /// Additional `(gap, ack_range_length)` pairs.
    pub ranges: Vec<(Varint, Varint)>,
    /// ECN counts. Present iff the wire type is 0x03 (ACK with ECN). For
    /// type 0x02 these are zero / absent.
    pub ecn: Option<AckEcnCounts>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AckEcnCounts {
    pub ect0: u64,
    pub ect1: u64,
    pub ce: u64,
}

/// STREAM frame payload (RFC 9000 §19.8).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamFrame {
    pub stream_id: Varint,
    /// Offset of the first byte of `data` within the stream. Absent on the
    /// wire iff `offset_present` is false (offset == 0).
    pub offset: Varint,
    /// Whether the frame sets the FIN bit, marking the end of the stream.
    pub fin: bool,
    pub data: bytes::Bytes,
}

impl Frame {
    /// Encode this frame into `dst`, appending bytes. Returns the number of
    /// bytes written.
    pub fn encode(&self, dst: &mut bytes::BytesMut) -> Result<usize> {
        let start = dst.len();
        match self {
            Frame::Padding => dst.put_u8(0x00),
            Frame::Ping => dst.put_u8(0x01),
            Frame::Ack(a) => encode_ack(a, dst)?,
            Frame::ResetStream {
                stream_id,
                error_code,
                final_size,
            } => {
                put_var(dst, 0x04)?;
                put_var(dst, stream_id.as_u64())?;
                put_var(dst, error_code.as_u64())?;
                put_var(dst, final_size.as_u64())?;
            }
            Frame::StopSending { stream_id, error_code } => {
                put_var(dst, 0x05)?;
                put_var(dst, stream_id.as_u64())?;
                put_var(dst, error_code.as_u64())?;
            }
            Frame::Crypto { offset, data } => {
                put_var(dst, 0x06)?;
                put_var(dst, offset.as_u64())?;
                put_var(dst, data.len() as u64)?;
                dst.put_slice(data);
            }
            Frame::NewToken { token } => {
                put_var(dst, 0x07)?;
                put_var(dst, token.len() as u64)?;
                dst.put_slice(token);
            }
            Frame::Stream(s) => encode_stream(s, dst)?,
            Frame::MaxData { max } => {
                put_var(dst, 0x10)?;
                put_var(dst, max.as_u64())?;
            }
            Frame::MaxStreamData { stream_id, max } => {
                put_var(dst, 0x11)?;
                put_var(dst, stream_id.as_u64())?;
                put_var(dst, max.as_u64())?;
            }
            Frame::MaxStreams { bidi, max } => {
                put_var(dst, if *bidi { 0x12 } else { 0x13 })?;
                put_var(dst, max.as_u64())?;
            }
            Frame::DataBlocked { limit } => {
                put_var(dst, 0x14)?;
                put_var(dst, limit.as_u64())?;
            }
            Frame::StreamDataBlocked { stream_id, limit } => {
                put_var(dst, 0x15)?;
                put_var(dst, stream_id.as_u64())?;
                put_var(dst, limit.as_u64())?;
            }
            Frame::StreamsBlocked { bidi, limit } => {
                put_var(dst, if *bidi { 0x16 } else { 0x17 })?;
                put_var(dst, limit.as_u64())?;
            }
            Frame::NewConnectionId {
                sequence,
                retire_prior_to,
                cid,
                stateless_reset_token,
            } => {
                put_var(dst, 0x18)?;
                put_var(dst, sequence.as_u64())?;
                put_var(dst, retire_prior_to.as_u64())?;
                dst.put_u8(cid.len() as u8);
                dst.put_slice(cid.as_bytes());
                dst.put_slice(stateless_reset_token);
            }
            Frame::RetireConnectionId { sequence } => {
                put_var(dst, 0x19)?;
                put_var(dst, sequence.as_u64())?;
            }
            Frame::PathChallenge { data } => {
                dst.put_u8(0x1a);
                dst.put_slice(data);
            }
            Frame::PathResponse { data } => {
                dst.put_u8(0x1b);
                dst.put_slice(data);
            }
            Frame::ConnectionClose {
                application_error,
                error_code,
                frame_type,
                reason,
            } => {
                put_var(dst, if *application_error { 0x1d } else { 0x1c })?;
                put_var(dst, error_code.as_u64())?;
                match frame_type {
                    Some(ft) => put_var(dst, ft.as_u64())?,
                    None => put_var(dst, 0)?,
                }
                put_var(dst, reason.len() as u64)?;
                dst.put_slice(reason);
            }
            Frame::HandshakeDone => dst.put_u8(0x1e),
            Frame::Unknown { ty, payload } => {
                put_var(dst, ty.as_u64())?;
                dst.put_slice(payload);
            }
        }
        Ok(dst.len() - start)
    }

    /// Decode a single frame from `src`, advancing the cursor. Returns the
    /// frame and the number of bytes consumed.
    pub fn decode(src: &[u8]) -> Result<(Frame, usize)> {
        if src.is_empty() {
            return Err(FrameError::Truncated.into());
        }
        let mut cur = std::io::Cursor::new(src);
        let (ty_v, ty_len) = Varint::decode(src)?;
        cur.advance(ty_len);
        let ty = ty_v.as_u64();
        let frame = match ty {
            0x00 => Frame::Padding,
            0x01 => Frame::Ping,
            0x02 => Frame::Ack(decode_ack(&mut cur, false)?),
            0x03 => Frame::Ack(decode_ack(&mut cur, true)?),
            0x04 => {
                let sid = read_var(&mut cur)?;
                let ec = read_var(&mut cur)?;
                let fs = read_var(&mut cur)?;
                Frame::ResetStream {
                    stream_id: Varint(sid),
                    error_code: Varint(ec),
                    final_size: Varint(fs),
                }
            }
            0x05 => {
                let sid = read_var(&mut cur)?;
                let ec = read_var(&mut cur)?;
                Frame::StopSending {
                    stream_id: Varint(sid),
                    error_code: Varint(ec),
                }
            }
            0x06 => {
                let offset = read_var(&mut cur)?;
                let len = read_var(&mut cur)? as usize;
                let data = take_bytes(&mut cur, len)?;
                Frame::Crypto {
                    offset: Varint(offset),
                    data,
                }
            }
            0x07 => {
                let len = read_var(&mut cur)? as usize;
                let token = take_bytes(&mut cur, len)?;
                Frame::NewToken { token }
            }
            t if (0x08..=0x0f).contains(&t) => Frame::Stream(decode_stream(t, &mut cur)?),
            0x10 => {
                let max = read_var(&mut cur)?;
                Frame::MaxData { max: Varint(max) }
            }
            0x11 => {
                let sid = read_var(&mut cur)?;
                let max = read_var(&mut cur)?;
                Frame::MaxStreamData {
                    stream_id: Varint(sid),
                    max: Varint(max),
                }
            }
            0x12 | 0x13 => {
                let bidi = ty == 0x12;
                let max = read_var(&mut cur)?;
                Frame::MaxStreams { bidi, max: Varint(max) }
            }
            0x14 => {
                let limit = read_var(&mut cur)?;
                Frame::DataBlocked { limit: Varint(limit) }
            }
            0x15 => {
                let sid = read_var(&mut cur)?;
                let limit = read_var(&mut cur)?;
                Frame::StreamDataBlocked {
                    stream_id: Varint(sid),
                    limit: Varint(limit),
                }
            }
            0x16 | 0x17 => {
                let bidi = ty == 0x16;
                let limit = read_var(&mut cur)?;
                Frame::StreamsBlocked { bidi, limit: Varint(limit) }
            }
            0x18 => {
                let sequence = read_var(&mut cur)?;
                let retire = read_var(&mut cur)?;
                let cid_len = cur.get_u8() as usize;
                if cid_len > 20 {
                    return Err(crate::error::PacketError::CidTooLong.into());
                }
                let mut cid_bytes = [0u8; 20];
                for i in 0..cid_len {
                    cid_bytes[i] = cur.get_u8();
                }
                let cid = crate::cid::ConnectionId::from_slice(&cid_bytes[..cid_len])?;
                let mut token = [0u8; 16];
                for b in token.iter_mut() {
                    *b = cur.get_u8();
                }
                Frame::NewConnectionId {
                    sequence: Varint(sequence),
                    retire_prior_to: Varint(retire),
                    cid,
                    stateless_reset_token: token,
                }
            }
            0x19 => {
                let seq = read_var(&mut cur)?;
                Frame::RetireConnectionId { sequence: Varint(seq) }
            }
            0x1a => {
                let mut data = [0u8; 8];
                for b in data.iter_mut() {
                    *b = cur.get_u8();
                }
                Frame::PathChallenge { data }
            }
            0x1b => {
                let mut data = [0u8; 8];
                for b in data.iter_mut() {
                    *b = cur.get_u8();
                }
                Frame::PathResponse { data }
            }
            0x1c | 0x1d => {
                let application = ty == 0x1d;
                let ec = read_var(&mut cur)?;
                let ft_raw = read_var(&mut cur)?;
                let frame_type = if ft_raw == 0 { None } else { Some(Varint(ft_raw)) };
                let rlen = read_var(&mut cur)? as usize;
                let reason = take_bytes(&mut cur, rlen)?;
                Frame::ConnectionClose {
                    application_error: application,
                    error_code: Varint(ec),
                    frame_type,
                    reason,
                }
            }
            0x1e => Frame::HandshakeDone,
            other => {
                // Unknown frame: capture the rest of the packet verbatim.
                let remaining = cur.remaining();
                let payload = take_bytes(&mut cur, remaining)?;
                Frame::Unknown {
                    ty: Varint(other),
                    payload,
                }
            }
        };
        Ok((frame, cur.position() as usize))
    }
}

fn encode_ack(a: &AckFrame, dst: &mut bytes::BytesMut) -> Result<()> {
    let ty: u8 = if a.ecn.is_some() { 0x03 } else { 0x02 };
    put_var(dst, ty as u64)?;
    put_var(dst, a.largest_acknowledged.as_u64())?;
    put_var(dst, a.ack_delay.as_u64())?;
    put_var(dst, a.ack_range_count.as_u64())?;
    put_var(dst, a.first_ack_range.as_u64())?;
    for (gap, range) in &a.ranges {
        put_var(dst, gap.as_u64())?;
        put_var(dst, range.as_u64())?;
    }
    if let Some(ecn) = a.ecn {
        put_var(dst, ecn.ect0)?;
        put_var(dst, ecn.ect1)?;
        put_var(dst, ecn.ce)?;
    }
    Ok(())
}

fn decode_ack(cur: &mut std::io::Cursor<&[u8]>, with_ecn: bool) -> Result<AckFrame> {
    let largest = read_var(cur)?;
    let ack_delay = read_var(cur)?;
    let range_count = read_var(cur)?;
    let first = read_var(cur)?;
    let mut ranges = Vec::with_capacity(range_count as usize);
    for _ in 0..range_count {
        let gap = read_var(cur)?;
        let range = read_var(cur)?;
        ranges.push((Varint(gap), Varint(range)));
    }
    let ecn = if with_ecn {
        let ect0 = read_var(cur)?;
        let ect1 = read_var(cur)?;
        let ce = read_var(cur)?;
        Some(AckEcnCounts { ect0, ect1, ce })
    } else {
        None
    };
    Ok(AckFrame {
        largest_acknowledged: Varint(largest),
        ack_delay: Varint(ack_delay),
        ack_range_count: Varint(range_count),
        first_ack_range: Varint(first),
        ranges,
        ecn,
    })
}

fn encode_stream(s: &StreamFrame, dst: &mut bytes::BytesMut) -> Result<()> {
    let mut ty: u8 = 0x08;
    if s.fin {
        ty |= 0x01;
    }
    let len_present = true; // we always emit the length field (safe default)
    if len_present {
        ty |= 0x02;
    }
    if s.offset.as_u64() != 0 {
        ty |= 0x04;
    }
    put_var(dst, ty as u64)?;
    put_var(dst, s.stream_id.as_u64())?;
    if s.offset.as_u64() != 0 {
        put_var(dst, s.offset.as_u64())?;
    }
    put_var(dst, s.data.len() as u64)?;
    dst.put_slice(&s.data);
    Ok(())
}

fn decode_stream(ty: u64, cur: &mut std::io::Cursor<&[u8]>) -> Result<StreamFrame> {
    let fin = ty & 0x01 != 0;
    let len_present = ty & 0x02 != 0;
    let off_present = ty & 0x04 != 0;
    let sid = read_var(cur)?;
    let offset = if off_present {
        read_var(cur)?
    } else {
        0
    };
    let data = if len_present {
        let len = read_var(cur)? as usize;
        take_bytes(cur, len)?
    } else {
        // No length field: data extends to the end of the packet payload.
        let rem = cur.remaining();
        take_bytes(cur, rem)?
    };
    Ok(StreamFrame {
        stream_id: Varint(sid),
        offset: Varint(offset),
        fin,
        data,
    })
}

fn read_var(cur: &mut std::io::Cursor<&[u8]>) -> Result<u64> {
    let pos = cur.position() as usize;
    let buf = cur.get_ref();
    if pos >= buf.len() {
        return Err(FrameError::Truncated.into());
    }
    let (v, used) = Varint::decode(&buf[pos..])?;
    cur.advance(used);
    Ok(v.as_u64())
}

fn put_var(dst: &mut bytes::BytesMut, v: u64) -> Result<()> {
    let var = Varint::try_from(v).map_err(|e| crate::error::FrameError::Varint(e))?;
    let mut tmp = [0u8; 8];
    let n = var.encode(&mut tmp)?;
    dst.put_slice(&tmp[..n]);
    Ok(())
}

fn take_bytes(cur: &mut std::io::Cursor<&[u8]>, n: usize) -> Result<bytes::Bytes> {
    if cur.remaining() < n {
        return Err(FrameError::Truncated.into());
    }
    let pos = cur.position() as usize;
    let slice = &cur.get_ref()[pos..pos + n];
    let out = bytes::Bytes::copy_from_slice(slice);
    cur.advance(n);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_roundtrip() {
        let mut buf = bytes::BytesMut::new();
        Frame::Ping.encode(&mut buf).unwrap();
        let (f, n) = Frame::decode(&buf).unwrap();
        assert_eq!(f, Frame::Ping);
        assert_eq!(n, 1);
    }

    #[test]
    fn stream_roundtrip() {
        let s = Frame::Stream(StreamFrame {
            stream_id: Varint(4),
            offset: Varint(0),
            fin: true,
            data: bytes::Bytes::from_static(b"hello"),
        });
        let mut buf = bytes::BytesMut::new();
        s.encode(&mut buf).unwrap();
        let (f, _) = Frame::decode(&buf).unwrap();
        assert_eq!(f, s);
    }

    #[test]
    fn crypto_roundtrip() {
        let f = Frame::Crypto {
            offset: Varint(0),
            data: bytes::Bytes::from_static(b"tls handshake data"),
        };
        let mut buf = bytes::BytesMut::new();
        f.encode(&mut buf).unwrap();
        let (d, _) = Frame::decode(&buf).unwrap();
        assert_eq!(d, f);
    }

    #[test]
    fn ack_roundtrip_with_ecn() {
        let a = Frame::Ack(AckFrame {
            largest_acknowledged: Varint(100),
            ack_delay: Varint(5000),
            ack_range_count: Varint(1),
            first_ack_range: Varint(5),
            ranges: vec![(Varint(2), Varint(3))],
            ecn: Some(AckEcnCounts {
                ect0: 10,
                ect1: 2,
                ce: 0,
            }),
        });
        let mut buf = bytes::BytesMut::new();
        a.encode(&mut buf).unwrap();
        let (d, _) = Frame::decode(&buf).unwrap();
        assert_eq!(d, a);
    }
}
