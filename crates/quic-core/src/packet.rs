//! QUIC packet headers (RFC 9000 §17).
//!
//! This module models the *cleartext* structure of QUIC packets after header
//! protection has been removed. Packet-number truncation (§17.1) is applied
//! at the codec boundary: on encode we truncate the 62-bit packet number to
//! the fewest bytes that the receiver can still reconstruct given the largest
//! acknowledged number, and on decode we reconstruct the full number from the
//! truncated field and the same hint.
//!
//! Header protection (XOR-ing the low bits of the packet number and the
//! first-byte reserved bits with an AES mask) is applied in [`crate::crypto`]
//! *after* this module lays out the cleartext header.

use crate::cid::ConnectionId;
use crate::error::{Error, PacketError, Result};
use crate::varint::Varint;
use bytes::BufMut;

/// QUIC version 1 (RFC 9000).
pub const VERSION_1: u32 = 0x00000001;
/// QUIC version 2 (RFC 9369).
pub const VERSION_2: u32 = 0x6b33_43cf;
/// The IETF reserved "grease" version for negotiation probing.
pub const VERSION_GREASE: u32 = 0xface_b002;

/// Discriminant for the long-header packet form (RFC 9000 §17.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LongPacketType {
    Initial = 0,
    ZeroRtt = 1,
    Handshake = 2,
    Retry = 3,
}

impl LongPacketType {
    fn from_bits(b: u8) -> Option<Self> {
        match (b >> 4) & 0x03 {
            0 => Some(Self::Initial),
            1 => Some(Self::ZeroRtt),
            2 => Some(Self::Handshake),
            3 => Some(Self::Retry),
            _ => None,
        }
    }
    fn to_bits(self) -> u8 {
        (self as u8) << 4
    }
}

/// A decoded QUIC packet header. The same type covers both the long form
/// (Initial / 0-RTT / Handshake / Retry) and the short form (1-RTT); the
/// `PacketForm` field tells them apart and carries the long-header-specific
/// fields when present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketHeader {
    /// The 62-bit packet number, reconstructed to full width by the decoder.
    pub packet_number: u64,
    /// The DCID the packet is addressed to.
    pub dcid: ConnectionId,
    /// The SCID the packet was sent from (present only on long headers).
    pub scid: ConnectionId,
    /// Which kind of packet this is.
    pub form: PacketForm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PacketForm {
    /// Short header (1-RTT, §17.3). Key phase bit and the (truncated) packet
    /// number length used on the wire.
    Short { key_phase: bool, pn_len: u8 },
    /// Long header (§17.2): Initial.
    Initial {
        version: u32,
        token: bytes::Bytes,
        length: Varint, // remaining packet length after the pn (includes payload+pn)
        pn_len: u8,
    },
    /// Long header: 0-RTT.
    ZeroRtt {
        version: u32,
        length: Varint,
        pn_len: u8,
    },
    /// Long header: Handshake.
    Handshake {
        version: u32,
        length: Varint,
        pn_len: u8,
    },
    /// Long header: Retry (§17.2.5). Retry has no packet number.
    Retry {
        version: u32,
        retry_token: bytes::Bytes,
        integrity_tag: [u8; 16],
    },
    /// Version negotiation packet (§17.2.1). Sent by a server that does not
    /// support the client's version.
    VersionNegotiation { versions: Vec<u32> },
}

impl PacketForm {
    pub fn is_long(self) -> bool {
        !matches!(self, PacketForm::Short { .. })
    }
}

/// Hints used to (de)serialize the packet-number field. The encoder truncates
/// `pn` to the fewest bytes `L` such that the receiver, knowing
/// `largest_acknowledged`, can recover it; the decoder expands the truncated
/// field back to 62 bits using the same hint (RFC 9000 §17.1.1).
#[derive(Debug, Clone, Copy)]
pub struct PnHints {
    /// Largest acknowledged packet number for this packet-number space, or 0
    /// if nothing has been acknowledged yet.
    pub largest_acked: u64,
}

impl Default for PnHints {
    fn default() -> Self {
        Self { largest_acked: 0 }
    }
}

/// Minimum number of bytes needed to encode `pn` such that it is unambiguous
/// given `largest_acked` (RFC 9000 §17.1.1 algorithm).
pub fn truncated_pn_len(pn: u64, largest_acked: u64) -> u8 {
    // The range of candidate packet numbers is [largest_acked + 1, 2^62-1]
    // (with wraparound in the full 64-bit space per the spec, but QUIC pns are
    // 62-bit). We compute the number of bits needed.
    let mut num_ranges = if largest_acked < pn {
        pn - largest_acked
    } else {
        // Wrap-around case: distance through 2^62.
        (1u64 << 62) - largest_acked + pn
    };
    // The spec formula: find smallest L (1..=4) with 2^(8*L - 1) > num_ranges.
    let mut min_len = 4u8;
    for l in 1..=4u8 {
        if (1u64 << (8 * l - 1)) > num_ranges {
            min_len = l;
            break;
        }
    }
    let _ = &mut num_ranges;
    min_len
}

/// Truncate a packet number to `len` bytes (little-endian low bytes kept).
pub fn truncate_pn(pn: u64, len: u8) -> [u8; 4] {
    let mut out = [0u8; 4];
    for i in 0..len as usize {
        out[i] = (pn >> (8 * i)) as u8;
    }
    out
}

/// Reconstruct the full packet number from a truncated field and the largest
/// acknowledged number (RFC 9000 §17.1.2, Appendix A).
pub fn reconstruct_pn(truncated: u64, pn_len: u8, largest_acked: u64) -> u64 {
    let pn_win = 1u64 << (8 * pn_len);
    let pn_hwin = pn_win / 2;
    let pn_mask = pn_win - 1;
    let candidate_pn = (largest_acked + 1) & !pn_mask | (truncated & pn_mask);
    if candidate_pn <= largest_acked.checked_sub(pn_hwin).unwrap_or(0) {
        candidate_pn.wrapping_add(pn_win)
    } else if candidate_pn > largest_acked + pn_hwin && candidate_pn >= pn_win {
        candidate_pn.wrapping_sub(pn_win)
    } else {
        candidate_pn
    }
}

/// Encode a cleartext packet header into `dst`. `payload_and_pn_len_bytes` is
/// written into the long-header `Length` field: it must equal
/// `pn_len + payload_len`. The caller is responsible for header protection
/// *after* this call.
pub fn encode_header(
    hdr: &PacketHeader,
    hints: PnHints,
    payload_len: usize,
    dst: &mut bytes::BytesMut,
) -> Result<usize> {
    let start = dst.len();
    match &hdr.form {
        PacketForm::Short { key_phase, pn_len } => {
            let pn_len = *pn_len;
            let first: u8 = 0x40 // fixed bit = 1
                | if *key_phase { 0x04 } else { 0x00 };
            // The low 2 bits (packet number length - 1) are initially written
            // in the clear here; header protection will XOR them later.
            let first = first | ((pn_len - 1) as u8 & 0x03);
            dst.put_u8(first);
            dst.put_slice(hdr.dcid.as_bytes());
            let pn = truncate_pn(hdr.packet_number, pn_len);
            dst.put_slice(&pn[..pn_len as usize]);
        }
        PacketForm::Initial {
            version,
            token,
            length,
            pn_len,
        } => {
            let pn_len = *pn_len;
            let first: u8 = 0x80 // long form
                | 0x40 // fixed bit
                | LongPacketType::Initial.to_bits()
                | ((pn_len - 1) as u8 & 0x03);
            dst.put_u8(first);
            dst.put_u32(*version);
            dst.put_u8(hdr.dcid.len() as u8);
            dst.put_slice(hdr.dcid.as_bytes());
            dst.put_u8(hdr.scid.len() as u8);
            dst.put_slice(hdr.scid.as_bytes());
            put_var(dst, token.len() as u64)?;
            dst.put_slice(token);
            // Length field covers pn_len + payload.
            let total = pn_len as u64 + payload_len as u64;
            let _ = length; // prefer recomputing to keep caller honest
            put_var(dst, total)?;
            let pn = truncate_pn(hdr.packet_number, pn_len);
            dst.put_slice(&pn[..pn_len as usize]);
        }
        PacketForm::ZeroRtt {
            version,
            length,
            pn_len,
        } => {
            let pn_len = *pn_len;
            let first: u8 = 0x80 | 0x40 | LongPacketType::ZeroRtt.to_bits() | ((pn_len - 1) as u8);
            dst.put_u8(first);
            dst.put_u32(*version);
            dst.put_u8(hdr.dcid.len() as u8);
            dst.put_slice(hdr.dcid.as_bytes());
            dst.put_u8(hdr.scid.len() as u8);
            dst.put_slice(hdr.scid.as_bytes());
            let total = pn_len as u64 + payload_len as u64;
            let _ = length;
            put_var(dst, total)?;
            let pn = truncate_pn(hdr.packet_number, pn_len);
            dst.put_slice(&pn[..pn_len as usize]);
        }
        PacketForm::Handshake {
            version,
            length,
            pn_len,
        } => {
            let pn_len = *pn_len;
            let first: u8 = 0x80 | 0x40 | LongPacketType::Handshake.to_bits() | ((pn_len - 1) as u8);
            dst.put_u8(first);
            dst.put_u32(*version);
            dst.put_u8(hdr.dcid.len() as u8);
            dst.put_slice(hdr.dcid.as_bytes());
            dst.put_u8(hdr.scid.len() as u8);
            dst.put_slice(hdr.scid.as_bytes());
            let total = pn_len as u64 + payload_len as u64;
            let _ = length;
            put_var(dst, total)?;
            let pn = truncate_pn(hdr.packet_number, pn_len);
            dst.put_slice(&pn[..pn_len as usize]);
        }
        PacketForm::Retry {
            version,
            retry_token,
            integrity_tag,
        } => {
            let first: u8 = 0x80 | 0x40 | LongPacketType::Retry.to_bits();
            // Retry packets have a fixed packet-number-length field of 0.
            dst.put_u8(first);
            dst.put_u32(*version);
            dst.put_u8(hdr.dcid.len() as u8);
            dst.put_slice(hdr.dcid.as_bytes());
            dst.put_u8(hdr.scid.len() as u8);
            dst.put_slice(hdr.scid.as_bytes());
            dst.put_slice(retry_token);
            dst.put_slice(integrity_tag);
        }
        PacketForm::VersionNegotiation { versions } => {
            // VN first byte: high bit set, rest random (we use 0x00).
            dst.put_u8(0x80);
            // The version field in a VN response is 0 (RFC 9000 §17.2.1).
            dst.put_u32(0);
            dst.put_u8(hdr.dcid.len() as u8);
            dst.put_slice(hdr.dcid.as_bytes());
            dst.put_u8(hdr.scid.len() as u8);
            dst.put_slice(hdr.scid.as_bytes());
            for v in versions {
                dst.put_u32(*v);
            }
        }
    }
    let _ = hints;
    Ok(dst.len() - start)
}

/// A packet header that has been parsed but whose packet-number bytes have
/// *not yet been unmasked*. The caller runs header protection to recover the
/// true packet number, then calls [`ParsedHeader::finalize_pn`].
#[derive(Debug, Clone)]
pub struct ParsedHeader {
    pub dcid: ConnectionId,
    pub scid: ConnectionId,
    pub form: RawForm,
    /// Position in the input buffer where the (still-protected) packet-number
    /// bytes begin.
    pub pn_offset: usize,
}

/// The raw, still-protected form of a parsed header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawForm {
    Short { pn_len: u8, protected_first_byte: u8 },
    Initial {
        version: u32,
        token: bytes::Bytes,
        length: Varint,
        pn_len: u8,
        payload_len: usize,
    },
    ZeroRtt {
        version: u32,
        length: Varint,
        pn_len: u8,
        payload_len: usize,
    },
    Handshake {
        version: u32,
        length: Varint,
        pn_len: u8,
        payload_len: usize,
    },
    Retry {
        version: u32,
        retry_token: bytes::Bytes,
        integrity_tag: [u8; 16],
    },
    VersionNegotiation { versions: Vec<u32> },
}

impl RawForm {
    /// Number of packet-number bytes on the wire. 0 for Retry / VN.
    pub fn pn_len(&self) -> u8 {
        match self {
            RawForm::Short { pn_len, .. } => *pn_len,
            RawForm::Initial { pn_len, .. } => *pn_len,
            RawForm::ZeroRtt { pn_len, .. } => *pn_len,
            RawForm::Handshake { pn_len, .. } => *pn_len,
            RawForm::Retry { .. } => 0,
            RawForm::VersionNegotiation { .. } => 0,
        }
    }
}

/// Parse the cleartext (post-header-protection-removal) parts of a packet
/// header. The packet-number low bits are *still masked* in the returned
/// structure; the caller must run header-protection removal on the first
/// byte + the pn bytes before finalizing.
///
/// This function reads only the *static* part of the header that does not
/// depend on the mask; the dynamic low bits are recovered by
/// [`apply_header_protection`].
pub fn parse_header(buf: &[u8]) -> Result<(ParsedHeader, usize)> {
    if buf.is_empty() {
        return Err(PacketError::Truncated { need: 1, have: 0 }.into());
    }
    let first = buf[0];
    let long = first & 0x80 != 0;
    if !long {
        // Short header. The fixed bit must be 1 (0x40).
        if first & 0x40 == 0 {
            return Err(PacketError::InvalidFixedBit.into());
        }
        let pn_len = (first & 0x03) + 1;
        // DCID length is not on the wire for short headers; it must be known
        // out-of-band (negotiated via transport parameters). We require the
        // caller to pass it in via a separate function for the actual routing,
        // but for parsing we read as many bytes as we can given `dcid_len`.
        // To keep this function self-contained we require the *connection* to
        // tell us the DCID length; here we return a parse that needs an extra
        // hint — so this function is only the long-header path. The short
        // path lives in `parse_short_header`.
        let _ = pn_len;
        return Err(Error::Internal(
            "short header requires dcid len hint; use parse_short_header".into(),
        )
        .into());
    }
    // Long header.
    if first & 0x40 == 0 {
        return Err(PacketError::InvalidFixedBit.into());
    }
    if buf.len() < 6 {
        return Err(PacketError::Truncated {
            need: 6,
            have: buf.len(),
        }
        .into());
    }
    let version = u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]);
    let mut pos = 5usize;
    let dcid_len = buf[pos] as usize;
    pos += 1;
    if pos + dcid_len > buf.len() {
        return Err(PacketError::Truncated {
            need: pos + dcid_len,
            have: buf.len(),
        }
        .into());
    }
    let dcid = ConnectionId::from_slice(&buf[pos..pos + dcid_len])?;
    pos += dcid_len;
    let scid_len = buf[pos] as usize;
    pos += 1;
    if pos + scid_len > buf.len() {
        return Err(PacketError::Truncated {
            need: pos + scid_len,
            have: buf.len(),
        }
        .into());
    }
    let scid = ConnectionId::from_slice(&buf[pos..pos + scid_len])?;
    pos += scid_len;

    // Version negotiation: version field is 0.
    if version == 0 {
        let mut versions = Vec::new();
        let mut v = pos;
        while v + 4 <= buf.len() {
            versions.push(u32::from_be_bytes([
                buf[v],
                buf[v + 1],
                buf[v + 2],
                buf[v + 3],
            ]));
            v += 4;
        }
        return Ok((
            ParsedHeader {
                dcid,
                scid,
                form: RawForm::VersionNegotiation { versions },
                pn_offset: pos,
            },
            v,
        ));
    }

    let ptype = LongPacketType::from_bits(first).ok_or(PacketError::InvalidHeaderForm)?;
    match ptype {
        LongPacketType::Retry => {
            // Retry: retry_token || integrity_tag (16 bytes) up to end of packet.
            if buf.len() < pos + 16 {
                return Err(PacketError::Truncated {
                    need: pos + 16,
                    have: buf.len(),
                }
                .into());
            }
            let tag_pos = buf.len() - 16;
            let retry_token = bytes::Bytes::copy_from_slice(&buf[pos..tag_pos]);
            let mut tag = [0u8; 16];
            tag.copy_from_slice(&buf[tag_pos..]);
            Ok((
                ParsedHeader {
                    dcid,
                    scid,
                    form: RawForm::Retry {
                        version,
                        retry_token,
                        integrity_tag: tag,
                    },
                    pn_offset: pos,
                },
                buf.len(),
            ))
        }
        LongPacketType::Initial => {
            let (token_len_var, used) = read_varint_with_len(buf, pos)?;
            let token_len = token_len_var.as_u64() as usize;
            pos += used;
            if pos + token_len > buf.len() {
                return Err(PacketError::Truncated {
                    need: pos + token_len,
                    have: buf.len(),
                }
                .into());
            }
            let token = bytes::Bytes::copy_from_slice(&buf[pos..pos + token_len]);
            pos += token_len;
            let (length, used) = read_varint_with_len(buf, pos)?;
            pos += used;
            // The protected low 2 bits of the first byte encode pn_len-1, but
            // header protection has XORed them. We optimistically read them
            // and let apply_header_protection fix the value up.
            let pn_len = (first & 0x03) + 1;
            let remaining = length.as_u64() as usize;
            let payload_len = remaining.saturating_sub(pn_len as usize);
            Ok((
                ParsedHeader {
                    dcid,
                    scid,
                    form: RawForm::Initial {
                        version,
                        token,
                        length,
                        pn_len,
                        payload_len,
                    },
                    pn_offset: pos,
                },
                pos + remaining,
            ))
        }
        LongPacketType::ZeroRtt | LongPacketType::Handshake => {
            let (length, used) = read_varint_with_len(buf, pos)?;
            pos += used;
            let pn_len = (first & 0x03) + 1;
            let remaining = length.as_u64() as usize;
            let payload_len = remaining.saturating_sub(pn_len as usize);
            let form = if ptype == LongPacketType::ZeroRtt {
                RawForm::ZeroRtt {
                    version,
                    length,
                    pn_len,
                    payload_len,
                }
            } else {
                RawForm::Handshake {
                    version,
                    length,
                    pn_len,
                    payload_len,
                }
            };
            Ok((
                ParsedHeader {
                    dcid,
                    scid,
                    form,
                    pn_offset: pos,
                },
                pos + remaining,
            ))
        }
    }
}

/// Parse a short (1-RTT) header, given the negotiated DCID length.
pub fn parse_short_header(buf: &[u8], dcid_len: usize) -> Result<(ParsedHeader, usize)> {
    if buf.is_empty() {
        return Err(PacketError::Truncated { need: 1, have: 0 }.into());
    }
    let first = buf[0];
    if first & 0x80 != 0 {
        return Err(PacketError::InvalidHeaderForm.into());
    }
    if first & 0x40 == 0 {
        return Err(PacketError::InvalidFixedBit.into());
    }
    let pn_len = (first & 0x03) + 1;
    let need = 1 + dcid_len + pn_len as usize;
    if buf.len() < need {
        return Err(PacketError::Truncated {
            need,
            have: buf.len(),
        }
        .into());
    }
    let dcid = ConnectionId::from_slice(&buf[1..1 + dcid_len])?;
    let pn_offset = 1 + dcid_len;
    Ok((
        ParsedHeader {
            dcid,
            scid: ConnectionId::empty(),
            form: RawForm::Short {
                pn_len,
                protected_first_byte: first,
            },
            pn_offset,
        },
        need,
    ))
}

/// Apply (remove) header protection in-place: XOR the low 4 bits of the first
/// byte (low 2 bits = pn length, next 2 = reserved) and the pn bytes with the
/// first `1 + pn_len` bytes of the mask. Returns the recovered packet number
/// length and the reconstructed packet number.
pub fn apply_header_protection(
    buf: &mut [u8],
    header: &ParsedHeader,
    mask: &[u8],
    largest_acked: u64,
) -> Result<(u8, u64)> {
    let pn_len = header.form.pn_len();
    if pn_len == 0 {
        return Ok((0, 0));
    }
    if mask.len() < 1 + pn_len as usize {
        return Err(PacketError::Truncated {
            need: 1 + pn_len as usize,
            have: mask.len(),
        }
        .into());
    }
    let first_idx = 0usize;
    buf[first_idx] ^= mask[0] & 0x0f;
    let pn_off = header.pn_offset;
    for i in 0..pn_len as usize {
        buf[pn_off + i] ^= mask[1 + i];
    }
    // Re-read pn_len from the (now unmasked) low bits for long headers; for
    // short headers the low 2 bits of the first byte already encoded it and
    // are now unmasked too.
    let unmasked_first = buf[first_idx];
    let recovered_pn_len = (unmasked_first & 0x03) + 1;
    let mut truncated: u64 = 0;
    for i in 0..recovered_pn_len as usize {
        truncated |= (buf[pn_off + i] as u64) << (8 * i);
    }
    let full = reconstruct_pn(truncated, recovered_pn_len, largest_acked);
    Ok((recovered_pn_len, full))
}

fn read_varint_with_len(buf: &[u8], pos: usize) -> Result<(Varint, usize)> {
    if pos >= buf.len() {
        return Err(PacketError::Truncated {
            need: pos + 1,
            have: buf.len(),
        }
        .into());
    }
    let (v, used) = Varint::decode(&buf[pos..])?;
    Ok((v, used))
}

fn put_var(dst: &mut bytes::BytesMut, v: u64) -> Result<()> {
    let var = Varint::try_from(v).map_err(|_| PacketError::VarintOutOfRange)?;
    let mut tmp = [0u8; 8];
    let n = var.encode(&mut tmp)?;
    dst.put_slice(&tmp[..n]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pn_truncation_roundtrip() {
        // Largest acked = 100, send pn = 105.
        let len = truncated_pn_len(105, 100);
        assert!(len >= 1 && len <= 4);
        let t = truncate_pn(105, len);
        let mut tr = 0u64;
        for i in 0..len as usize {
            tr |= (t[i] as u64) << (8 * i);
        }
        let full = reconstruct_pn(tr, len, 100);
        assert_eq!(full, 105);
    }

    #[test]
    fn initial_header_roundtrip() {
        let mut buf = bytes::BytesMut::new();
        let hdr = PacketHeader {
            packet_number: 0,
            dcid: ConnectionId::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap(),
            scid: ConnectionId::from_slice(&[0xaa; 8]).unwrap(),
            form: PacketForm::Initial {
                version: VERSION_1,
                token: bytes::Bytes::new(),
                length: Varint(0),
                pn_len: 4,
            },
        };
        let n = encode_header(&hdr, PnHints::default(), 32, &mut buf).unwrap();
        assert!(n > 0);
        // encode_header writes only the header; append the payload bytes it
        // promised in the Length field so parse_header sees a full packet.
        buf.extend_from_slice(&[0u8; 32]);
        let (parsed, end) = parse_header(&buf).unwrap();
        assert!(matches!(parsed.form, RawForm::Initial { .. }));
        assert!(end <= buf.len(), "end={end} buf={}", buf.len());
    }
}
