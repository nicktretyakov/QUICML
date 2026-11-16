//! QUIC variable-length integers (RFC 9000 §16).
//!
//! A QUIC varint encodes unsigned integers up to 2^62-1 in 1, 2, 4 or 8 bytes.
//! The two high bits of the first byte encode the length:
//!
//! ```text
//!  2MSB | length | usable bits
//!  ------+--------+-------------
//!   00   |   1    |     6
//!   01   |   2    |    14
//!   10   |   4    |    30
//!   11   |   8    |    62
//! ```

use crate::error::{PacketError, Result};

/// Maximum value representable as a QUIC varint: 2^62 - 1.
pub const MAX_VARINT: u64 = (1u64 << 62) - 1;

/// Error returned when a value cannot be encoded as a varint.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("varint out of range: {0} > 2^62-1")]
pub struct VarintError(pub u64);

/// A typed QUIC variable-length integer.
///
/// Wrapping the integer in a newtype makes it impossible to accidentally pass
/// a raw `u64` that exceeds the 62-bit range into wire-encoding code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Varint(pub u64);

impl Varint {
    /// Construct a varint, returning an error if the value exceeds 2^62-1.
    pub const fn new(v: u64) -> std::result::Result<Self, VarintError> {
        if v <= MAX_VARINT {
            Ok(Varint(v))
        } else {
            Err(VarintError(v))
        }
    }

    /// Construct a varint without bounds-checking. The caller MUST ensure
    /// `v <= 2^62-1`. Used only by internal const contexts.
    pub const fn from_u64_unchecked(v: u64) -> Self {
        Varint(v)
    }

    /// The underlying integer value.
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Number of bytes the varint occupies on the wire for this value.
    pub const fn encoded_len(self) -> usize {
        if self.0 < (1 << 6) {
            1
        } else if self.0 < (1 << 14) {
            2
        } else if self.0 < (1 << 30) {
            4
        } else {
            8
        }
    }

    /// Encode this varint into `out`, returning the number of bytes written.
    /// Returns an error if `out` is too small.
    pub fn encode(self, out: &mut [u8]) -> Result<usize> {
        let len = self.encoded_len();
        if out.len() < len {
            return Err(PacketError::Truncated {
                need: len,
                have: out.len(),
            }
            .into());
        }
        let prefix: u8 = match len {
            1 => 0b00,
            2 => 0b01,
            4 => 0b10,
            8 => 0b11,
            _ => unreachable!(),
        } << 6;
        let v = self.0;
        match len {
            1 => out[0] = prefix | (v as u8),
            2 => {
                out[0] = prefix | ((v >> 8) as u8);
                out[1] = v as u8;
            }
            4 => {
                out[0] = prefix | ((v >> 24) as u8);
                out[1] = (v >> 16) as u8;
                out[2] = (v >> 8) as u8;
                out[3] = v as u8;
            }
            8 => {
                out[0] = prefix | ((v >> 56) as u8);
                out[1] = (v >> 48) as u8;
                out[2] = (v >> 40) as u8;
                out[3] = (v >> 32) as u8;
                out[4] = (v >> 24) as u8;
                out[5] = (v >> 16) as u8;
                out[6] = (v >> 8) as u8;
                out[7] = v as u8;
            }
            _ => unreachable!(),
        }
        Ok(len)
    }

    /// Decode a varint from `buf`, returning the value and the number of
    /// bytes consumed.
    pub fn decode(buf: &[u8]) -> Result<(Self, usize)> {
        if buf.is_empty() {
            return Err(PacketError::Truncated { need: 1, have: 0 }.into());
        }
        let tag = buf[0] >> 6;
        let len = 1usize << tag; // 1, 2, 4, 8
        if buf.len() < len {
            return Err(PacketError::Truncated {
                need: len,
                have: buf.len(),
            }
            .into());
        }
        // Mask off the 2-bit length prefix and assemble the value.
        let mut v: u64 = (buf[0] & 0x3f) as u64;
        for &b in &buf[1..len] {
            v = (v << 8) | b as u64;
        }
        Ok((Varint(v), len))
    }
}

impl From<u8> for Varint {
    fn from(v: u8) -> Self {
        Varint(v as u64)
    }
}
impl From<u16> for Varint {
    fn from(v: u16) -> Self {
        Varint(v as u64)
    }
}
impl From<u32> for Varint {
    fn from(v: u32) -> Self {
        Varint(v as u64)
    }
}

impl TryFrom<u64> for Varint {
    type Error = VarintError;
    fn try_from(v: u64) -> std::result::Result<Self, VarintError> {
        Self::new(v)
    }
}

impl std::fmt::Display for Varint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all_lengths() {
        let cases: &[(u64, &[u8])] = &[
            (0, &[0x00]),
            (63, &[0x3f]),
            (64, &[0x40, 0x40]),
            (16383, &[0x7f, 0xff]),
            (16384, &[0x80, 0x00, 0x40, 0x00]),
            (1073741823, &[0xbf, 0xff, 0xff, 0xff]),
            (1073741824, &[0xc0, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00]),
            (MAX_VARINT, &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]),
        ];
        for &(v, wire) in cases {
            let var = Varint(v);
            assert_eq!(var.encoded_len(), wire.len(), "len mismatch for {v}");
            let mut out = [0u8; 8];
            let n = var.encode(&mut out).unwrap();
            assert_eq!(&out[..n], wire, "encode mismatch for {v}");
            let (decoded, used) = Varint::decode(wire).unwrap();
            assert_eq!(decoded.0, v);
            assert_eq!(used, wire.len());
        }
    }

    #[test]
    fn rejects_oversized() {
        assert!(Varint::new(MAX_VARINT).is_ok());
        assert!(Varint::new(MAX_VARINT + 1).is_err());
    }
}
