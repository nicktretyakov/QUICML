//! QUIC connection identifiers (RFC 9000 §5).
//!
//! A connection ID is an opaque, variable-length blob of 0..=20 bytes. The
//! empty CID is valid. CIDs are used to route datagrams to the right
//! connection on a stateless load balancer and to defeat link-layer
//! correlation, so they are generated randomly by the endpoint that will
//! *receive* traffic on them.

use crate::error::{PacketError, Result};
use rand::RngCore;
use std::fmt;

/// Maximum legal length of a connection ID (RFC 9000 §17.2).
pub const MAX_CID_LEN: usize = 20;

/// A connection ID — an opaque byte string of at most 20 bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConnectionId {
    bytes: [u8; MAX_CID_LEN],
    len: u8,
}

impl ConnectionId {
    /// Construct a CID from a byte slice. Returns an error if the slice is
    /// longer than 20 bytes.
    pub fn from_slice(b: &[u8]) -> Result<Self> {
        if b.len() > MAX_CID_LEN {
            return Err(PacketError::CidTooLong.into());
        }
        let mut bytes = [0u8; MAX_CID_LEN];
        bytes[..b.len()].copy_from_slice(b);
        Ok(Self {
            bytes,
            len: b.len() as u8,
        })
    }

    /// The empty connection ID (length 0).
    pub const fn empty() -> Self {
        Self {
            bytes: [0u8; MAX_CID_LEN],
            len: 0,
        }
    }

    /// Generate a random CID of the given length using the thread-local RNG.
    pub fn random(len: usize) -> Self {
        debug_assert!(len <= MAX_CID_LEN);
        let mut bytes = [0u8; MAX_CID_LEN];
        rand::thread_rng().fill_bytes(&mut bytes[..len]);
        Self {
            bytes,
            len: len as u8,
        }
    }

    /// The raw bytes of the CID.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    /// Length of the CID in bytes.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Whether the CID is the empty (zero-length) CID.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl fmt::Debug for ConnectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cid(")?;
        for b in self.as_bytes() {
            write!(f, "{b:02x}")?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for ConnectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.as_bytes() {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// A pair of connection IDs describing a routing decision: the CID the peer
/// should use to address us (`local`), and the CID we should use to address
/// the peer (`remote`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CidPair {
    pub local: ConnectionId,
    pub remote: ConnectionId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_bytes() {
        let b = [1, 2, 3, 4, 5, 6, 7, 8];
        let c = ConnectionId::from_slice(&b).unwrap();
        assert_eq!(c.as_bytes(), &b);
        assert_eq!(c.len(), 8);
    }

    #[test]
    fn rejects_too_long() {
        let long = [0u8; 21];
        assert!(ConnectionId::from_slice(&long).is_err());
    }

    #[test]
    fn empty_is_zero_len() {
        let e = ConnectionId::empty();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
        assert!(e.as_bytes().is_empty());
    }

    #[test]
    fn random_has_requested_len() {
        for len in [0, 1, 8, 20] {
            assert_eq!(ConnectionId::random(len).len(), len);
        }
    }
}
