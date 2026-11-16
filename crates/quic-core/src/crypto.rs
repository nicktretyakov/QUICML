//! QUIC packet protection (RFC 9001 §5).
//!
//! This module derives the *initial* traffic keys from the destination
//! connection ID of an INITIAL packet, and provides AEAD
//! encrypt/decrypt plus header-protection mask generation for any
//! QUIC packet-number space.
//!
//! The handshake traffic keys for the 0-RTT, Handshake and 1-RTT spaces are
//! produced by rustls during the TLS 1.3 handshake (see [`crate::tls`]); the
//! AEAD/header-protection helpers here are reused for every space.
//!
//! Implementation notes:
//! - HKDF-Extract / HKDF-Expand are implemented on top of `ring::hmac` so we
//!   are not coupled to the fiddly `ring::hkdf` high-level API and can support
//!   both SHA-256 (Initial / 1-RTT with SHA-256 suites) and SHA-384 suites.
//! - AEAD uses `ring::aead` (AES-128/256-GCM).
//! - Header-protection mask uses a single AES block encryption (`aes` crate),
//!   since neither ring nor rustls expose raw ECB of one block.

use crate::cid::ConnectionId;
use crate::error::{Error, Result};
use ring::{aead, hmac};

/// QUIC v1 initial salt (RFC 9001 §5.2).
pub const QUIC_V1_INITIAL_SALT: [u8; 20] = [
    0x38, 0x76, 0x2c, 0xf7, 0xf5, 0x59, 0x34, 0xb3, 0x4d, 0x17, 0x9a, 0xe6, 0xa4, 0xc8, 0x0c,
    0xad, 0xcc, 0xbb, 0x7f, 0x0a,
];

/// QUIC v2 initial salt (RFC 9369 §1.4).
pub const QUIC_V2_INITIAL_SALT: [u8; 20] = [
    0x0d, 0xed, 0xe3, 0xde, 0xf7, 0x00, 0xa6, 0xdb, 0xa1, 0x69, 0x47, 0x5c, 0x51, 0xb1, 0x03,
    0x07, 0x8e, 0xa0, 0xf6, 0xe7,
];

/// Hash algorithm driving HKDF for a given key space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HkdfAlgo {
    Sha256,
    Sha384,
}

impl HkdfAlgo {
    fn hmac_algo(self) -> &'static hmac::Algorithm {
        match self {
            HkdfAlgo::Sha256 => &hmac::HMAC_SHA256,
            HkdfAlgo::Sha384 => &hmac::HMAC_SHA384,
        }
    }
    fn digest_len(self) -> usize {
        match self {
            HkdfAlgo::Sha256 => 32,
            HkdfAlgo::Sha384 => 48,
        }
    }
}

/// AEAD algorithm paired with a hash for HKDF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AeadAlgo {
    Aes128Gcm, // key 16, iv 12, hp 16
    Aes256Gcm, // key 32, iv 12, hp 32
}

impl AeadAlgo {
    pub fn key_len(self) -> usize {
        match self {
            AeadAlgo::Aes128Gcm => 16,
            AeadAlgo::Aes256Gcm => 32,
        }
    }
    pub const fn iv_len(self) -> usize {
        12
    }
    pub fn hp_len(self) -> usize {
        match self {
            AeadAlgo::Aes128Gcm => 16,
            AeadAlgo::Aes256Gcm => 32,
        }
    }
    fn hkdf_algo(self) -> HkdfAlgo {
        match self {
            AeadAlgo::Aes128Gcm => HkdfAlgo::Sha256,
            AeadAlgo::Aes256Gcm => HkdfAlgo::Sha384,
        }
    }
    fn ring_aead(self) -> &'static aead::Algorithm {
        match self {
            AeadAlgo::Aes128Gcm => &aead::AES_128_GCM,
            AeadAlgo::Aes256Gcm => &aead::AES_256_GCM,
        }
    }
}

/// A full set of packet-protection keys for one direction of one space.
#[derive(Debug, Clone)]
pub struct PacketKey {
    pub aead: AeadAlgo,
    pub key: Vec<u8>,
    pub iv: [u8; 12],
    pub hp: Vec<u8>,
    /// Cached ring key for AEAD.
    ring_key: aead::LessSafeKey,
}

impl PacketKey {
    /// Derive the Initial packet-protection keys for one side from the
    /// connection ID and the chosen salt.
    ///
    /// `is_server` selects whether to derive the server or client initial key.
    pub fn derive_initial(dcid: &ConnectionId, is_server: bool, v2: bool) -> Result<Self> {
        let salt = if v2 {
            &QUIC_V2_INITIAL_SALT[..]
        } else {
            &QUIC_V1_INITIAL_SALT[..]
        };
        let algo = HkdfAlgo::Sha256;
        let initial_secret = hkdf_extract(algo, salt, dcid.as_bytes());
        let label = if is_server { "server in" } else { "client in" };
        let secret = hkdf_expand_label(algo, &initial_secret, label, &[], algo.digest_len());
        Self::from_secret(&secret, AeadAlgo::Aes128Gcm)
    }

    /// Derive keys for any traffic secret produced by the TLS handshake
    /// (0-RTT, Handshake, 1-RTT).
    pub fn from_secret(secret: &[u8], aead: AeadAlgo) -> Result<Self> {
        let algo = aead.hkdf_algo();
        // QUIC treats the TLS secret directly as the HKDF "secret" and uses
        // HKDF-Expand-Label on it (no re-extract); see RFC 9001 §5.1.
        let key = hkdf_expand_label(algo, secret, "quic key", &[], aead.key_len());
        let iv = hkdf_expand_label_bytes(algo, secret, "quic iv", &[], aead.iv_len());
        let hp = hkdf_expand_label_bytes(algo, secret, "quic hp", &[], aead.hp_len());
        let mut iv_arr = [0u8; 12];
        iv_arr.copy_from_slice(&iv);
        let unbound = aead::UnboundKey::new(aead.ring_aead(), &key)
            .map_err(|_| Error::Crypto("failed to construct aead key".into()))?;
        let ring_key = aead::LessSafeKey::new(unbound);
        Ok(Self {
            aead,
            key,
            iv: iv_arr,
            hp,
            ring_key,
        })
    }

    /// Build the 12-byte nonce for `pn`: IV XOR pn (big-endian, last 8 bytes).
    pub fn nonce(&self, pn: u64) -> [u8; 12] {
        let mut n = self.iv;
        let be = pn.to_be_bytes();
        for i in 0..8 {
            n[4 + i] ^= be[i];
        }
        n
    }

    /// Encrypt `plaintext` in place into `out`, which must have room for
    /// `plaintext.len() + aead.tag_len()` bytes. Returns the number of bytes
    /// written. `header` is the AAD.
    pub fn encrypt(
        &self,
        header: &[u8],
        plaintext: &[u8],
        pn: u64,
        out: &mut [u8],
    ) -> Result<usize> {
        let needed = plaintext.len() + self.aead.ring_aead().tag_len();
        if out.len() < needed {
            return Err(Error::Crypto("encrypt output buffer too small".into()));
        }
        out[..plaintext.len()].copy_from_slice(plaintext);
        let nonce = aead::Nonce::try_assume_unique_for_key(&self.nonce(pn))
            .map_err(|_| Error::Crypto("invalid nonce length".into()))?;
        let tag = self
            .ring_key
            .seal_in_place_separate_tag(nonce, aead::Aad::from(header), &mut out[..plaintext.len()])
            .map_err(|e| Error::Crypto(format!("aead seal failed: {e}")))?;
        out[plaintext.len()..plaintext.len() + tag.as_ref().len()].copy_from_slice(tag.as_ref());
        Ok(needed)
    }

    /// Decrypt a ciphertext+tag in place. `in_out` is the full packet buffer
    /// (header || ciphertext || tag); the header is used as AAD. Returns a
    /// slice into `in_out` containing the recovered plaintext.
    pub fn decrypt<'a>(
        &self,
        header: &[u8],
        in_out: &'a mut [u8],
        pn: u64,
    ) -> Result<&'a mut [u8]> {
        let nonce = aead::Nonce::try_assume_unique_for_key(&self.nonce(pn))
            .map_err(|_| Error::Crypto("invalid nonce length".into()))?;
        let pt = self
            .ring_key
            .open_in_place(nonce, aead::Aad::from(header), in_out)
            .map_err(|_| Error::Crypto("aead open failed (authentication tag mismatch)".into()))?;
        Ok(pt)
    }

    /// Generate the 5-byte header-protection mask from a 16-byte sample of
    /// ciphertext (RFC 9001 §5.4). Returns the mask.
    pub fn header_protection_mask(&self, sample: &[u8; 16]) -> Result<[u8; 5]> {
        match self.aead {
            AeadAlgo::Aes128Gcm => {
                use aes::cipher::{BlockEncrypt, KeyInit};
                use aes::Aes128;
                use aes::cipher::generic_array::GenericArray;
                let cipher = Aes128::new(GenericArray::from_slice(&self.hp));
                let mut block = GenericArray::clone_from_slice(sample);
                cipher.encrypt_block(&mut block);
                let mut mask = [0u8; 5];
                mask.copy_from_slice(&block[..5]);
                Ok(mask)
            }
            AeadAlgo::Aes256Gcm => {
                use aes::cipher::{BlockEncrypt, KeyInit};
                use aes::Aes256;
                use aes::cipher::generic_array::GenericArray;
                let cipher = Aes256::new(GenericArray::from_slice(&self.hp));
                let mut block = GenericArray::clone_from_slice(sample);
                cipher.encrypt_block(&mut block);
                let mut mask = [0u8; 5];
                mask.copy_from_slice(&block[..5]);
                Ok(mask)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HKDF primitives implemented on ring::hmac (stable, no high-level API pain).
// ---------------------------------------------------------------------------

fn hkdf_extract(algo: HkdfAlgo, salt: &[u8], ikm: &[u8]) -> Vec<u8> {
    let key = hmac::Key::new(*algo.hmac_algo(), salt);
    hmac::sign(&key, ikm).as_ref().to_vec()
}

fn hkdf_expand(algo: HkdfAlgo, prk: &[u8], info: &[u8], len: usize) -> Vec<u8> {
    let key = hmac::Key::new(*algo.hmac_algo(), prk);
    let mut out = Vec::with_capacity(len);
    let mut t_prev: Vec<u8> = Vec::new();
    let mut counter: u8 = 1;
    while out.len() < len {
        let mut msg = t_prev.clone();
        msg.extend_from_slice(info);
        msg.push(counter);
        let mac = hmac::sign(&key, &msg);
        t_prev.clear();
        t_prev.extend_from_slice(mac.as_ref());
        out.extend_from_slice(&t_prev);
        counter = counter.wrapping_add(1);
        if counter == 0 {
            break; // cannot exceed 255 blocks
        }
    }
    out.truncate(len);
    out
}

/// HKDF-Expand-Label (RFC 8446 §7.1): label is prefixed with "tls13 ".
fn hkdf_expand_label(
    algo: HkdfAlgo,
    secret: &[u8],
    label: &str,
    context: &[u8],
    length: usize,
) -> Vec<u8> {
    let full = format!("tls13 {label}");
    let mut info = Vec::with_capacity(2 + 1 + full.len() + 1 + context.len());
    info.extend_from_slice(&(length as u16).to_be_bytes());
    info.push(full.len() as u8);
    info.extend_from_slice(full.as_bytes());
    info.push(context.len() as u8);
    info.extend_from_slice(context);
    hkdf_expand(algo, secret, &info, length)
}

/// Variant returning a `Vec<u8>` of the requested length.
fn hkdf_expand_label_bytes(
    algo: HkdfAlgo,
    secret: &[u8],
    label: &str,
    context: &[u8],
    length: usize,
) -> Vec<u8> {
    hkdf_expand_label(algo, secret, label, context, length)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test vector from RFC 9001 Appendix A (recomputed independently with
    /// Python's `hmac`/`hashlib` to confirm the exact bytes).
    #[test]
    fn rfc9001_initial_keys_client() {
        // dcid = 0x8394c8f03e515708 (Appendix A example).
        let dcid =
            ConnectionId::from_slice(&[0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08]).unwrap();
        let salt = &QUIC_V1_INITIAL_SALT[..];
        let initial_secret = hkdf_extract(HkdfAlgo::Sha256, salt, dcid.as_bytes());
        let expected_initial: [u8; 32] = [
            0x7d, 0xb5, 0xdf, 0x06, 0xe7, 0xa6, 0x9e, 0x43, 0x24, 0x96, 0xad, 0xed, 0xb0, 0x08,
            0x51, 0x92, 0x35, 0x95, 0x22, 0x15, 0x96, 0xae, 0x2a, 0xe9, 0xfb, 0x81, 0x15, 0xc1,
            0xe9, 0xed, 0x0a, 0x44,
        ];
        assert_eq!(&initial_secret[..], &expected_initial[..]);

        let client_secret =
            hkdf_expand_label(HkdfAlgo::Sha256, &initial_secret, "client in", &[], 32);
        let expected_client_secret: [u8; 32] = [
            0xc0, 0x0c, 0xf1, 0x51, 0xca, 0x5b, 0xe0, 0x75, 0xed, 0x0e, 0xbf, 0xb5, 0xc8, 0x03,
            0x23, 0xc4, 0x2d, 0x6b, 0x7d, 0xb6, 0x78, 0x81, 0x28, 0x9a, 0xf4, 0x00, 0x8f, 0x1f,
            0x6c, 0x35, 0x7a, 0xea,
        ];
        assert_eq!(&client_secret[..], &expected_client_secret[..]);

        // And the derived AEAD key for the client Initial space.
        let key = PacketKey::derive_initial(&dcid, false, false).unwrap();
        let expected_key: [u8; 16] = [
            0x1f, 0x36, 0x96, 0x13, 0xdd, 0x76, 0xd5, 0x46, 0x77, 0x30, 0xef, 0xcb, 0xe3, 0xb1,
            0xa2, 0x2d,
        ];
        assert_eq!(&key.key[..], &expected_key[..]);
        let expected_iv: [u8; 12] = [
            0xfa, 0x04, 0x4b, 0x2f, 0x42, 0xa3, 0xfd, 0x3b, 0x46, 0xfb, 0x25, 0x5c,
        ];
        assert_eq!(&key.iv[..], &expected_iv[..]);
    }

    #[test]
    fn aead_roundtrip() {
        let dcid = ConnectionId::from_slice(&[0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08])
            .unwrap();
        let key = PacketKey::derive_initial(&dcid, false, false).unwrap();
        let header = b"header";
        let pt = b"hello quic payload";
        let mut ct = vec![0u8; pt.len() + 16];
        let n = key.encrypt(header, pt, 42, &mut ct).unwrap();
        assert_eq!(n, ct.len());
        // decrypt expects in_out = ciphertext || tag (the header is AAD, not
        // part of in_out), so slice off the header.
        let mut buf = vec![0u8; ct.len()];
        buf.copy_from_slice(&ct);
        let pt2 = key.decrypt(header, &mut buf, 42).unwrap();
        assert_eq!(pt2, pt);
    }
}
