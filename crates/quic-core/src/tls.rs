//! TLS 1.3 integration via rustls (RFC 9001 §4).
//!
//! rustls's `quic` module drives the QUIC handshake: CRYPTO frame payloads
//! are fed in via [`TlsSession::read_crypto`], and TLS records to send are
//! produced via [`TlsSession::write_crypto`], which also surfaces
//! [`rustls::quic::KeyChange`] events so the connection layer can rotate
//! packet keys between the Initial, Handshake and 1-RTT spaces.
//!
//! The Initial-space keys are *not* produced by rustls; they are derived
//! directly from the connection ID in [`crate::crypto`] (RFC 9001 §5.2),
//! because the very first CRYPTO frames must be protected before the TLS
//! state machine has produced any keys.

use crate::error::{Error, Result};
use rustls::quic::{ClientConnection, KeyChange, ServerConnection, Version};
use rustls::{ClientConfig, ServerConfig};
use std::sync::Arc;

/// QUIC protocol version handed to rustls. We negotiate QUIC v1 (RFC 9000);
/// v2 (RFC 9369) differs only in the initial salt, handled in
/// [`crate::crypto`].
const QUIC_TLS_VERSION: Version = Version::V1;

/// A rustls QUIC TLS session, either client or server side.
pub enum TlsSession {
    Client(ClientConnection),
    Server(ServerConnection),
}

impl std::fmt::Debug for TlsSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Client(_) => write!(f, "TlsSession(Client)"),
            Self::Server(_) => write!(f, "TlsSession(Server)"),
        }
    }
}

impl TlsSession {
    /// Construct a client-side session with the given QUIC transport
    /// parameters (their on-wire encoding from [`crate::transport_params`]).
    pub fn new_client(
        cfg: Arc<ClientConfig>,
        server_name: &str,
        quic_transport_params: Vec<u8>,
    ) -> Result<Self> {
        let name = rustls::pki_types::ServerName::try_from(server_name)
            .map_err(|e| Error::Tls(rustls::Error::General(format!("bad server name: {e}"))))?
            .to_owned();
        let conn = ClientConnection::new(cfg, QUIC_TLS_VERSION, name, quic_transport_params)?;
        Ok(Self::Client(conn))
    }

    /// Construct a server-side session.
    pub fn new_server(cfg: Arc<ServerConfig>, quic_transport_params: Vec<u8>) -> Result<Self> {
        let conn = ServerConnection::new(cfg, QUIC_TLS_VERSION, quic_transport_params)?;
        Ok(Self::Server(conn))
    }

    /// Feed a CRYPTO frame's plaintext payload into the TLS engine. Per
    /// RFC 9001 §4.3 each epoch's CRYPTO data is delivered in a separate
    /// call; the caller splits by epoch.
    pub fn read_crypto(&mut self, plaintext: &[u8]) -> Result<()> {
        match self {
            Self::Client(c) => c.read_hs(plaintext).map_err(Error::Tls),
            Self::Server(c) => c.read_hs(plaintext).map_err(Error::Tls),
        }
    }

    /// Pull pending TLS records to send as CRYPTO frames. `out` is appended
    /// with the raw TLS record bytes. Returns `(bytes_written, key_change)`
    /// where `key_change` is `Some` when rustls signals that the next epoch's
    /// keys are available (Handshake or 1-RTT).
    pub fn write_crypto(&mut self, out: &mut Vec<u8>) -> Result<(usize, Option<KeyChange>)> {
        let mut total = 0;
        let mut change = None;
        loop {
            let before = out.len();
            let kc = match self {
                Self::Client(c) => c.write_hs(out),
                Self::Server(c) => c.write_hs(out),
            };
            let written = out.len().saturating_sub(before);
            total += written;
            if kc.is_some() {
                change = kc;
            }
            if written == 0 {
                break;
            }
        }
        Ok((total, change))
    }

    /// Whether the handshake is still in progress.
    pub fn is_handshaking(&self) -> bool {
        match self {
            Self::Client(c) => c.is_handshaking(),
            Self::Server(c) => c.is_handshaking(),
        }
    }

    /// The peer's QUIC transport parameters, decoded from the TLS extension,
    /// once the handshake has progressed far enough to receive them.
    pub fn peer_quic_transport_parameters(&self) -> Option<&[u8]> {
        match self {
            Self::Client(c) => c.quic_transport_parameters(),
            Self::Server(c) => c.quic_transport_parameters(),
        }
    }

    /// Whether a fatal TLS alert has been raised.
    pub fn alert(&self) -> Option<rustls::AlertDescription> {
        match self {
            Self::Client(c) => c.alert(),
            Self::Server(c) => c.alert(),
        }
    }

    /// Which side we are (drives key direction naming).
    pub fn side(&self) -> rustls::Side {
        match self {
            Self::Client(_) => rustls::Side::Client,
            Self::Server(_) => rustls::Side::Server,
        }
    }
}

/// Build a rustls server config suitable for QUIC, with the supplied
/// certificate chain and key.
pub fn server_config(
    cert_chain: Vec<rustls::pki_types::CertificateDer<'static>>,
    key_der: rustls::pki_types::PrivateKeyDer<'static>,
) -> Result<Arc<ServerConfig>> {
    let provider = rustls::crypto::ring::default_provider();
    let cfg = ServerConfig::builder_with_provider(Arc::new(provider))
        .with_safe_default_protocol_versions()
        .map_err(|e| Error::Tls(rustls::Error::General(format!("proto versions: {e}"))))?
        .with_no_client_auth()
        .with_single_cert(cert_chain, key_der)
        .map_err(Error::Tls)?;
    Ok(Arc::new(cfg))
}

/// Build a rustls client config that trusts the supplied root store and,
/// optionally, advertises the given ALPN protocols.
pub fn client_config(
    roots: Arc<rustls::RootCertStore>,
    alpn: Option<&[&str]>,
) -> Result<Arc<ClientConfig>> {
    let provider = rustls::crypto::ring::default_provider();
    let mut cfg = ClientConfig::builder_with_provider(Arc::new(provider))
        .with_safe_default_protocol_versions()
        .map_err(|e| Error::Tls(rustls::Error::General(format!("proto versions: {e}"))))?
        .with_root_certificates(roots)
        .with_no_client_auth();
    if let Some(protocols) = alpn {
        cfg.alpn_protocols = protocols.iter().map(|p| p.as_bytes().to_vec()).collect();
    }
    Ok(Arc::new(cfg))
}
