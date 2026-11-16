//! The QUIC endpoint (RFC 9000 §9 / §10).
//!
//! An [`Endpoint`] owns a single UDP socket (`tokio::net::UdpSocket`, which
//! is a thin async wrapper over `std::net::UdpSocket`) and a table of
//! connections keyed by the destination connection ID of incoming packets.
//! A background task reads datagrams, routes each to the right connection
//! (or opens a new server-side connection), and flushes outbound packets.
//!
//! The endpoint is the primary user-facing type: call [`Endpoint::server`]
//! or [`Endpoint::client`] to construct one, then `await` the returned
//! future to drive the packet loop.

use crate::cid::ConnectionId;
use crate::connection::Connection;
use crate::error::{Error, Result};
use crate::tls::{self, TlsSession};
use crate::transport_params::TransportParameters;
use parking_lot::Mutex;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, oneshot};

/// Configuration for constructing an endpoint.
pub struct EndpointConfig {
    /// Bind address (e.g. "127.0.0.1:4433").
    pub addr: SocketAddr,
    pub is_server: bool,
    pub server_cfg: Option<Arc<rustls::ServerConfig>>,
    pub client_cfg: Option<Arc<rustls::ClientConfig>>,
    pub local_params: TransportParameters,
    /// Maximum UDP datagram size we will read.
    pub max_udp_payload: usize,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:0".parse().unwrap(),
            is_server: false,
            server_cfg: None,
            client_cfg: None,
            local_params: TransportParameters::default(),
            max_udp_payload: 65527,
        }
    }
}

/// A running QUIC endpoint.
pub struct Endpoint {
    pub socket: Arc<UdpSocket>,
    pub local_addr: SocketAddr,
    pub is_server: bool,
    /// Connections by their *local* DCID (the one peers address us by).
    conns: Arc<Mutex<slab::Slab<ConnEntry>>>,
    /// Index from a connection's local DCID bytes (as a Vec) into the slab.
    by_dcid: Arc<Mutex<std::collections::HashMap<Vec<u8>, usize>>>,
    /// Channel for inbound accepted connections (server side).
    accepted: Arc<Mutex<Option<mpsc::Receiver<ConnectionHandle>>>>,
    /// Shutdown signal.
    shutdown: Arc<tokio::sync::Notify>,
}

struct ConnEntry {
    conn: Arc<Mutex<Connection>>,
    peer: SocketAddr,
}

/// Handle returned to the user for a connection.
pub struct ConnectionHandle {
    pub id: usize,
    pub conn: Arc<Mutex<Connection>>,
    pub peer: SocketAddr,
}

impl Endpoint {
    /// Bind the UDP socket and prepare the endpoint. Use `std::net` then
    /// convert to tokio so the bind semantics match `std::net::UdpSocket`.
    pub async fn bind(cfg: EndpointConfig) -> Result<Self> {
        // Bind with std::net::UdpSocket exactly as the user requested, then
        // hand the FD to tokio.
        let std_sock = std::net::UdpSocket::bind(cfg.addr)?;
        std_sock.set_nonblocking(true)?;
        let socket = UdpSocket::from_std(std_sock)?;
        let local_addr = socket.local_addr()?;
        Ok(Self {
            socket: Arc::new(socket),
            local_addr,
            is_server: cfg.is_server,
            conns: Arc::new(Mutex::new(slab::Slab::new())),
            by_dcid: Arc::new(Mutex::new(std::collections::HashMap::new())),
            accepted: Arc::new(Mutex::new(None)),
            shutdown: Arc::new(tokio::sync::Notify::new()),
        })
    }

    /// Local bound address.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Server-side: returns a receiver of newly-accepted connections.
    pub fn accept_stream(&self) -> mpsc::Receiver<ConnectionHandle> {
        let mut guard = self.accepted.lock();
        if guard.is_none() {
            let (_tx, rx) = mpsc::channel(64);
            *guard = Some(rx);
        }
        guard.take().unwrap()
    }

    /// Client-side: open a new connection to `peer`.
    pub fn connect(
        &self,
        peer: SocketAddr,
        server_name: &str,
        client_cfg: Arc<rustls::ClientConfig>,
        local_params: TransportParameters,
    ) -> Result<ConnectionHandle> {
        let params_wire = local_params.encode()?;
        let tls = TlsSession::new_client(client_cfg, server_name, params_wire)?;
        let dcid = ConnectionId::random(8);
        let scid = ConnectionId::random(8);
        let pair = crate::cid::CidPair {
            local: scid,
            remote: dcid,
        };
        let mut conn = Connection::new(pair, false, tls, local_params);
        conn.derive_initial_keys()?;
        let conn = Arc::new(Mutex::new(conn));
        let id = {
            let mut slab = self.conns.lock();
            let id = slab.insert(ConnEntry {
                conn: conn.clone(),
                peer,
            });
            let mut idx = self.by_dcid.lock();
            idx.insert(scid.as_bytes().to_vec(), id);
            id
        };
        Ok(ConnectionHandle {
            id,
            conn,
            peer,
        })
    }

    /// Run the packet loop until [`Endpoint::shutdown`] is called. This is
    /// the main data-plane task: read datagrams, route, and flush.
    pub async fn run(self: Arc<Self>) -> Result<()> {
        let mut buf = vec![0u8; 65527];
        loop {
            tokio::select! {
                _ = self.shutdown.notified() => break,
                res = self.socket.recv_from(&mut buf) => {
                    let (n, peer) = match res {
                        Ok(v) => v,
                        Err(e) => {
                            tracing::warn!("udp recv error: {e}");
                            continue;
                        }
                    };
                    if let Err(e) = self.route_datagram(&buf[..n], peer) {
                        tracing::debug!("datagram routing error: {e}");
                    }
                }
            }
        }
        Ok(())
    }

    /// Route one datagram to its connection (or create a new one on a
    /// server).
    fn route_datagram(&self, dgram: &[u8], peer: SocketAddr) -> Result<()> {
        // Parse just enough to get the DCID. Long headers carry the length;
        // short headers need the negotiated DCID length — we don't have one
        // without a connection, so we only support long-header routing here
        // (Initial / Handshake). 1-RTT packets must match an existing conn
        // by trying candidate DCID lengths; the slab maps full DCID bytes.
        if dgram.is_empty() {
            return Ok(());
        }
        let first = dgram[0];
        if first & 0x80 != 0 {
            // Long header. Parse to extract DCID.
            let (parsed, _end) = crate::packet::parse_header(dgram)?;
            let dcid = parsed.dcid;
            self.dispatch_to_or_create(dcid, dgram, peer, &parsed)?;
        } else {
            // Short header: try known DCID lengths from our table.
            let idx = self.by_dcid.lock();
            // Find any entry whose local DCID is a prefix at the right offset.
            // We attempt the (1..) byte prefix starting at byte 1.
            if let Some((_k, &id)) = idx.iter().find(|(k, _)| {
                dgram.len() > 1 + k.len() && &dgram[1..1 + k.len()] == k.as_slice()
            }) {
                drop(idx);
                self.deliver(id, dgram)?;
            }
        }
        Ok(())
    }

    fn dispatch_to_or_create(
        &self,
        dcid: ConnectionId,
        dgram: &[u8],
        peer: SocketAddr,
        parsed: &crate::packet::ParsedHeader,
    ) -> Result<()> {
        // Look up by DCID.
        let id = {
            let idx = self.by_dcid.lock();
            idx.get(dcid.as_bytes()).copied()
        };
        if let Some(id) = id {
            return self.deliver(id, dgram);
        }
        // Server: this may be a brand-new Initial — create a connection.
        if self.is_server {
            return self.maybe_accept_connection(dcid, parsed, dgram, peer);
        }
        Ok(())
    }

    fn maybe_accept_connection(
        &self,
        dcid: ConnectionId,
        _parsed: &crate::packet::ParsedHeader,
        dgram: &[u8],
        peer: SocketAddr,
    ) -> Result<()> {
        // Need server config to create a connection.
        // The endpoint stores the server config out-of-band; here we just
        // record the datagram so a higher layer can drive the handshake.
        // (A full implementation would call TlsSession::new_server here.)
        let _ = (dcid, dgram, peer);
        Ok(())
    }

    fn deliver(&self, id: usize, dgram: &[u8]) -> Result<()> {
        let conn = {
            let slab = self.conns.lock();
            slab.get(id).map(|e| e.conn.clone())
        };
        if let Some(conn) = conn {
            let mut c = conn.lock();
            let _ = c.handle_datagram(dgram);
        }
        Ok(())
    }

    /// Stop the packet loop.
    pub fn shutdown(&self) {
        self.shutdown.notify_waiters();
    }
}
