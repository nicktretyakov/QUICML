//! QUIC server example — binds a UDP socket via the `Endpoint`, loads a PEM
//! certificate/key, builds a rustls server config and runs the packet loop.
//!
//! This demonstrates the `tokio` + `std::net::UdpSocket` + `rustls` stack the
//! crate is built on. The transport primitives (packet, frame, crypto,
//! transport params) are all live; full handshake-key rotation to 1-RTT is
//! the documented integration step (see ARCHITECTURE.md).
//!
//! Run with:
//! ```bash
//! ./scripts/cargo.sh run --release --example server -p quic-core -- \
//!     --cert /path/to/cert.pem --key /path/to/key.pem --addr 127.0.0.1:4433
//! ```
//!
//! Generate a quick self-signed cert for local testing:
//! ```bash
//! openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem \
//!     -days 365 -nodes -subj "/CN=localhost"
//! ```

use quic_core::endpoint::{Endpoint, EndpointConfig};
use quic_core::tls;
use quic_core::transport_params::TransportParameters;
use std::io::Read;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        addr: "127.0.0.1:4433".parse().unwrap(),
        cert: PathBuf::new(),
        key: PathBuf::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--addr" => args.addr = it.next().ok_or("--addr needs a value")?.parse().map_err(|e: std::net::AddrParseError| e.to_string())?,
            "--cert" => args.cert = PathBuf::from(it.next().ok_or("--cert needs a value")?),
            "--key" => args.key = PathBuf::from(it.next().ok_or("--key needs a value")?),
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    if args.cert.as_os_str().is_empty() || args.key.as_os_str().is_empty() {
        return Err("--cert and --key are required".into());
    }
    Ok(args)
}

struct Args {
    addr: SocketAddr,
    cert: PathBuf,
    key: PathBuf,
}

fn read_file(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    let mut v = Vec::new();
    f.read_to_end(&mut v)?;
    Ok(v)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("quic_core=info")),
        )
        .init();
    let args = parse_args()?;

    let cert_pem = read_file(&args.cert)?;
    let key_pem = read_file(&args.key)?;
    let cert_chain: Vec<rustls::pki_types::CertificateDer<'static>> =
        rustls_pemfile::certs(&mut &cert_pem[..])
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
            .into_iter()
            .map(rustls::pki_types::CertificateDer::from)
            .collect();
    let key_der = rustls_pemfile::private_key(&mut &key_pem[..])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no private key"))?;

    let server_cfg = tls::server_config(cert_chain, key_der)?;

    let mut local_params = TransportParameters::default();
    local_params.initial_max_data = 1 << 20;
    local_params.initial_max_streams_bidi = 128;
    local_params.initial_max_stream_data_bidi_local = 256 * 1024;
    local_params.initial_max_stream_data_bidi_remote = 256 * 1024;
    local_params.max_idle_timeout_ms = 30_000;

    let cfg = EndpointConfig {
        addr: args.addr,
        is_server: true,
        server_cfg: Some(server_cfg),
        client_cfg: None,
        local_params,
        max_udp_payload: 65527,
    };
    let ep = Arc::new(Endpoint::bind(cfg).await?);
    tracing::info!("QUIC server listening on {}", ep.local_addr());

    let ep_run = ep.clone();
    tokio::select! {
        res = ep_run.run() => {
            if let Err(e) = res {
                tracing::error!("endpoint error: {e}");
            }
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("shutdown signal received");
            ep.shutdown();
        }
    }
    Ok(())
}
