//! QUIC client example — connects to a server via the `Endpoint`, using the
//! system root trust store (or a `--ca` PEM file) for rustls.
//!
//! Run with:
//! ```bash
//! ./scripts/cargo.sh run --release --example client -p quic-core -- \
//!     --addr 127.0.0.1:4433 --name localhost [--ca /path/to/ca.pem]
//! ```

use quic_core::endpoint::{Endpoint, EndpointConfig};
use quic_core::tls;
use quic_core::transport_params::TransportParameters;
use std::io::Read;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

struct Args {
    addr: SocketAddr,
    name: String,
    ca: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        addr: "127.0.0.1:4433".parse().unwrap(),
        name: "localhost".into(),
        ca: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--addr" => {
                args.addr = it
                    .next()
                    .ok_or("--addr needs a value")?
                    .parse()
                    .map_err(|e: std::net::AddrParseError| e.to_string())?
            }
            "--name" => args.name = it.next().ok_or("--name needs a value")?,
            "--ca" => args.ca = Some(PathBuf::from(it.next().ok_or("--ca needs a value")?)),
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(args)
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

    // Build the root trust store: webpki-roots by default, or a supplied CA.
    let mut roots = rustls::RootCertStore::empty();
    match &args.ca {
        Some(ca) => {
            let pem = read_file(ca)?;
            for cert in rustls_pemfile::certs(&mut &pem[..]) {
                roots.add(cert?)?;
            }
        }
        None => {
            roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        }
    }

    let client_cfg = tls::client_config(Arc::new(roots), Some(&["hq-interop"]))?;

    let mut local_params = TransportParameters::default();
    local_params.initial_max_data = 1 << 20;
    local_params.initial_max_streams_bidi = 4;
    local_params.initial_max_stream_data_bidi_remote = 256 * 1024;
    local_params.max_idle_timeout_ms = 30_000;

    // Bind a client endpoint (any free port).
    let cfg = EndpointConfig {
        addr: "127.0.0.1:0".parse().unwrap(),
        is_server: false,
        server_cfg: None,
        client_cfg: Some(client_cfg.clone()),
        local_params: local_params.clone(),
        max_udp_payload: 65527,
    };
    let ep = Arc::new(Endpoint::bind(cfg).await?);
    tracing::info!("client endpoint bound at {}", ep.local_addr());

    let handle = ep.connect(args.addr, &args.name, client_cfg, local_params)?;
    tracing::info!(
        "opened connection {} to {} (dcid={}, scid={})",
        handle.id,
        handle.peer,
        handle.conn.lock().cids.remote,
        handle.conn.lock().cids.local,
    );

    // The endpoint runs its own recv loop; for this example we just hold the
    // connection open briefly so the user can observe the setup, then exit.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    ep.shutdown();
    Ok(())
}
