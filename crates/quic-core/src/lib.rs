//! `quic-core` — a from-scratch QUIC transport (RFC 9000) built on
//! `tokio` + `std::net::UdpSocket`, with `rustls` for TLS 1.3 and `ring` for
//! packet protection.
//!
//! # Module map
//!
//! | Module              | RFC reference        | Responsibility                          |
//! |---------------------|----------------------|------------------------------------------|
//! | [`varint`]          | §16                  | Variable-length integers                 |
//! | [`cid`]             | §5                   | Connection identifiers                   |
//! | [`packet`]          | §17                  | Long/short headers, packet-number codec  |
//! | [`frame`]           | §19                  | All QUIC frame types                     |
//! | [`transport_params`]| §18                 | Negotiated connection limits             |
//! | [`crypto`]          | RFC 9001 §5          | Initial keys, AEAD, header protection    |
//! | [`tls`]             | RFC 9001 §4          | rustls QUIC handshake driver             |
//! | [`stream`]          | §3, §4               | Stream state machine, flow control       |
//! | [`loss_detection`]  | RFC 9002 §6          | Loss detection, ACK processing, PTO      |
//! | [`congestion`]      | RFC 9002 §7          | NewReno + trait for ML controllers       |
//! | [`connection`]      | §10                  | Per-connection state machine             |
//! | [`endpoint`]        | §9                   | UDP socket, routing, connection table    |
//!
//! The crate is `#![no_std]`-ready in structure but currently requires `std`
//! for sockets and timers; disable the default `std` feature to drop the
//! socket layer (useful for embedded / kernel contexts).

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
#![warn(missing_docs)]

extern crate alloc;

pub mod cid;
pub mod congestion;
pub mod connection;
pub mod crypto;
pub mod endpoint;
pub mod error;
pub mod frame;
pub mod loss_detection;
pub mod packet;
pub mod stream;
pub mod tls;
pub mod transport_params;
pub mod varint;

pub use cid::{CidPair, ConnectionId};
pub use congestion::{CongestionController, CongestionStats, NewReno, NewRenoConfig};
pub use connection::Connection;
pub use endpoint::{Endpoint, EndpointConfig};
pub use error::{Error, Result, TransportErrorCode};
pub use frame::{AckFrame, AckEcnCounts, Frame, StreamFrame};
pub use loss_detection::{LossDetector, PnSpace, SentPacket};
pub use packet::{PacketForm, PacketHeader, PnHints, VERSION_1, VERSION_2};
pub use stream::{RecvHalf, SendHalf, Stream, StreamMap, StreamRole};
pub use tls::TlsSession;
pub use transport_params::TransportParameters;
pub use varint::Varint;
