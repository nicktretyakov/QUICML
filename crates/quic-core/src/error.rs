//! Crate-wide error types.
//!
//! Every fallible public API of `quic-core` returns [`Error`] (or a typed
//! variant via [`Result<T>`]). Errors are cheaply constructible and carry an
//! optional backtrace-free context string for diagnostics.

use thiserror::Error;

/// A specialized [`Result`] for `quic-core` operations.
pub type Result<T> = std::result::Result<T, Error>;

/// All errors produced by the QUIC transport layer.
#[derive(Debug, Error)]
pub enum Error {
    /// A packet could not be (de)serialized, or violated an invariant of
    /// RFC 9000 §17 (e.g. a varint outside the allowed range, a truncated
    /// header, an impossible long-header form).
    #[error("packet encoding error: {0}")]
    Packet(#[from] PacketError),

    /// A frame was malformed or referenced an invalid stream/offset
    /// (RFC 9000 §19).
    #[error("frame error: {0}")]
    Frame(#[from] FrameError),

    /// Transport parameter parsing failure (RFC 9000 §18).
    #[error("transport parameter error: {0}")]
    TransportParam(#[from] TransportParamError),

    /// Failure in the cryptographic layer: AEAD, header protection, HKDF,
    /// or the rustls handshake driver (RFC 9001).
    #[error("crypto error: {0}")]
    Crypto(String),

    /// A TLS handshake error surfaced from rustls.
    #[error("tls error: {0}")]
    Tls(#[from] rustls::Error),

    /// Connection-level protocol violation that maps to a QUIC transport
    /// error code (RFC 9000 §20). The second element is the human-readable
    /// reason phrase sent in the CONNECTION_CLOSE frame.
    #[error("connection error: code={code:?} reason={reason}")]
    Connection {
        code: TransportErrorCode,
        reason: String,
    },

    /// A stream operation was attempted in a state that forbids it
    /// (RFC 9000 §3 — e.g. writing to a stream in the "Data Sent" state).
    #[error("stream {stream_id} state error: {reason}")]
    StreamState { stream_id: u64, reason: String },

    /// Flow-control budget exhausted (data or stream limits, RFC 9000 §4).
    #[error("flow control blocked: {0}")]
    FlowControl(&'static str),

    /// A peer violated a flow-control limit, forcing a connection close with
    /// `FLOW_CONTROL_ERROR`.
    #[error("flow control violated: {0}")]
    FlowControlViolated(&'static str),

    /// The socket I/O layer returned an error (UDP send/recv on
    /// `std::net::UdpSocket`).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// The connection was closed (locally or by the peer) and is no longer
    /// usable. The optional string carries a reason phrase if one was
    /// signalled.
    #[error("connection closed")]
    Closed(Option<String>),

    /// An internal invariant was violated. These should never happen in
    /// well-formed input and indicate a bug in the implementation.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Errors raised while (de)serializing QUIC packet headers and varints.
#[derive(Debug, Error)]
pub enum PacketError {
    #[error("buffer too short: need {need} bytes, have {have}")]
    Truncated { need: usize, have: usize },

    #[error("varint out of range (max 2^62-1)")]
    VarintOutOfRange,

    #[error("invalid packet header form bit")]
    InvalidHeaderForm,

    #[error("invalid fixed bit: expected 1, got 0")]
    InvalidFixedBit,

    #[error("unsupported packet version: {0:#x}")]
    UnsupportedVersion(u32),

    #[error("connection id longer than 20 bytes (RFC 9000 §17.2)")]
    CidTooLong,

    #[error("invalid token in INITIAL packet")]
    InvalidToken,

    #[error("retry packet integrity tag mismatch")]
    RetryTagMismatch,
}

/// Errors raised while parsing or building QUIC frames.
#[derive(Debug, Error)]
pub enum FrameError {
    #[error("frame buffer truncated")]
    Truncated,

    #[error("unknown frame type {0:#x}")]
    UnknownType(u64),

    #[error("invalid varint in frame field: {0}")]
    Varint(#[from] crate::varint::VarintError),

    #[error("stream id {0} refers to a stream that does not exist")]
    UnknownStream(u64),

    #[error("final size field disagrees with previously signalled size")]
    FinalSizeMismatch,

    #[error("gap block references offset {0} below the previous offset")]
    GapOutOfOrder(u64),
}

/// Errors raised while parsing transport parameters.
#[derive(Debug, Error)]
pub enum TransportParamError {
    #[error("malformed transport parameter block")]
    Malformed,

    #[error("duplicate transport parameter id {0}")]
    Duplicate(u64),

    #[error("invalid value for parameter {0:?}: {1}")]
    InvalidValue(crate::transport_params::TransportParameterId, String),

    #[error("initial_max_data must be >= sum of per-stream limits when negotiated")]
    InconsistentLimits,
}

/// QUIC transport error codes (RFC 9000 §20.2).
///
/// Only the codes used by this crate are listed; unknown codes received from
/// the peer are preserved as the raw 62-bit integer via
/// [`TransportErrorCode::Other`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportErrorCode {
    NoError,
    Internal,
    ConnectionRefused,
    FlowControlError,
    StreamLimitError,
    StreamStateError,
    FinalSizeError,
    FrameEncodingError,
    TransportParameterError,
    ConnectionIdLimitError,
    ProtocolViolation,
    InvalidToken,
    ApplicationError,
    CryptoBufferExceeded,
    CryptoError(u8),
    Other(u64),
}

impl TransportErrorCode {
    /// Wire encoding as a 62-bit value (RFC 9000 §20.2).
    pub fn to_wire(self) -> u64 {
        use TransportErrorCode::*;
        match self {
            NoError => 0,
            Internal => 1,
            ConnectionRefused => 2,
            FlowControlError => 3,
            StreamLimitError => 4,
            StreamStateError => 5,
            FinalSizeError => 6,
            FrameEncodingError => 7,
            TransportParameterError => 8,
            ConnectionIdLimitError => 9,
            ProtocolViolation => 10,
            InvalidToken => 11,
            ApplicationError => 12,
            CryptoBufferExceeded => 13,
            CryptoError(c) => 0x100 + c as u64,
            Other(v) => v,
        }
    }

    /// Decode a wire-encoded 62-bit value into a typed code where possible.
    pub fn from_wire(v: u64) -> Self {
        use TransportErrorCode::*;
        match v {
            0 => NoError,
            1 => Internal,
            2 => ConnectionRefused,
            3 => FlowControlError,
            4 => StreamLimitError,
            5 => StreamStateError,
            6 => FinalSizeError,
            7 => FrameEncodingError,
            8 => TransportParameterError,
            9 => ConnectionIdLimitError,
            10 => ProtocolViolation,
            11 => InvalidToken,
            12 => ApplicationError,
            13 => CryptoBufferExceeded,
            v if (0x100..=0x1FF).contains(&v) => CryptoError(v as u8),
            other => Other(other),
        }
    }
}

impl Error {
    /// Build a connection-level error with the given code and reason.
    pub fn connection(code: TransportErrorCode, reason: impl Into<String>) -> Self {
        Error::Connection {
            code,
            reason: reason.into(),
        }
    }
}
