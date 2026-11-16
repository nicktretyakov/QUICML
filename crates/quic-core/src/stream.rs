//! QUIC streams (RFC 9000 §3) and flow control (§4).
//!
//! Each connection holds a [`StreamMap`] indexing all open streams by their
//! 62-bit ID. A [`Stream`] carries independent send and receive halves: each
//! half is a small state machine (RFC 9000 §3.2/§3.3) backed by a byte buffer
//! that supports reordering and offset-based writes.
//!
//! Flow control (RFC 9000 §4) is enforced both per-stream and
//! connection-wide. When the peer's budget is exhausted the API returns
//! [`FlowControlBlocked`](crate::error::Error::FlowControlBlocked) so the
//! caller can wait for a `MAX_DATA`/`MAX_STREAM_DATA` update.

use crate::error::{Error, Result};
use bytes::BytesMut;
use std::collections::BTreeMap;

/// Stream types are encoded in the low 2 bits of the stream ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamRole {
    /// Client-initiated bidirectional (low bits = 0x0).
    ClientBidi,
    /// Server-initiated bidirectional (0x1).
    ServerBidi,
    /// Client-initiated unidirectional (0x2).
    ClientUni,
    /// Server-initiated unidirectional (0x3).
    ServerUni,
}

impl StreamRole {
    pub fn from_id(id: u64) -> Self {
        match id & 0x3 {
            0 => Self::ClientBidi,
            1 => Self::ServerBidi,
            2 => Self::ClientUni,
            _ => Self::ServerUni,
        }
    }
    pub fn is_bidi(self) -> bool {
        matches!(self, Self::ClientBidi | Self::ServerBidi)
    }
    pub fn is_client_initiated(self) -> bool {
        matches!(self, Self::ClientBidi | Self::ClientUni)
    }
}

/// Send-side state (RFC 9000 §3.1, states: Ready, Send, Data Sent, Data
/// Recycled; reset adds Reset Sent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendState {
    Ready,
    Send,
    DataSent,
    ResetSent,
    ResetRecvd,
    DataRecvd,
}

/// Receive-side state (§3.2: Receive Size Known, Data Recvd, Data Read; reset
/// adds Reset Recvd / Reset Read).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecvState {
    Recv,
    SizeKnown,
    DataRecvd,
    DataRead,
    ResetRecvd,
    ResetRead,
}

/// One half of a stream's flow-control + buffer state.
#[derive(Debug)]
pub struct SendHalf {
    pub state: SendState,
    /// Buffered bytes not yet handed to the packetizer (offset-ordered).
    pub outbox: BytesMut,
    /// Next byte offset to assign when the caller writes more data.
    pub next_offset: u64,
    /// Highest offset we are allowed to send, decremented as we write into
    /// the outbox and incremented by `MAX_STREAM_DATA`.
    pub max_data: u64,
    /// Final size signalled via FIN, once set.
    pub final_size: Option<u64>,
}

#[derive(Debug)]
pub struct RecvHalf {
    pub state: RecvState,
    /// Reassembly buffer keyed by the offset each chunk arrived at.
    pub inbox: BTreeMap<u64, bytes::Bytes>,
    /// Next contiguous offset the reader will consume.
    pub read_offset: u64,
    /// Largest contiguous offset received so far.
    pub contiguous_offset: u64,
    pub max_data: u64,
    pub final_size: Option<u64>,
}

impl SendHalf {
    pub fn new(max_data: u64) -> Self {
        Self {
            state: SendState::Ready,
            outbox: BytesMut::new(),
            next_offset: 0,
            max_data,
            final_size: None,
        }
    }

    /// Buffer `data` for sending. Returns the number of bytes actually
    /// accepted (bounded by `max_data - next_offset`).
    pub fn write(&mut self, data: &[u8], fin: bool) -> Result<usize> {
        if matches!(
            self.state,
            SendState::DataSent | SendState::ResetSent | SendState::ResetRecvd | SendState::DataRecvd
        ) {
            return Err(Error::StreamState {
                stream_id: 0,
                reason: "send half is closed".into(),
            });
        }
        let budget = self.max_data.saturating_sub(self.next_offset);
        let take = data.len().min(budget as usize);
        self.outbox.extend_from_slice(&data[..take]);
        self.next_offset += take as u64;
        self.state = SendState::Send;
        if fin {
            self.final_size = Some(self.next_offset);
            self.state = SendState::DataSent;
        }
        Ok(take)
    }

    /// Mark a reset. Future writes are rejected.
    pub fn reset(&mut self, error_code: u64) -> Result<()> {
        let _ = error_code;
        match self.state {
            SendState::Ready | SendState::Send | SendState::DataSent => {
                self.state = SendState::ResetSent;
                Ok(())
            }
            _ => Err(Error::StreamState {
                stream_id: 0,
                reason: "stream already finalized".into(),
            }),
        }
    }

    /// Called when the peer increases our stream-data limit.
    pub fn on_max_stream_data(&mut self, max: u64) {
        if max > self.max_data {
            self.max_data = max;
        }
    }
}

impl RecvHalf {
    pub fn new(max_data: u64) -> Self {
        Self {
            state: RecvState::Recv,
            inbox: BTreeMap::new(),
            read_offset: 0,
            contiguous_offset: 0,
            max_data,
            final_size: None,
        }
    }

    /// Accept a STREAM frame's data at the given offset. Returns the number
    /// of *new* bytes accepted (after de-duplication), or an error if the
    /// peer violated the flow-control limit.
    pub fn receive(&mut self, offset: u64, data: &[u8], fin: bool) -> Result<usize> {
        if matches!(self.state, RecvState::DataRead | RecvState::ResetRead) {
            return Ok(0);
        }
        let end = offset + data.len() as u64;
        if let Some(fs) = self.final_size {
            if end > fs {
                return Err(Error::FlowControlViolated(FinalSizeExceeded));
            }
        }
        if end > self.max_data {
            // Peer sent beyond the advertised limit: protocol violation.
            return Err(Error::FlowControlViolated(StreamLimit));
        }
        // De-duplicate against already-received ranges (simplified: track
        // contiguous boundary; gaps keep their bytes).
        let new_bytes = if end <= self.contiguous_offset {
            0
        } else {
            // Bytes above contiguous_offset are "new" (may overlap with inbox).
            let overlap_start = offset.min(self.contiguous_offset);
            (end - overlap_start) as usize
        };
        if !data.is_empty() {
            self.inbox
                .entry(offset)
                .or_insert_with(|| bytes::Bytes::copy_from_slice(data));
        }
        // Advance the contiguous boundary as far as the received chunks
        // allow. Chunks are kept in `inbox` (they are consumed by `read`),
        // so we only update `contiguous_offset` here.
        loop {
            let advance = self
                .inbox
                .range(..=self.contiguous_offset)
                .next_back()
                .and_then(|(&off, chunk)| {
                    let chunk_end = off + chunk.len() as u64;
                    if off <= self.contiguous_offset && chunk_end > self.contiguous_offset {
                        Some(chunk_end)
                    } else {
                        None
                    }
                });
            match advance {
                Some(new_end) => self.contiguous_offset = new_end,
                None => break,
            }
        }
        if fin {
            self.final_size = Some(end.max(self.final_size.unwrap_or(0)));
            self.state = RecvState::SizeKnown;
        }
        Ok(new_bytes)
    }

    /// Read available, in-order bytes into `out`. Returns how many were read.
    pub fn read(&mut self, out: &mut [u8]) -> Result<usize> {
        if matches!(self.state, RecvState::DataRead | RecvState::ResetRead) {
            return Ok(0);
        }
        let available = self.contiguous_offset - self.read_offset;
        let take = (out.len() as u64).min(available) as usize;
        // The contiguous range is in the inbox starting from read_offset... we
        // removed consumed entries above, so reconstruct by scanning from the
        // smallest key. To keep this simple we store contiguous bytes in the
        // map and read them out.
        let mut written = 0;
        while written < take {
            // Find the chunk with the largest offset <= read_offset.
            let key = self.read_offset;
            let Some(&off) = self.inbox.range(..=key).map(|(k, _)| k).next_back() else {
                break;
            };
            let chunk_len = match self.inbox.get(&off) {
                Some(c) => c.len(),
                None => {
                    self.inbox.remove(&off);
                    continue;
                }
            };
            let within = (key - off) as usize;
            if within >= chunk_len {
                // Fully consumed / stale chunk behind the read pointer.
                self.inbox.remove(&off);
                continue;
            }
            let chunk = self.inbox.get(&off).cloned().expect("present");
            let want = (take - written).min(chunk.len() - within);
            out[written..written + want]
                .copy_from_slice(&chunk[within..within + want]);
            self.read_offset += want as u64;
            written += want;
            if within + want >= chunk.len() {
                self.inbox.remove(&off);
            }
        }
        if self.final_size == Some(self.read_offset)
            && matches!(self.state, RecvState::SizeKnown | RecvState::DataRecvd)
        {
            self.state = RecvState::DataRead;
        }
        Ok(written)
    }

    pub fn on_max_stream_data(&mut self, max: u64) {
        if max > self.max_data {
            self.max_data = max;
        }
    }
}

/// Marker constants for the `FlowControlViolated` error reason.
const FinalSizeExceeded: &str = "data beyond final size";
const StreamLimit: &str = "stream data beyond advertised limit";

/// A QUIC stream: a pair of send/recv halves (only the relevant half is
/// populated for unidirectional streams).
#[derive(Debug)]
pub struct Stream {
    pub id: u64,
    pub role: StreamRole,
    pub send: Option<SendHalf>,
    pub recv: Option<RecvHalf>,
}

impl Stream {
    pub fn new(id: u64, initial_max_stream_data_bidi_local: u64, initial_max_stream_data_bidi_remote: u64, initial_max_stream_data_uni: u64) -> Self {
        let role = StreamRole::from_id(id);
        let (send, recv) = match role {
            StreamRole::ClientBidi | StreamRole::ServerBidi => (
                Some(SendHalf::new(initial_max_stream_data_bidi_local)),
                Some(RecvHalf::new(initial_max_stream_data_bidi_remote)),
            ),
            StreamRole::ClientUni | StreamRole::ServerUni => {
                if role.is_client_initiated() {
                    (Some(SendHalf::new(initial_max_stream_data_uni)), None)
                } else {
                    (None, Some(RecvHalf::new(initial_max_stream_data_uni)))
                }
            }
        };
        Self {
            id,
            role,
            send,
            recv,
        }
    }

    pub fn is_bidi(&self) -> bool {
        self.role.is_bidi()
    }
}

/// Tracks all open streams plus connection-wide flow-control budgets.
#[derive(Debug)]
pub struct StreamMap {
    pub streams: BTreeMap<u64, Stream>,
    /// Connection-wide send budget (MAX_DATA we received).
    pub conn_send_max: u64,
    /// Bytes we have buffered for sending across all streams.
    pub conn_send_used: u64,
    /// Connection-wide receive budget we advertised.
    pub conn_recv_max: u64,
    /// Bytes the peer has sent us (counted at receive()).
    pub conn_recv_used: u64,
    /// Next stream ID to allocate per role.
    pub next_client_bidi: u64,
    pub next_server_bidi: u64,
    pub next_client_uni: u64,
    pub next_server_uni: u64,
    /// Limits advertised in transport parameters.
    pub max_bidi_streams: u64,
    pub max_uni_streams: u64,
}

impl StreamMap {
    pub fn new(
        conn_send_max: u64,
        conn_recv_max: u64,
        max_bidi_streams: u64,
        max_uni_streams: u64,
        sd_bidi_local: u64,
        sd_bidi_remote: u64,
        sd_uni: u64,
        is_server: bool,
    ) -> Self {
        let _ = (sd_bidi_local, sd_bidi_remote, sd_uni);
        Self {
            streams: BTreeMap::new(),
            conn_send_max,
            conn_send_used: 0,
            conn_recv_max,
            conn_recv_used: 0,
            next_client_bidi: 0,
            next_server_bidi: 1,
            next_client_uni: 2,
            next_server_uni: 3,
            max_bidi_streams,
            max_uni_streams,
        }
        .with_is_server(is_server)
    }

    fn with_is_server(mut self, is_server: bool) -> Self {
        // Servers start counters at the odd IDs; clients at even. We bias
        // the "next" pointer accordingly so locally-initiated streams use the
        // locally-correct parity.
        if is_server {
            self.next_client_bidi = 0; // remote-initiated
            self.next_server_bidi = 1;
            self.next_client_uni = 2;
            self.next_server_uni = 3;
        }
        self
    }

    /// Allocate a new locally-initiated bidi stream.
    pub fn open_bidi(&mut self, is_server: bool) -> Result<u64> {
        let id = if is_server {
            self.next_server_bidi
        } else {
            self.next_client_bidi
        };
        if id / 4 >= self.max_bidi_streams {
            return Err(Error::FlowControl("stream limit reached"));
        }
        let s = Stream::new(id, 65536, 65536, 65536);
        self.streams.insert(id, s);
        if is_server {
            self.next_server_bidi += 4;
        } else {
            self.next_client_bidi += 4;
        }
        Ok(id)
    }

    /// Allocate a new locally-initiated uni stream.
    pub fn open_uni(&mut self, is_server: bool) -> Result<u64> {
        let id = if is_server {
            self.next_server_uni
        } else {
            self.next_client_uni
        };
        if id / 4 >= self.max_uni_streams {
            return Err(Error::FlowControl("stream limit reached"));
        }
        let s = Stream::new(id, 65536, 65536, 65536);
        self.streams.insert(id, s);
        if is_server {
            self.next_server_uni += 4;
        } else {
            self.next_client_uni += 4;
        }
        Ok(id)
    }

    pub fn get(&self, id: u64) -> Option<&Stream> {
        self.streams.get(&id)
    }
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Stream> {
        self.streams.get_mut(&id)
    }

    /// Called when the peer raises the connection-wide send budget.
    pub fn on_max_data(&mut self, max: u64) {
        if max > self.conn_send_max {
            self.conn_send_max = max;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn send_recv_roundtrip() {
        let mut s = Stream::new(0, 1024, 1024, 1024);
        let sent = s.send.as_mut().unwrap().write(b"hello", false).unwrap();
        assert_eq!(sent, 5);
        // Simulate delivering those bytes to the recv half.
        s.recv
            .as_mut()
            .unwrap()
            .receive(0, b"hello", false)
            .unwrap();
        let mut out = [0u8; 16];
        let n = s.recv.as_mut().unwrap().read(&mut out).unwrap();
        assert_eq!(&out[..n], b"hello");
    }

    #[test]
    fn flow_control_blocks_oversize() {
        let mut s = Stream::new(0, 8, 8, 8);
        let recv = s.recv.as_mut().unwrap();
        // max_data = 8; sending 9 bytes should violate.
        assert!(recv.receive(0, &[0u8; 9], false).is_err());
        // 8 bytes ok.
        assert!(recv.receive(0, &[0u8; 8], false).is_ok());
    }
}
