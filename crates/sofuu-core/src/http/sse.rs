// sofuu-core — SSE (Server-Sent Events) parser.
//
// Rust port of src/http/sse.c — a streaming event parser. The C version was
// a JS-visible class with a `feed()` method; here it's pure logic (a state
// machine over an internal buffer) with safe string handling — no manual
// realloc/memmove, no buffer overruns.
//
// Byte-safety (AUDIT-2026-09-01 P1-11): the buffer holds RAW BYTES, not a
// String. A multi-byte UTF-8 character split across two network reads would
// decode to two U+FFFD replacement chars under per-chunk lossy decoding.
// SSE framing is newline-based and `\n`/`\r` never occur inside a multi-byte
// UTF-8 sequence, so scanning the byte buffer for `\n\n` / `\r\n\r\n` is
// always safe; only COMPLETE blocks are decoded to text (a block whose
// bytes are not valid UTF-8 is replaced lossily at that point — the split-
// codepoint corruption case can no longer happen).

#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    pub event: String, // event name ("message" default)
    pub data: String,
}

#[derive(Default)]
pub struct SseParser {
    /// Raw undecoded bytes of the current (possibly still-incomplete) block.
    buffer: Vec<u8>,
    /// P3 (AUDIT-2026-09-07): set when MAX_BUFFER forced a drop of the
    /// oldest buffered bytes — the pending block lost its head and WILL
    /// parse into corrupted data. The C SSE class surfaces this as a
    /// synthetic error event instead of silently emitting the garbage.
    pub overflowed: bool,
}

impl SseParser {
    /// Hard cap on the buffered incomplete block: a peer that never sends
    /// `\n\n` would otherwise grow the Vec unbounded (OOM). 4MB is far
    /// above any legitimate SSE frame (LLM deltas are bytes–KB).
    pub const MAX_BUFFER: usize = 4 * 1024 * 1024;
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of raw bytes; returns any complete events parsed.
    /// Handles both `\n\n` and `\r\n\r\n` block separators.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(chunk);
        // Bound the buffer: a peer that never sends `\n\n` would otherwise
        // grow the Vec unbounded (OOM). The pending block lost its head and
        // can never parse correctly again — drop it WHOLE, so the parser
        // recovers on the next complete block (draining only the excess
        // would pin the unterminated garbage at the cap forever and merge
        // it into every subsequent block).
        if self.buffer.len() > Self::MAX_BUFFER {
            self.buffer.clear();
            self.buffer.shrink_to(64 * 1024);
            // P3 (AUDIT-2026-09-07): record the corruption instead of
            // dropping bytes silently — consumers can abort the stream.
            self.overflowed = true;
        }
        let mut events = Vec::new();

        loop {
            // Find the next block terminator in the RAW byte buffer (a `\n`
            // or `\r` byte can never be part of a multi-byte UTF-8 sequence,
            // so byte-level scanning cannot split a codepoint).
            let Some((end, sep_len)) = find_block_end(&self.buffer) else {
                break;
            };

            let block: Vec<u8> = self.buffer.drain(..end + sep_len).collect();
            let ev = parse_block(&block);
            if let Some(ev) = ev {
                events.push(ev);
            }
        }
        events
    }

    /// Any leftover partial data (for debugging / flush).
    pub fn pending(&self) -> &[u8] {
        &self.buffer
    }
}

/// Find the block terminator in the raw buffer. Returns the byte index where
/// the block ENDS (start of the separator) and the separator length.
/// CRLF is scanned first: a `\n\n` window could otherwise match inside a
/// `"...a\n" + "\n..."` slice of `\r\n\r\n`-framed data.
fn find_block_end(buf: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i + 4 <= buf.len() {
        if &buf[i..i + 4] == b"\r\n\r\n" {
            return Some((i, 4));
        }
        i += 1;
    }
    let mut i = 0;
    while i + 2 <= buf.len() {
        if &buf[i..i + 2] == b"\n\n" {
            return Some((i, 2));
        }
        i += 1;
    }
    None
}

/// Parse one SSE block (lines like `event: x` / `data: y`) into an event.
/// `block` is raw bytes INCLUDING the separator; decoding happens here, on
/// complete blocks only.
fn parse_block(block: &[u8]) -> Option<SseEvent> {
    // Trim the separator itself.
    let block = if block.ends_with(b"\r\n\r\n") {
        &block[..block.len() - 4]
    } else if block.ends_with(b"\n\n") {
        &block[..block.len() - 2]
    } else {
        block
    };

    let text = String::from_utf8_lossy(block);
    let mut event = String::new();
    let mut data = String::new();

    for line in text.lines() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if let Some(rest) = line.strip_prefix("data:") {
            let rest = rest.strip_prefix(' ').unwrap_or(rest);
            if !data.is_empty() {
                data.push('\n'); // multi-line data joins with \n
            }
            data.push_str(rest);
        } else if let Some(rest) = line.strip_prefix("event:") {
            event = rest.strip_prefix(' ').unwrap_or(rest).to_string();
        }
        // Comments (`:`) and other fields ignored — matches C behavior.
    }

    if data.is_empty() {
        None // no data → no event (SSE spec: data required)
    } else {
        Some(SseEvent {
            event: if event.is_empty() {
                "message".to_string()
            } else {
                event
            },
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_event() {
        let mut p = SseParser::new();
        let evs = p.feed(b"data: hello world\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].event, "message");
        assert_eq!(evs[0].data, "hello world");
    }

    #[test]
    fn streams_across_chunks() {
        let mut p = SseParser::new();
        assert!(p.feed(b"data: part one\n").is_empty()); // no terminator yet
        let evs = p.feed(b"data: part two\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "part one\npart two");
    }

    #[test]
    fn multiple_events_one_feed() {
        let mut p = SseParser::new();
        let evs = p.feed(b"data: a\n\ndata: b\n\n");
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].data, "a");
        assert_eq!(evs[1].data, "b");
    }

    #[test]
    fn handles_crlf() {
        let mut p = SseParser::new();
        let evs = p.feed(b"data: crlf\r\n\r\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "crlf");
    }

    /// P1-11 regression: a multi-byte UTF-8 char split across two feeds must
    /// survive intact (the old per-chunk lossy decode turned it into two
    /// U+FFFD replacement chars).
    #[test]
    fn split_codepoint_across_feeds_survives() {
        let mut p = SseParser::new();
        let emoji = "🚀".as_bytes(); // 4-byte UTF-8 sequence
        // First feed: "data: " + first 2 bytes of the emoji, no terminator.
        assert!(p
            .feed(&[b'd', b'a', b't', b'a', b':', b' ', emoji[0], emoji[1]])
            .is_empty());
        // Second feed: last 2 bytes + terminator.
        let evs = p.feed(&[emoji[2], emoji[3], b'\n', b'\n']);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "🚀");
    }

    /// The same hazard inside CRLF-framed streams.
    #[test]
    fn split_codepoint_crlf_stream() {
        let mut p = SseParser::new();
        let s = "héllo".as_bytes(); // h, 0xC3, 0xA9, l, l, o — é = 2 bytes
        let mut a = b"data: h".to_vec();
        a.push(s[1]); // first byte of é, no terminator yet
        assert!(p.feed(&a).is_empty());
        let mut b2 = vec![s[2]]; // second byte of é
        b2.extend_from_slice(b"llo\r\n\r\n");
        let evs = p.feed(&b2);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "héllo");
    }

    /// CRLF- and LF-framed events in one buffer must not mis-frame.
    #[test]
    fn crlf_and_lf_events_in_one_feed() {
        let mut p = SseParser::new();
        let evs = p.feed(b"data: a\r\n\r\ndata: b\n\n");
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].data, "a");
        assert_eq!(evs[1].data, "b");
    }

    #[test]
    fn named_event() {
        let mut p = SseParser::new();
        let evs = p.feed(b"event: ping\ndata: pong\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].event, "ping");
        assert_eq!(evs[0].data, "pong");
    }

    #[test]
    fn ignores_empty_data_blocks() {
        let mut p = SseParser::new();
        let evs = p.feed(b"event: ping\n\n"); // no data → ignored
        assert!(evs.is_empty());
    }

    #[test]
    fn done_sentinel() {
        let mut p = SseParser::new();
        let evs = p.feed(b"data: [DONE]\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "[DONE]");
    }

    #[test]
    fn no_panic_on_garbage() {
        let mut p = SseParser::new();
        // Garbage without a terminator is held (no events, no crash).
        let evs = p.feed(b"\x00\x01\x02random bytes without terminator");
        assert!(evs.is_empty());
        // A clean event after garbage (garbage is part of the same block,
        // so no event — but no panic either).
        let evs = p.feed(b"data: ok\n\n");
        // The garbage merged with the block's only line → no `data:` line.
        assert!(evs.is_empty());
        // Feed garbage with a terminator to flush it, then a clean event.
        let mut p2 = SseParser::new();
        p2.feed(b"garbage\n\n");
        let evs = p2.feed(b"data: ok\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "ok");
    }

    /// pending() is raw bytes now — an incomplete tail must be visible as
    /// exactly the undelivered bytes.
    #[test]
    fn pending_holds_partial_bytes() {
        let mut p = SseParser::new();
        p.feed(b"data: par");
        assert_eq!(p.pending(), b"data: par");
        p.feed(b"tial\n\n");
        assert_eq!(p.pending(), b"");
    }

    #[test]
    fn overflow_sets_flag_and_recovers() {
        // P3 (AUDIT-2026-09-07): a peer that never sends a terminator trips
        // the cap — the drop must be recorded, not silent.
        let mut p = SseParser::new();
        let big = vec![b'a'; SseParser::MAX_BUFFER + 1024];
        assert!(p.feed(&big).is_empty()); // no terminator → no events
        assert!(p.overflowed);
        // The parser stays usable: the next complete block still parses.
        let evs = p.feed(b"data: after\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "after");
    }
}
