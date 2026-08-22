// sofuu-core — SSE (Server-Sent Events) parser.
//
// Rust port of src/http/sse.c — a streaming event parser. The C version was
// a JS-visible class with a `feed()` method; here it's pure logic (a state
// machine over an internal buffer) with safe string handling — no manual
// realloc/memmove, no buffer overruns.

#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    pub event: String, // event name ("message" default)
    pub data: String,
}

#[derive(Default)]
pub struct SseParser {
    buffer: String,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of raw bytes; returns any complete events parsed.
    /// Handles both `\n\n` and `\r\n\r\n` block separators.
    pub fn feed(&mut self, chunk: &str) -> Vec<SseEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();

        loop {
            // Find the next block terminator.
            let term = find_block_end(&self.buffer);
            let Some(end) = term else { break };

            let block: String = self.buffer.drain(..end).collect();
            // Drain the separator too.
            let sep = if self.buffer.starts_with("\r\n\r\n") {
                4
            } else {
                2
            };
            self.buffer.drain(..sep.min(self.buffer.len()));

            let ev = parse_block(&block);
            if let Some(ev) = ev {
                events.push(ev);
            }
        }
        events
    }

    /// Any leftover partial data (for debugging / flush).
    pub fn pending(&self) -> &str {
        &self.buffer
    }
}

/// Find the byte index of the block terminator (`\n\n` or `\r\n\r\n`).
fn find_block_end(buf: &str) -> Option<usize> {
    buf.find("\n\n").or_else(|| buf.find("\r\n\r\n"))
}

/// Parse one SSE block (lines like `event: x` / `data: y`) into an event.
fn parse_block(block: &str) -> Option<SseEvent> {
    let mut event = String::new();
    let mut data = String::new();

    for line in block.lines() {
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
        let evs = p.feed("data: hello world\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].event, "message");
        assert_eq!(evs[0].data, "hello world");
    }

    #[test]
    fn streams_across_chunks() {
        let mut p = SseParser::new();
        assert!(p.feed("data: part one\n").is_empty()); // no terminator yet
        let evs = p.feed("data: part two\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "part one\npart two");
    }

    #[test]
    fn multiple_events_one_feed() {
        let mut p = SseParser::new();
        let evs = p.feed("data: a\n\ndata: b\n\n");
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].data, "a");
        assert_eq!(evs[1].data, "b");
    }

    #[test]
    fn handles_crlf() {
        let mut p = SseParser::new();
        let evs = p.feed("data: crlf\r\n\r\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "crlf");
    }

    #[test]
    fn named_event() {
        let mut p = SseParser::new();
        let evs = p.feed("event: ping\ndata: pong\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].event, "ping");
        assert_eq!(evs[0].data, "pong");
    }

    #[test]
    fn ignores_empty_data_blocks() {
        let mut p = SseParser::new();
        let evs = p.feed("event: ping\n\n"); // no data → ignored
        assert!(evs.is_empty());
    }

    #[test]
    fn done_sentinel() {
        let mut p = SseParser::new();
        let evs = p.feed("data: [DONE]\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "[DONE]");
    }

    #[test]
    fn no_panic_on_garbage() {
        let mut p = SseParser::new();
        // Garbage without a terminator is held (no events, no crash).
        let evs = p.feed("\x00\x01\x02random bytes without terminator");
        assert!(evs.is_empty());
        // A clean event after garbage (garbage is part of the same block,
        // so no event — but no panic either).
        let evs = p.feed("data: ok\n\n");
        // The garbage merged with the block's only line → no `data:` line.
        assert!(evs.is_empty());
        // Feed garbage with a terminator to flush it, then a clean event.
        let mut p2 = SseParser::new();
        p2.feed("garbage\n\n");
        let evs = p2.feed("data: ok\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "ok");
    }
}
