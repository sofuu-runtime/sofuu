//! cargo-fuzz target for the Rust MCP JSON-RPC parser
//! (sofuu_core::mcp::jsonrpc::parse). The security-sensitive inbound parser.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    // parse() must never panic on arbitrary input; every Inbound must be
    // safely constructible (it's an enum of owned data — no unsafe).
    let _ = sofuu_core::mcp::jsonrpc::parse(&text);
    // Round-trip the builders too (request/response/error/notify).
    let _ = sofuu_core::mcp::jsonrpc::request(1, &text, None);
    let _ = sofuu_core::mcp::jsonrpc::notify(&text, None);
    let _ = sofuu_core::mcp::jsonrpc::response(1, Some(&text));
    let _ = sofuu_core::mcp::jsonrpc::error(1, -32601, &text);
});
