#!/usr/bin/env python3
"""Run a real local MCP file-read integration test in a temporary directory."""
import os
import json
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading
import time

root = Path(__file__).resolve().parent.parent
binary = Path(os.environ.get("SOFUU_BIN", root / "sofuu")).resolve()
if not binary.is_file() or not os.access(binary, os.X_OK):
    print("BLOCKED: set SOFUU_BIN to a built Sofuu executable", file=sys.stderr)
    raise SystemExit(2)
print(f"Artifact: {binary}, sha256={hashlib.sha256(binary.read_bytes()).hexdigest()}", flush=True)

with tempfile.TemporaryDirectory(prefix="sofuu-real-mcp-") as directory:
    # macOS may expose the temporary directory through /var while getcwd()
    # resolves it to /private/var. Use the same canonical path as the runtime
    # so the built-in read_file test stays inside its actual cwd jail.
    temp = Path(directory).resolve()
    sample = temp / "sample.txt"
    receipt = temp / "receipt.json"
    starts = temp / "server-starts.txt"
    marker = "mcp-file-" + secrets.token_hex(12)
    sample.write_text(marker, encoding="utf-8")
    env = os.environ.copy()
    env["SOFUU_BIN"] = str(binary)
    env["SOFUU_HOME"] = str(temp / "home")
    env["SOFUU_REAL_MCP_SERVER"] = str(root / "scripts/contract/harness/real_file_mcp_server.js")
    result = subprocess.run(
        [str(binary), "run", str(root / "scripts/contract/harness/real_file_mcp_client.js"),
         str(sample), str(receipt), marker, str(starts)], cwd=temp, env=env, timeout=30,
    )
    if result.returncode:
        raise SystemExit(result.returncode)

    # A real minimal loopback MCP service: its tools perform filesystem work;
    # only the protocol negotiation and tool descriptions are static data.
    sample.write_text(marker, encoding="utf-8")
    receipt.unlink(missing_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            method = request.get("method")
            if method == "notifications/initialized":
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if method == "initialize":
                value = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                         "serverInfo": {"name": "real-http-file", "version": "1"}}
            elif method == "ping":
                value = {}
            elif method == "tools/list":
                value = {"tools": [{"name": "read_sample", "inputSchema": {"type": "object"}},
                                   {"name": "record_and_wait", "inputSchema": {"type": "object"}}]}
            elif method == "tools/call" and request.get("params", {}).get("name") == "read_sample":
                contents = sample.read_text(encoding="utf-8")
                receipt.write_text(contents, encoding="utf-8")
                value = {"content": [{"type": "text", "text": contents}]}
            elif method == "tools/call" and request.get("params", {}).get("name") == "record_and_wait":
                with receipt.open("a", encoding="utf-8") as output:
                    output.write("recorded\n")
                time.sleep(1)
                value = {"content": [{"type": "text", "text": "recorded"}]}
            else:
                self.send_error(400)
                return
            body = json.dumps({"jsonrpc": "2.0", "id": request.get("id"), "result": value}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if method == "initialize":
                self.send_header("Mcp-Session-Id", "real-test-session")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_DELETE(self):
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env["SOFUU_ALLOW_LOOPBACK"] = "1"
    try:
        result = subprocess.run(
            [str(binary), "run", str(root / "scripts/contract/harness/real_http_mcp_client.js"),
             "http://127.0.0.1:%d/" % server.server_port, str(receipt), marker],
            cwd=temp, env=env, timeout=30,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    raise SystemExit(result.returncode)
