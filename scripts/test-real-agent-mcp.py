#!/usr/bin/env python3
"""Compare real model + MCP and built-in file-read agent traces.

Set SOFUU_TEST_PROVIDER, SOFUU_TEST_MODEL, SOFUU_TEST_BASE_URL,
SOFUU_TEST_CONTEXT_WINDOW and SOFUU_TEST_API_KEY. No credentials are written
to disk or printed. SOFUU_BIN selects the runtime artifact under test.
"""
import hashlib
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parent.parent
    binary = Path(os.environ.get("SOFUU_BIN", root / "sofuu")).resolve()
    config_names = ("SOFUU_TEST_PROVIDER", "SOFUU_TEST_MODEL", "SOFUU_TEST_BASE_URL",
                    "SOFUU_TEST_CONTEXT_WINDOW", "SOFUU_TEST_API_KEY")
    if not binary.is_file() or not os.access(binary, os.X_OK) or any(not os.environ.get(k) for k in config_names):
        print("BLOCKED: executable and real provider test configuration are required", file=sys.stderr)
        return 2
    try:
        window = int(os.environ["SOFUU_TEST_CONTEXT_WINDOW"])
        if window <= 0 or window > 2_147_483_647:
            raise ValueError()
    except ValueError:
        print("BLOCKED: SOFUU_TEST_CONTEXT_WINDOW must be a positive 32-bit integer", file=sys.stderr)
        return 2
    base = os.environ["SOFUU_TEST_BASE_URL"].rstrip("/")
    if not base.startswith("https://"):
        print("BLOCKED: provider base URL must use HTTPS", file=sys.stderr)
        return 2
    endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
    print(f"Artifact SHA-256: {hashlib.sha256(binary.read_bytes()).hexdigest()}", flush=True)
    with tempfile.TemporaryDirectory(prefix="sofuu-real-agent-") as directory:
        temp = Path(directory).resolve()
        sample, receipt, starts = temp / "sample.txt", temp / "receipt.json", temp / "starts.txt"
        marker = "sofuu-real-" + secrets.token_hex(12)
        sample.write_text(marker, encoding="utf-8")
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = env.pop("SOFUU_TEST_API_KEY")
        env["SOFUU_HOME"] = str(temp / "home")
        script = root / "scripts/contract/harness/t56_mcp_parity.js"
        server = root / "scripts/contract/harness/real_file_mcp_server.js"
        try:
            result = subprocess.run(
                [str(binary), "run", str(script), str(binary), str(server), str(sample),
                 str(receipt), str(starts), marker, os.environ["SOFUU_TEST_PROVIDER"],
                 os.environ["SOFUU_TEST_MODEL"], endpoint, str(window)],
                cwd=temp, env=env, capture_output=True, text=True, timeout=260,
            )
        except subprocess.TimeoutExpired:
            print("FAIL: real agent test exceeded its deadline", file=sys.stderr)
            return 1
        if result.returncode == 0 and "PASS real agent model and MCP file read" in result.stdout:
            print("PASS: real model used MCP and built-in file reads with matching event envelopes")
            return 0
        # Do not print provider messages, prompts, model text, paths or credentials.
        failure = next((line for line in result.stderr.splitlines()
                        if line.startswith("FAIL real agent model and MCP file read: E_TEST_")), None)
        print(failure or f"FAIL: real agent test exited {result.returncode}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
