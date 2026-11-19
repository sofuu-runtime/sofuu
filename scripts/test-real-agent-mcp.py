#!/usr/bin/env python3
"""Run the real agent + MCP contract case (T5.6) against a real backend.

Two real backends are supported and nothing is substituted:

  * a configured provider — SOFUU_TEST_PROVIDER, SOFUU_TEST_MODEL and
    SOFUU_TEST_BASE_URL name it, SOFUU_TEST_CONTEXT_WINDOW names the model's
    context window and SOFUU_TEST_API_KEY carries its credential. A remote
    endpoint must use HTTPS; a loopback endpoint (a local OpenAI-compatible
    server) may use plain HTTP;
  * a local backend — SOFUU_TEST_LOCAL_MODEL names an existing local GGUF and
    the real local HTTP server (`sofuu serve`) serves it; the agent then talks
    to that server over the OpenAI wire. When SOFUU_TEST_LOCAL_MODEL is unset
    the weights the contract runner selected (SOFUU_CONTRACT_MODEL) are used,
    and otherwise the first available candidate below.

No model weights are downloaded and none are removed. Credentials are never
written to disk or printed. SOFUU_BIN selects the runtime artifact under
test. A model that does not follow the instruction is a failed run, never
evidence: the case only reports PASS when the tool really ran and the model's
own answer carries the freshly written marker.
"""
import hashlib
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


# Existing local weights only (AGENTS.md): the first one present is used.
LOCAL_MODEL_CANDIDATES = (
    "Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf",
    "Qwen-3-0.6B-Claude-4.7-Opus-Distilled.i1-Q4_K_M.gguf",
    "Llama-3.2-1B-Instruct-Q8_0.gguf",
    "SmolLM2-360M-Instruct-Q8_0.gguf",
)


def block(message):
    print(f"BLOCKED: {message}", file=sys.stderr)
    return 2


def plain_lines(text):
    """Lines without the runtime's ANSI colouring, so a typed failure line
    can be matched and reported verbatim."""
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    return [ansi.sub("", line).strip() for line in text.splitlines()]


def is_loopback(base_url):
    """True for a plain-HTTP endpoint on this machine."""
    if not base_url.startswith("http://"):
        return False
    host = base_url[len("http://"):].split("/")[0].split(":")[0]
    return host in ("127.0.0.1", "localhost", "::1", "[::1]")


def http_ok(url, headers, timeout=3.0):
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def pick_local_model():
    """An existing local GGUF, or None. Never downloads, never moves files."""
    configured = os.environ.get("SOFUU_TEST_LOCAL_MODEL")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    # The contract runner exports the weights it selected for every case:
    # the local path serves the same file, so one suite runs on one model.
    shared = os.environ.get("SOFUU_CONTRACT_MODEL")
    if shared:
        path = Path(shared).expanduser()
        if path.is_file():
            return path
    for name in LOCAL_MODEL_CANDIDATES:
        path = Path.home() / "Downloads" / name
        if path.is_file():
            return path
    return None


def free_port():
    """A free loopback port for the local server."""
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def start_local_server(binary, model_path, home):
    """Start the real local HTTP server on a free loopback port."""
    port = free_port()
    env = os.environ.copy()
    env["SOFUU_HOME"] = str(home)
    env["HOME"] = str(home)
    process = subprocess.Popen(
        [str(binary), "serve", "--model", str(model_path),
         "--model-id", "contract-local-model", "--port", str(port),
         "--host", "127.0.0.1", "--token", "contract-local-token",
         "--brain", str(home / "brain.qtsq")],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    headers = {"Authorization": "Bearer contract-local-token"}
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return None, None
        if http_ok(base + "/v1/models", headers):
            return process, (base + "/v1", headers)
        time.sleep(0.5)
    process.terminate()
    return None, None


def main():
    root = Path(__file__).resolve().parent.parent
    binary = Path(os.environ.get("SOFUU_BIN", root / "target/debug/sofuu")).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        return block("an executable runtime artifact is required (set SOFUU_BIN)")
    remote_names = ("SOFUU_TEST_PROVIDER", "SOFUU_TEST_MODEL", "SOFUU_TEST_BASE_URL")
    remote_configured = all(os.environ.get(name) for name in remote_names)
    window_text = os.environ.get("SOFUU_TEST_CONTEXT_WINDOW", "")
    if not window_text:
        return block("SOFUU_TEST_CONTEXT_WINDOW must name the model's context window")
    try:
        window = int(window_text)
        if window <= 0 or window > 2_147_483_647:
            raise ValueError()
    except ValueError:
        return block("SOFUU_TEST_CONTEXT_WINDOW must be a positive 32-bit integer")
    if remote_configured:
        base = os.environ["SOFUU_TEST_BASE_URL"].rstrip("/")
        # A loopback endpoint is a local server: plain HTTP there carries no
        # credential over a network. Everything else must use HTTPS.
        if not (base.startswith("https://") or is_loopback(base)):
            return block("a non-loopback provider base URL must use HTTPS")
        endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        provider = os.environ["SOFUU_TEST_PROVIDER"]
        model = os.environ["SOFUU_TEST_MODEL"]
        print(f"Provider: {provider} ({base})", flush=True)
    else:
        model_path = pick_local_model()
        if model_path is None:
            return block("no real model backend is available: set SOFUU_TEST_PROVIDER and "
                         "friends for a configured provider, or SOFUU_TEST_LOCAL_MODEL to an "
                         "existing local GGUF")
        provider = "llamacpp"
        endpoint = ""
        print(f"Provider: local HTTP server with {model_path.name}", flush=True)

    print(f"Artifact SHA-256: {hashlib.sha256(binary.read_bytes()).hexdigest()}", flush=True)
    with tempfile.TemporaryDirectory(prefix="sofuu-real-agent-") as directory:
        temp = Path(directory).resolve()
        home = temp / "home"
        home.mkdir(parents=True, exist_ok=True)
        sample, receipt, starts = temp / "sample.txt", temp / "receipt.json", temp / "starts.txt"
        marker = "sofuu-real-" + secrets.token_hex(12)
        sample.write_text(marker, encoding="utf-8")
        env = os.environ.copy()
        env["SOFUU_HOME"] = str(home)
        env["HOME"] = str(home)
        # The agent talks to the loopback server it was told to use.
        env["SOFUU_ALLOW_LOOPBACK"] = "1"
        script = root / "scripts/contract/harness/t56_mcp_parity.js"
        server = root / "scripts/contract/harness/real_file_mcp_server.js"
        server_process = None
        try:
            if remote_configured:
                api_key = os.environ.get("SOFUU_TEST_API_KEY", "")
                if api_key:
                    env["OPENAI_API_KEY"] = api_key
                    env["SOFUU_TEST_PROVIDER_KEY"] = api_key
                args = [str(binary), "run", str(script), str(binary), str(server), str(sample),
                        str(receipt), str(starts), marker, provider, model, str(window), endpoint]
            else:
                server_process, (base_url, headers) = start_local_server(binary, model_path, home)
                if server_process is None:
                    return block("the local HTTP server did not come up with the configured model")
                env["SOFUU_TEST_PROVIDER_KEY"] = headers["Authorization"].split(" ", 1)[1]
                args = [str(binary), "run", str(script), str(binary), str(server), str(sample),
                        str(receipt), str(starts), marker, provider, "contract-local-model",
                        str(window), base_url]
            try:
                result = subprocess.run(args, cwd=temp, env=env, capture_output=True,
                                        text=True, timeout=900)
            except subprocess.TimeoutExpired:
                print("FAIL: the real agent test exceeded its deadline", file=sys.stderr)
                return 1
            if result.returncode == 0 and "PASS real agent model and MCP file read" in result.stdout:
                print("PASS: real model used MCP and built-in file reads with matching event envelopes")
                return 0
            if result.returncode == 2:
                # The harness itself reported a blocked precondition.
                for line in plain_lines(result.stderr):
                    if line.startswith("BLOCKED:"):
                        print(line, file=sys.stderr)
                        return 2
            # Do not print provider messages, prompts, model text, paths or credentials.
            failure = next((line for line in plain_lines(result.stderr)
                            if line.startswith("FAIL real agent model and MCP file read: E_TEST_")), None)
            print(failure or f"FAIL: the real agent test exited {result.returncode}", file=sys.stderr)
            return 1
        finally:
            if server_process is not None:
                try:
                    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    server_process.terminate()
                try:
                    server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server_process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
