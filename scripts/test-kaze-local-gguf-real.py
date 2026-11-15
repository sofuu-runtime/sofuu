#!/usr/bin/env python3
"""Pure-Kaze real milestone with local GGUF weights: an authorized MCP tool reads
an actual file and a real local model loads existing GGUF weights on this Mac to
process the prompt through the asynchronous Kaze bridge.

Strictly adheres to AGENTS.md: zero model weights downloaded. Uses the existing
local weights at ~/Downloads/SmolLM2-360M-Instruct-Q8_0.gguf.
"""
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

AGENT = """needs: ai, tool, log

action main():
    log.info("kaze real mcp + local gguf check")
    let sample = tool.call("read_sample")
    return ai.prompt("Document: " + sample + "\\nWhat is the secret word? Answer:")
"""

DRIVER = """(function () {
  var env = process.env;
  var loaded = JSON.parse(sofuu.policy.load());
  if (!loaded || loaded.ok !== true) {
    console.error('FAIL kaze local ai: E_TEST_POLICY');
    process.exit(1);
  }
  sofuu.kaze.runAsync(env.SOFUU_TEST_AGENT, {}, {
    timeoutMs: 180000,
    grants: ['ai', 'tool', 'log'],
    policy: loaded.policy,
    ai: {
      provider: 'local',
      model: env.SOFUU_TEST_MODEL
    },
    mcp: { path: env.SOFUU_TEST_MCP_CONFIG }
  }).then(function (report) {
    if (!report || report.ok !== true) {
      console.error('FAIL kaze local ai: E_TEST_NO_REPORT: ' + JSON.stringify(report));
      process.exit(1);
    }
    var raw = report.result == null ? '' : report.result;
    var text = (typeof raw === 'object' ? JSON.stringify(raw) : String(raw)).toLowerCase();
    var marker = String(env.SOFUU_TEST_MARKER || '').toLowerCase();
    if (text.indexOf(marker) === -1) {
      console.error('FAIL kaze local ai: E_TEST_MARKER_ABSENT in output: ' + text);
      process.exit(1);
    }
    console.log('PASS kaze local ai source');
  }, function (error) {
    console.error('FAIL kaze local ai: ' + ((error && error.code) || 'E_TEST_BRIDGE') + ': ' + ((error && error.message) || error));
    process.exit(1);
  });
})();
"""


def run(command, env, cwd, timeout=240):
    try:
        return subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def main():
    root = Path(__file__).resolve().parent.parent
    binary = Path(os.environ.get("SOFUU_BIN", root / "target/debug/sofuu")).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        print("BLOCKED: executable required at target/debug/sofuu", file=sys.stderr)
        return 2

    # Verify existing local GGUF model exists
    candidate_paths = [
        Path(os.path.expanduser("~/Downloads/SmolLM2-360M-Instruct-Q8_0.gguf")),
        Path(os.path.expanduser("~/Downloads/Llama-3.2-1B-Instruct-Q8_0.gguf")),
    ]
    model_path = next((p for p in candidate_paths if p.is_file()), None)
    if not model_path:
        print("BLOCKED: No existing local GGUF weights found in ~/Downloads per AGENTS.md", file=sys.stderr)
        return 2

    print(f"Using local GGUF weights: {model_path} ({model_path.stat().st_size} bytes)")
    print(f"Artifact SHA-256: {hashlib.sha256(binary.read_bytes()).hexdigest()}", flush=True)

    server = root / "scripts/contract/harness/real_file_mcp_server.js"
    with tempfile.TemporaryDirectory(prefix="sofuu-kaze-local-ai-") as directory:
        temp = Path(directory).resolve()
        home = temp / "home"
        (home / ".sofuu").mkdir(parents=True)
        agent, driver = temp / "agent.kaze", temp / "driver.js"
        sample, receipt, starts = temp / "sample.txt", temp / "receipt.json", temp / "starts.txt"
        marker = secrets.choice(["sapphire", "emerald", "crimson", "obsidian", "turquoise", "amethyst"])

        agent.write_text(AGENT, encoding="utf-8")
        driver.write_text(DRIVER, encoding="utf-8")
        sample.write_text(f"The secret word is {marker}.", encoding="utf-8")

        policy = {"tools": {"allow": ["kaze.*", "read_sample"]}}
        policy_path = home / ".sofuu" / "policy.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        mcp_path = home / ".sofuu" / "mcp.json"
        mcp_path.write_text(json.dumps([{
            "name": "real-file-read",
            "command": str(binary),
            "args": ["run", str(server), str(sample), str(receipt), str(starts)],
            "tools": {"allow": ["read_sample"]},
        }]), encoding="utf-8")

        env = os.environ.copy()
        env["HOME"] = str(home)
        env["SOFUU_HOME"] = str(home)
        env["SOFUU_TEST_AGENT"] = str(agent)
        env["SOFUU_TEST_MODEL"] = str(model_path)
        env["SOFUU_TEST_MCP_CONFIG"] = str(mcp_path)
        env["SOFUU_TEST_MARKER"] = marker
        env["SOFUU_AI_PROVIDER"] = "local"
        env["SOFUU_AI_MODEL"] = str(model_path)

        # 1. Test via JS runAsync driver
        print("Running pure-Kaze local GGUF + MCP via sofuu.kaze.runAsync...", flush=True)
        res = run([str(binary), "run", str(driver)], env, temp, 240)
        assert res is not None, "sofuu.kaze.runAsync timed out"
        assert res.returncode == 0, (res.stdout, res.stderr)
        assert "PASS kaze local ai source" in res.stdout, (res.stdout, res.stderr)
        assert receipt.is_file(), "MCP tool receipt was not written"
        rcpt = json.loads(receipt.read_text())
        assert marker in rcpt.get("contents"), rcpt
        print("  ✓ sofuu.kaze.runAsync passed", flush=True)

        # 2. Standalone compiled agent path
        print("Compiling pure-Kaze agent to standalone binary...", flush=True)
        standalone = temp / "kaze-local-agent"
        compiled = run([str(binary), "compile", str(agent), "-o", str(standalone)],
                       env, temp, 300)
        assert compiled is not None and compiled.returncode == 0 and standalone.is_file(), (
            compiled.stdout if compiled else "timed out", compiled.stderr if compiled else "")
        receipt.unlink()

        print("Running pure-Kaze standalone binary with local GGUF + MCP...", flush=True)
        env["SOFUU_AI_PROVIDER"] = "local"
        env["SOFUU_AI_MODEL"] = str(model_path)
        standalone_res = run([str(standalone), "--cap", "ai", "--cap", "tool", "--cap", "log"], env, temp, 240)
        assert standalone_res is not None, "standalone run timed out"
        assert standalone_res.returncode == 0, (standalone_res.stdout, standalone_res.stderr)
        out_lower = standalone_res.stdout.lower()
        assert marker.lower() in out_lower, (standalone_res.stdout, standalone_res.stderr)
        assert receipt.is_file(), "standalone MCP read recorded no receipt"
        rcpt = json.loads(receipt.read_text())
        assert marker in rcpt.get("contents"), rcpt
        print("  ✓ standalone binary passed", flush=True)

    print("PASS: Pure-Kaze local GGUF model inference + real MCP tool grounding verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
