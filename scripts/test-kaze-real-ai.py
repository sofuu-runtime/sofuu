#!/usr/bin/env python3
"""Pure-Kaze real milestone: an authorized MCP tool reads an actual file and a
real model receives its marker through the asynchronous Kaze bridge.

Set SOFUU_TEST_PROVIDER, SOFUU_TEST_MODEL, SOFUU_TEST_BASE_URL and
SOFUU_TEST_API_KEY. SOFUU_BIN selects the runtime artifact under test; the same
artifact is also the compile base for the standalone run. No credential, prompt
or model text is written to disk or printed.

The program is one `agent.kaze` file with no per-agent JavaScript sidecar. The
source path uses a generic driver that calls the documented
`sofuu.kaze.runAsync` entry point with that same file; the standalone path runs
the compiled artifact whose bootstrap uses the same bridge.
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
    log.info("kaze real mcp + model check")
    let sample = tool.call("read_sample")
    return ai.prompt("The sample file contains the token " + sample + "\\nReply with exactly that token and nothing else.")
"""

DRIVER = """(function () {
  var env = process.env;
  var loaded = JSON.parse(sofuu.policy.load());
  if (!loaded || loaded.ok !== true) {
    console.error('FAIL kaze real ai: E_TEST_POLICY');
    process.exit(1);
  }
  sofuu.kaze.runAsync(env.SOFUU_TEST_AGENT, {}, {
    timeoutMs: 150000,
    grants: ['ai', 'tool', 'log'],
    policy: loaded.policy,
    ai: {
      provider: env.SOFUU_TEST_PROVIDER,
      model: env.SOFUU_TEST_MODEL,
      base_url: env.SOFUU_TEST_ENDPOINT
    },
    mcp: { path: env.SOFUU_TEST_MCP_CONFIG }
  }).then(function (report) {
    if (!report || report.ok !== true) {
      console.error('FAIL kaze real ai: E_TEST_NO_REPORT');
      process.exit(1);
    }
    var text = String(report.result == null ? '' : report.result);
    if (text.indexOf(env.SOFUU_TEST_MARKER) === -1) {
      console.error('FAIL kaze real ai: E_TEST_MARKER_ABSENT');
      process.exit(1);
    }
    console.log('PASS kaze real ai source');
  }, function (error) {
    console.error('FAIL kaze real ai: ' + ((error && error.code) || 'E_TEST_BRIDGE'));
    process.exit(1);
  });
})();
"""


def run(command, env, cwd, timeout):
    try:
        return subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def main():
    root = Path(__file__).resolve().parent.parent
    binary = Path(os.environ.get("SOFUU_BIN", root / "sofuu")).resolve()
    required = ("SOFUU_TEST_PROVIDER", "SOFUU_TEST_MODEL", "SOFUU_TEST_BASE_URL",
                "SOFUU_TEST_API_KEY")
    if not binary.is_file() or not os.access(binary, os.X_OK) or any(
            not os.environ.get(key) for key in required):
        print("BLOCKED: executable and real provider test configuration are required",
              file=sys.stderr)
        return 2
    base = os.environ["SOFUU_TEST_BASE_URL"].rstrip("/")
    if not base.startswith("https://"):
        print("BLOCKED: provider base URL must use HTTPS", file=sys.stderr)
        return 2
    endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
    print(f"Artifact SHA-256: {hashlib.sha256(binary.read_bytes()).hexdigest()}", flush=True)

    server = root / "scripts/contract/harness/real_file_mcp_server.js"
    with tempfile.TemporaryDirectory(prefix="sofuu-kaze-real-") as directory:
        temp = Path(directory).resolve()
        home = temp / "home"
        (home / ".sofuu").mkdir(parents=True)
        agent, driver = temp / "agent.kaze", temp / "driver.js"
        sample, receipt, starts = temp / "sample.txt", temp / "receipt.json", temp / "starts.txt"
        marker = "sofuu-kaze-" + secrets.token_hex(12)
        agent.write_text(AGENT, encoding="utf-8")
        driver.write_text(DRIVER, encoding="utf-8")
        sample.write_text(marker, encoding="utf-8")

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
        env["OPENAI_API_KEY"] = env.pop("SOFUU_TEST_API_KEY")
        env["HOME"] = str(home)
        env["SOFUU_HOME"] = str(home)
        env["SOFUU_TEST_AGENT"] = str(agent)
        env["SOFUU_TEST_ENDPOINT"] = endpoint
        env["SOFUU_TEST_MARKER"] = marker
        env["SOFUU_TEST_MCP_CONFIG"] = str(mcp_path)

        result = run([str(binary), "run", str(driver)], env, temp, 260)
        if result is None:
            print("FAIL: the source-path run exceeded its deadline", file=sys.stderr)
            return 1
        if result.returncode != 0 or "PASS kaze real ai source" not in result.stdout:
            detail = next((line for line in result.stderr.splitlines()
                           if line.startswith("FAIL kaze real ai: E_TEST_")), None)
            print(detail or f"FAIL: the source-path run exited {result.returncode}",
                  file=sys.stderr)
            return 1
        if not receipt.is_file() or json.loads(receipt.read_text())["contents"] != marker:
            print("FAIL: the MCP read did not record its independent receipt", file=sys.stderr)
            return 1
        print("PASS: a pure Kaze routine read a real file through MCP and returned "
              "a real model answer (source path)")

        standalone = temp / "kaze-agent"
        compiled = run([str(binary), "compile", str(agent), "-o", str(standalone)],
                       env, temp, 300)
        if compiled is None or compiled.returncode != 0 or not standalone.is_file():
            print("FAIL: the Kaze agent could not be compiled to a standalone artifact",
                  file=sys.stderr)
            return 1
        receipt.unlink()

        # The compiled artifact has no flag of its own for the model, so the
        # coordinator reads the documented SOFUU_AI_* environment. Its policy
        # and MCP servers come from $HOME/.sofuu, the same files the CLI reads.
        env["SOFUU_AI_PROVIDER"] = os.environ["SOFUU_TEST_PROVIDER"]
        env["SOFUU_AI_MODEL"] = os.environ["SOFUU_TEST_MODEL"]
        env["SOFUU_AI_BASE_URL"] = endpoint
        result = run([str(standalone), "--cap", "ai", "--cap", "tool", "--cap", "log"],
                     env, temp, 260)
        if result is None:
            print("FAIL: the standalone run exceeded its deadline", file=sys.stderr)
            return 1
        if result.returncode == 0 and marker in result.stdout:
            if not receipt.is_file() or json.loads(receipt.read_text())["contents"] != marker:
                print("FAIL: the standalone MCP read recorded no receipt", file=sys.stderr)
                return 1
            print("PASS: the same pure Kaze routine returned a real model answer "
                  "(standalone path)")
            return 0
        print(f"FAIL: the standalone run exited {result.returncode} without the marker",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
