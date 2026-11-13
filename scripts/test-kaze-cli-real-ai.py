#!/usr/bin/env python3
"""Real `sofuu script run` check for an ai/tool agent.

The CLI is its own entry point: a script whose first line declares `ai`/`tool`
must reach the real provider and a real MCP server through the asynchronous
bridge, with the same grants, policy and configuration hierarchy as the
synchronous runner.

Set SOFUU_TEST_PROVIDER, SOFUU_TEST_MODEL and SOFUU_TEST_BASE_URL. The API key
comes from SOFUU_TEST_API_KEY, or from the operator's own
`~/.sofuu/config.json` entry for that provider when the variable is absent;
either way it only ever reaches the child process environment. SOFUU_BIN
selects the artifact under test. No credential, prompt or model text is
written to disk or printed, and no model weights are downloaded.

What this proves: one `agent.kaze` file with no sidecar read a real temporary
file through an authorized stdio MCP server and returned a real model answer
that contains the fresh marker from that file. What it does not prove: local
GGUF weights, other providers, bytecode artifacts, or released binaries.
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
    log.info("kaze cli bridge check")
    let sample = tool.call("read_sample")
    return ai.prompt("The sample file contains the token " + sample + "\\nReply with exactly that token and nothing else.")
"""


def operator_key(provider):
    """The key the operator already configured for this provider, if any.

    Reading it here keeps a test run from needing the secret on the command
    line; the value is never printed, logged or copied outside the child env.
    """
    config = Path(os.environ.get("SOFUU_TEST_CONFIG",
                                 Path.home() / ".sofuu" / "config.json"))
    try:
        document = json.loads(config.read_text())
    except (OSError, ValueError):
        return None
    entries = document.get("providers")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name") == provider:
            key = entry.get("api_key")
            return key if isinstance(key, str) and key else None
    return None


def run(command, env, cwd, timeout):
    try:
        return subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def main():
    root = Path(__file__).resolve().parent.parent
    binary = Path(os.environ.get("SOFUU_BIN", root / "target/debug/sofuu")).resolve()
    required = ("SOFUU_TEST_PROVIDER", "SOFUU_TEST_MODEL", "SOFUU_TEST_BASE_URL")
    if not binary.is_file() or not os.access(binary, os.X_OK) or any(
            not os.environ.get(key) for key in required):
        print("BLOCKED: executable and real provider test configuration are required",
              file=sys.stderr)
        return 2
    provider = os.environ["SOFUU_TEST_PROVIDER"]
    api_key = os.environ.get("SOFUU_TEST_API_KEY") or operator_key(provider)
    if not api_key:
        print("BLOCKED: no API key for that provider (SOFUU_TEST_API_KEY or "
              "~/.sofuu/config.json)", file=sys.stderr)
        return 2
    base = os.environ["SOFUU_TEST_BASE_URL"].rstrip("/")
    if not base.startswith("https://"):
        print("BLOCKED: provider base URL must use HTTPS", file=sys.stderr)
        return 2
    endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
    print(f"Artifact SHA-256: {hashlib.sha256(binary.read_bytes()).hexdigest()}", flush=True)

    server = root / "scripts/contract/harness/real_file_mcp_server.js"
    with tempfile.TemporaryDirectory(prefix="sofuu-kaze-cli-") as directory:
        temp = Path(directory).resolve()
        agent = temp / "agent.kaze"
        sample, receipt, starts = temp / "sample.txt", temp / "receipt.json", temp / "starts.txt"
        marker = "sofuu-cli-" + secrets.token_hex(12)
        agent.write_text(AGENT, encoding="utf-8")
        sample.write_text(marker, encoding="utf-8")
        policy_path = temp / "policy.json"
        policy_path.write_text(json.dumps(
            {"tools": {"allow": ["kaze.*", "read_sample"]}}), encoding="utf-8")
        mcp_path = temp / "mcp.json"
        mcp_path.write_text(json.dumps([{
            "name": "real-file-read",
            "command": str(binary),
            "args": ["run", str(server), str(sample), str(receipt), str(starts)],
            "tools": {"allow": ["read_sample"]},
        }]), encoding="utf-8")

        # The provider options travel in the documented SOFUU_AI_* variables
        # and the key in OPENAI_API_KEY; the MCP server list comes from
        # SOFUU_MCP_CONFIG, so the operator's own ~/.sofuu/config.json stays
        # untouched and unread by the test.
        env = os.environ.copy()
        env.pop("SOFUU_TEST_API_KEY", None)
        env["OPENAI_API_KEY"] = api_key
        env["SOFUU_MCP_CONFIG"] = str(mcp_path)
        env["SOFUU_AI_PROVIDER"] = provider
        env["SOFUU_AI_MODEL"] = os.environ["SOFUU_TEST_MODEL"]
        env["SOFUU_AI_BASE_URL"] = endpoint
        # No max-token cap: a small budget can be spent inside the
        # provider's own reasoning and return a genuinely empty
        # completion, which would make this check flaky rather than
        # informative.

        command = [str(binary), "script", "run", str(agent),
                   "--cap", "ai", "--cap", "tool", "--cap", "log",
                   "--policy", str(policy_path)]

        # 1. The human-facing run: a real tool call and a real model answer.
        result = run(command, env, temp, 240)
        if result is None:
            print("FAIL: the CLI run exceeded its deadline", file=sys.stderr)
            return 1
        if result.returncode != 0:
            print(f"FAIL: the CLI run exited {result.returncode}", file=sys.stderr)
            print(result.stderr.strip()[-2000:], file=sys.stderr)
            return 1
        if marker not in result.stdout:
            print("FAIL: the model answer did not contain the tool's marker", file=sys.stderr)
            return 1
        if "simulation" in result.stdout.lower():
            print("FAIL: a real run was reported as simulation", file=sys.stderr)
            return 1
        if not receipt.is_file() or json.loads(receipt.read_text())["contents"] != marker:
            print("FAIL: the MCP read did not record its independent receipt", file=sys.stderr)
            return 1
        print("PASS: `sofuu script run` returned a real model answer grounded in a real MCP read")

        # 2. The same run in the machine-readable form.
        receipt.unlink()
        machine = run(command + ["--json"], env, temp, 240)
        if machine is None or machine.returncode != 0:
            code = "timeout" if machine is None else machine.returncode
            print(f"FAIL: the --json run exited {code}", file=sys.stderr)
            return 1
        try:
            report = json.loads(machine.stdout)
        except ValueError:
            print("FAIL: --json printed no parseable report", file=sys.stderr)
            return 1
        if report.get("ok") is not True or marker not in str(report.get("result", "")):
            print("FAIL: the JSON report did not carry the real answer", file=sys.stderr)
            return 1
        if report.get("simulated_families"):
            print("FAIL: the JSON report labelled a real run as simulated", file=sys.stderr)
            return 1
        if not receipt.is_file():
            print("FAIL: the --json run recorded no MCP receipt", file=sys.stderr)
            return 1
        print("PASS: the same routine reports machine-readable success with real evidence")

        # 3. The compiled artifact: a `.kazebc` that declares ai/tool must run
        # through the same bridge, not be refused for being bytecode.
        receipt.unlink()
        artifact = temp / "agent.kazebc"
        built = run([str(binary), "script", "compile", str(agent), "-o", str(artifact)],
                    env, temp, 120)
        if built is None or built.returncode != 0 or not artifact.is_file():
            print("FAIL: the agent could not be compiled to .kazebc", file=sys.stderr)
            return 1
        bytecode = run([str(binary), "script", "run", str(artifact),
                        "--cap", "ai", "--cap", "tool", "--cap", "log",
                        "--policy", str(policy_path), "--json"], env, temp, 240)
        if bytecode is None or bytecode.returncode != 0:
            code = "timeout" if bytecode is None else bytecode.returncode
            print(f"FAIL: the .kazebc run exited {code}", file=sys.stderr)
            return 1
        try:
            report = json.loads(bytecode.stdout)
        except ValueError:
            print("FAIL: the .kazebc run printed no parseable report", file=sys.stderr)
            return 1
        if report.get("ok") is not True or marker not in str(report.get("result", "")):
            print("FAIL: the .kazebc run did not return the real answer", file=sys.stderr)
            return 1
        if not receipt.is_file() or json.loads(receipt.read_text())["contents"] != marker:
            print("FAIL: the .kazebc run recorded no MCP receipt", file=sys.stderr)
            return 1
        print("PASS: the compiled .kazebc artifact runs the same real tool and model path")

        # 4. Fail closed: no model configured, the same command must refuse.
        offline = dict(env)
        for key in ("SOFUU_AI_MODEL", "SOFUU_AI_BASE_URL", "SOFUU_AI_PROVIDER"):
            offline.pop(key, None)
        refused = run(command, offline, temp, 120)
        if refused is None:
            print("FAIL: the unconfigured run hung", file=sys.stderr)
            return 1
        if refused.returncode == 0:
            print("FAIL: the unconfigured run reported success", file=sys.stderr)
            return 1
        if marker in refused.stdout:
            print("FAIL: the unconfigured run produced the marker anyway", file=sys.stderr)
            return 1
        if "E_CAP_MISSING" not in refused.stderr:
            print("FAIL: the refusal did not name the missing AI capability",
                  file=sys.stderr)
            return 1
        print("PASS: with no model configured the CLI refuses instead of inventing an answer")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
