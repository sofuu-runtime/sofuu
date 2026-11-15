#!/usr/bin/env python3
"""Verify real alert sinks (stdout, stderr, file:<path>) with truthful fail-closed behavior."""
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile


def main():
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "target/debug/sofuu").resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        print("BLOCKED: build a Sofuu binary first", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="sofuu-kaze-alert-") as directory:
        root = Path(directory)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("AGENT_", "SOFUU_"))}
        env["HOME"] = directory
        alert_file = root / "alerts.log"
        marker = "alert-marker-" + secrets.token_hex(8)

        (root / "allow.json").write_text('{"tools":{"allow":["kaze.alert.*"]}}')
        (root / "deny.json").write_text('{"tools":{"deny":["kaze.alert.*"]}}')
        (root / "ask.json").write_text('{"tools":{"allow":["kaze.alert.*"]},"approval":{"mode":"always"}}')

        (root / "alert.kaze").write_text(
            'needs: alert\naction main(msg):\n    alert.send(msg)\n')
        (root / "bad.kaze").write_text(
            'needs: alert\naction main():\n    alert.send("")\n')

        def call(*args):
            return subprocess.run([str(a) for a in args], cwd=root, env=env,
                                  capture_output=True, text=True, timeout=30)

        def run(path, flags):
            proc = call(binary, "script", "run", path, *flags, "--json")
            try:
                # When alert writes to stdout, json output follows
                idx = proc.stdout.find("{\n  ")
                if idx == -1:
                    idx = proc.stdout.find("{")
                json_str = proc.stdout[idx:] if idx != -1 else proc.stdout
                result = json.loads(json_str)
            except json.JSONDecodeError:
                raise AssertionError((proc.args, proc.stdout, proc.stderr))
            return proc, result

        # 1. Unconfigured alert backend fails closed with E_CAP_MISSING
        proc, result = run("alert.kaze", ["--cap", "alert", "--policy", "allow.json", "--arg", f"msg={marker}"])
        assert proc.returncode != 0 and result["code"] == "E_CAP_MISSING", (proc.stdout, proc.stderr)

        # 2. Denied policy fails closed with E_CAP_DENIED
        proc, result = run("alert.kaze", ["--cap", "alert=stdout", "--policy", "deny.json", "--arg", f"msg={marker}"])
        assert proc.returncode != 0 and result["code"] == "E_CAP_DENIED", (proc.stdout, proc.stderr)

        # 3. Approval policy requires approval
        proc, result = run("alert.kaze", ["--cap", "alert=stdout", "--policy", "ask.json", "--arg", f"msg={marker}"])
        assert proc.returncode != 0 and result["code"] == "E_CAP_APPROVAL_REQUIRED", (proc.stdout, proc.stderr)

        # 4. Stdout alert sink
        proc, result = run("alert.kaze", ["--cap", "alert=stdout", "--policy", "allow.json", "--arg", f"msg={marker}"])
        assert proc.returncode == 0 and result["ok"] is True, (proc.stdout, proc.stderr)
        assert f"[ALERT] {marker}" in proc.stdout, proc.stdout

        # 5. Stderr alert sink
        proc, result = run("alert.kaze", ["--cap", "alert=stderr", "--policy", "allow.json", "--arg", f"msg={marker}"])
        assert proc.returncode == 0 and result["ok"] is True, (proc.stdout, proc.stderr)
        assert f"[ALERT] {marker}" in proc.stderr, proc.stderr

        # 6. File alert sink
        proc, result = run("alert.kaze", ["--cap", f"alert=file:{alert_file}", "--policy", "allow.json", "--arg", f"msg={marker}"])
        assert proc.returncode == 0 and result["ok"] is True, (proc.stdout, proc.stderr)
        assert alert_file.exists(), "alert file was not created"
        logged = alert_file.read_text()
        assert f"[ALERT] {marker}" in logged, logged

        # 7. Range check on empty alert
        proc, result = run("bad.kaze", ["--cap", "alert=stdout", "--policy", "allow.json"])
        assert proc.returncode != 0 and result["code"] == "E_KAZE_RANGE", (proc.stdout, proc.stderr)

        # 8. Bytecode compilation & execution with alert
        compiled = call(binary, "script", "compile", "alert.kaze", "-o", "alert.kazebc")
        assert compiled.returncode == 0, (compiled.stdout, compiled.stderr)
        proc, result = run("alert.kazebc", ["--cap", f"alert=file:{alert_file}", "--policy", "allow.json", "--arg", f"msg=from-bc-{marker}"])
        assert proc.returncode == 0 and result["ok"] is True, (proc.stdout, proc.stderr)
        assert f"[ALERT] from-bc-{marker}" in alert_file.read_text()

        # 9. Standalone compiled agent with alert
        proc = call(binary, "compile", "alert.kaze", "-o", "alert-agent")
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        proc = call(root / "alert-agent", "--cap", f"alert=file:{alert_file}", "--policy", "allow.json", "--arg", f"msg=from-standalone-{marker}")
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        assert f"[ALERT] from-standalone-{marker}" in alert_file.read_text()

    print("PASS: Kaze real alert sinks (stdout, stderr, file), policy, limits, source, bytecode, and standalone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
