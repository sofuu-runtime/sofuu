#!/usr/bin/env python3
"""Verify Kaze durable memory persistence, process re-open, isolation, and policy enforcement."""
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

    with tempfile.TemporaryDirectory(prefix="sofuu-kaze-durable-") as directory:
        root = Path(directory)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("AGENT_", "SOFUU_"))}
        env["HOME"] = directory
        store_path = root / "store.json"
        marker = "durable-" + secrets.token_hex(10)
        label = "my-durable-key-" + secrets.token_hex(4)

        (root / "allow.json").write_text('{"tools":{"allow":["kaze.memory.*"]}}')
        (root / "deny.json").write_text('{"tools":{"deny":["kaze.memory.*"]}}')
        (root / "ask.json").write_text('{"tools":{"allow":["kaze.memory.*"]},"approval":{"mode":"always"}}')

        (root / "write.kaze").write_text(
            'needs: memory\naction main(value):\n'
            f'    memory.save("{label}", value)\n'
            f'    return memory.recall("{label}")\n')
        (root / "read.kaze").write_text(
            f'needs: memory\naction main():\n    return memory.recall("{label}")\n')
        (root / "bad.kaze").write_text(
            'needs: memory\naction main():\n    memory.save(999, "bad")\n')

        def call(*args):
            return subprocess.run([str(a) for a in args], cwd=root, env=env,
                                  capture_output=True, text=True, timeout=30)

        def run(path, flags):
            proc = call(binary, "script", "run", path, *flags, "--json")
            try:
                result = json.loads(proc.stdout)
            except json.JSONDecodeError:
                raise AssertionError((proc.args, proc.stdout, proc.stderr))
            return proc, result

        flags = ["--cap", f"memory=file:{store_path}", "--policy", "allow.json"]
        payload = json.dumps({"token": marker, "count": 42}, separators=(',', ':'))

        # 1. Process 1: Write to durable store
        proc, result = run("write.kaze", flags + ["--arg", f"value={payload}"])
        assert proc.returncode == 0 and marker in result["result"], (proc.stdout, proc.stderr)
        assert store_path.exists(), "store file was not created on disk"

        # 2. Process 2: Reopen in a separate process and recall the data written by Process 1
        proc, result = run("read.kaze", flags)
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        assert marker in result["result"] and "42" in result["result"], result

        # 3. Policy denial fails closed with E_CAP_DENIED
        proc, result = run("read.kaze", ["--cap", f"memory=file:{store_path}", "--policy", "deny.json"])
        assert proc.returncode != 0 and result["code"] == "E_CAP_DENIED", result

        # 4. Approval policy fails with E_CAP_APPROVAL_REQUIRED
        proc, result = run("read.kaze", ["--cap", f"memory=file:{store_path}", "--policy", "ask.json"])
        assert proc.returncode != 0 and result["code"] == "E_CAP_APPROVAL_REQUIRED", result

        # 5. Ephemeral store in a new process must NOT see the durable store
        proc, result = run("read.kaze", ["--cap", "memory=ephemeral", "--policy", "allow.json"])
        assert proc.returncode == 0 and result["result"] == "[]", result

        # 6. Default durable backend (--cap memory=durable)
        durable_marker = "def-durable-" + secrets.token_hex(6)
        durable_flags = ["--cap", "memory=durable", "--policy", "allow.json"]
        durable_payload = json.dumps({"val": durable_marker})
        proc, result = run("write.kaze", durable_flags + ["--arg", f"value={durable_payload}"])
        assert proc.returncode == 0 and durable_marker in result["result"], result

        proc, result = run("read.kaze", durable_flags)
        assert proc.returncode == 0, result
        assert durable_marker in result["result"], result

        # 7. Bytecode persistence parity
        compiled = call(binary, "script", "compile", "read.kaze", "-o", "read.kazebc")
        assert compiled.returncode == 0, (compiled.stdout, compiled.stderr)
        proc, result = run("read.kazebc", flags)
        assert proc.returncode == 0 and marker in result["result"], (proc.stdout, proc.stderr)

        # 8. Standalone compiled agent persistence
        proc = call(binary, "compile", "read.kaze", "-o", "read-agent")
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        proc = call(root / "read-agent", *flags)
        assert proc.returncode == 0 and marker in proc.stdout, (proc.stdout, proc.stderr)

    print("PASS: Kaze durable memory (process reopen, persistence, isolation, bytecode, standalone)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
