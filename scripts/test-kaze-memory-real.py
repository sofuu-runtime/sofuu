#!/usr/bin/env python3
"""Verify explicitly selected per-run Kaze CMA using real in-memory storage."""
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
    with tempfile.TemporaryDirectory(prefix="sofuu-kaze-memory-") as directory:
        root = Path(directory)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("AGENT_", "SOFUU_"))}
        env["HOME"] = directory
        marker = "memory-" + secrets.token_hex(10)
        label = "private-label-" + secrets.token_hex(6)
        (root / "allow.json").write_text('{"tools":{"allow":["kaze.memory.*","kaze.drone.*"]}}')
        (root / "deny.json").write_text('{"tools":{"deny":["kaze.memory.*"]}}')
        (root / "ask.json").write_text('{"tools":{"allow":["kaze.memory.*"]},"approval":{"mode":"always"}}')
        (root / "write.kaze").write_text(
            'needs: memory\naction main(value):\n'
            f'    memory.save("{label}", value)\n'
            f'    return memory.recall("{label}")\n')
        (root / "read.kaze").write_text(
            f'needs: memory\naction main():\n    return memory.recall("{label}")\n')
        (root / "bad.kaze").write_text(
            'needs: memory\naction main():\n    memory.save(123, "value")\n')
        (root / "matching.kaze").write_text(
            'needs: memory\naction main():\n'
            '    memory.save("key", "one")\n    memory.save("key", "two")\n'
            '    memory.save("key", "one")\n    memory.save("other", "secret")\n'
            '    return memory.recall("key")\n')
        (root / "mixed.kaze").write_text(
            'needs: drone, memory\naction main():\n    arm drone\n'
            '    memory.save("key", "real memory")\n    return memory.recall("key")\n')
        (root / "limit.kaze").write_text(
            'needs: memory\naction main():\n' +
            ''.join(f'    memory.save("limit", {n})\n' for n in range(257)))

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

        flags = ["--cap", "memory=ephemeral", "--policy", "allow.json"]
        payload = json.dumps({"nested": [marker, 7, True]}, separators=(',', ':'))
        proc, result = run("write.kaze", flags + ["--arg", "value=" + payload])
        assert proc.returncode == 0 and marker in result["result"], (proc.stdout, proc.stderr)
        assert label not in json.dumps(result["actions"]) and marker not in json.dumps(result["actions"]), result
        proc, result = run("read.kaze", flags)
        assert proc.returncode == 0 and result["result"] == "[]", (proc.stdout, proc.stderr)
        proc, result = run("matching.kaze", flags)
        assert proc.returncode == 0 and result["result"] == '["one", "two"]', result
        proc, result = run("mixed.kaze", ["--cap", "drone=sim"] + flags)
        assert proc.returncode == 0 and result["result"] == '["real memory"]', result
        assert result["simulated_families"] == ["drone"], result
        proc, result = run("write.kaze", ["--cap", "memory", "--policy", "allow.json", "--arg", "value=x"])
        assert proc.returncode != 0 and result["code"] == "E_CAP_MISSING", result
        proc, result = run("write.kaze", ["--cap", "memory=ephemeral", "--policy", "deny.json", "--arg", "value=x"])
        assert proc.returncode != 0 and result["code"] == "E_CAP_DENIED", result
        proc, result = run("write.kaze", ["--cap", "memory=ephemeral", "--policy", "ask.json", "--arg", "value=x"])
        assert proc.returncode != 0 and result["code"] == "E_CAP_APPROVAL_REQUIRED", result
        for path in ("bad.kaze", "limit.kaze"):
            proc, result = run(path, flags)
            assert proc.returncode != 0 and result["code"] == "E_KAZE_MEMORY", (path, result)

        compiled = call(binary, "script", "compile", "write.kaze", "-o", "write.kazebc")
        assert compiled.returncode == 0, (compiled.stdout, compiled.stderr)
        proc, result = run("write.kazebc", flags + ["--arg", "value=" + payload])
        assert proc.returncode == 0 and marker in result["result"], (proc.stdout, proc.stderr)

        source = (root / "write.kaze").read_text()
        js = ("const src = " + json.dumps(source) + ";\n" +
              "const opts = {caps:{memory:'ephemeral'},policy:{tools:{allow:['kaze.memory.*']}}};\n" +
              "for (const api of [sofuu.kaze, sofuu.sfs]) {\n" +
              "const r = api.run(src, {value:" + payload + "}, opts);\n" +
              "if (!r.ok || r.result.indexOf(" + json.dumps(marker) + ") < 0) throw Error(JSON.stringify(r)); }\n" +
              "console.log('JS memory passed');")
        proc = call(binary, "eval", js)
        assert proc.returncode == 0 and "JS memory passed" in proc.stdout, (proc.stdout, proc.stderr)

        proc = call(binary, "compile", "write.kaze", "-o", "memory-agent")
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        proc = call(root / "memory-agent", *flags, "--arg", "value=" + payload)
        assert proc.returncode == 0 and marker in proc.stdout, (proc.stdout, proc.stderr)
    print("PASS: Kaze per-run CMA, isolation, policy, limits, source, bytecode, JS and standalone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
