#!/usr/bin/env python3
"""Exercise embedded Kaze's gates through real local execution surfaces.

Build both binaries first:
  cargo build --offline -p sofuu-core --bin sofuu --bin sofuu-agent
  python3 scripts/test-kaze-fail-closed.py target/debug/sofuu
An optional second path selects an older base for the incompatibility check.
No model, external tool, or physical device is used. Drone runs are labelled
simulator tests and pass through the VM's normal gates.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "target/debug/sofuu").resolve()
    with tempfile.TemporaryDirectory(prefix="sofuu-kaze-gates-") as directory:
        root = Path(directory)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("AGENT_", "SOFUU_"))}
        env["HOME"] = directory
        (root / "pure.kaze").write_text("action main():\n    return 2 + 3\n")
        (root / "ai.kaze").write_text('needs: ai\naction main():\n    return ai.prompt("hello")\n')
        (root / "drone.kaze").write_text('needs: drone\naction main():\n    arm drone\n    return drone.telemetry("battery")\n')
        (root / "bad-goto.kaze").write_text('needs: drone\naction main():\n    arm drone\n    takeoff to 15 meters\n    drone.goto(91, 0, 15)\n')
        (root / "unknown-telemetry.kaze").write_text('needs: drone\naction main():\n    return drone.telemetry("unknown")\n')
        (root / "mixed.kaze").write_text('needs: drone, ai\naction main():\n    arm drone\n    return ai.prompt("hello")\n')
        (root / "allow.json").write_text('{"tools":{"allow":["kaze.*"]}}')
        (root / "deny.json").write_text('{"tools":{"deny":["kaze.ai.*"]}}')
        (root / "invalid.json").write_text('{"tools":{"deny":[],"deny":[]}}')
        (root / "unrelated.json").write_text('{"fs":{"write":"off"}}')
        (root / "empty-allow.json").write_text('{"tools":{"allow":[]}}')

        def call(*args, input_text=None):
            return subprocess.run([str(arg) for arg in args], cwd=root, env=env,
                                  input=input_text, capture_output=True, text=True, timeout=30)

        def check_run(path, flags, code=None, result=None):
            proc = call(binary, "script", "run", path, *flags, "--json")
            report = json.loads(proc.stdout)
            assert proc.returncode == (0 if code is None else 1), (proc.args, proc.stdout, proc.stderr)
            assert report.get("code") == code, (proc.args, report)
            if result is not None:
                assert report["result"] == result, report
            return report

        check_run("pure.kaze", [], result="5")
        check_run("ai.kaze", [], "E_CAP_MISSING")
        check_run("ai.kaze", ["--cap", "ai", "--no-policy"], "E_CAP_DENIED")
        for policy in ("deny.json", "invalid.json", "unrelated.json", "empty-allow.json"):
            check_run("ai.kaze", ["--cap", "ai", "--policy", policy], "E_CAP_DENIED")
        check_run("ai.kaze", ["--cap", "ai", "--policy", "allow.json"], "E_CAP_MISSING")
        check_run("drone.kaze", ["--cap", "drone", "--policy", "allow.json"], "E_CAP_MISSING")
        report = check_run("drone.kaze", ["--cap", "drone=sim", "--policy", "allow.json",
                                           "--sim", "battery=42"], result="41")
        assert report["simulated_families"] == ["drone"], report
        physical_flags = ["--cap", "drone=sim", "--policy", "allow.json"]
        check_run("bad-goto.kaze", physical_flags, "E_KAZE_RANGE")
        check_run("unknown-telemetry.kaze", physical_flags, "E_CAP_MISSING")
        check_run("mixed.kaze", physical_flags + ["--cap", "ai"], "E_CAP_MISSING")

        compiled = call(binary, "script", "compile", "ai.kaze", "-o", "ai.kazebc")
        assert compiled.returncode == 0, compiled.stderr
        check_run("ai.kazebc", ["--cap", "ai", "--policy", "allow.json"], "E_CAP_MISSING")

        js = """
        const source = 'needs: ai\\naction main():\\n    return ai.prompt("hello")\\n';
        for (const api of [sofuu.kaze, sofuu.sfs]) {
          const opts = {grants: ['ai'], policy: {tools: {allow: ['kaze.*']}}};
          const result = api.run(source, {}, opts);
          if (result.ok || result.code !== 'E_CAP_MISSING') throw Error(JSON.stringify(result));
          if (api.run(source, {}, {grants: ['ai']}).code !== 'E_CAP_DENIED') throw Error('policy');
          if (api.run(source, {}, {caps: {ai: 'sim'}}).code !== 'E_CAP_INVALID') throw Error('sim');
        }
        console.log('JS gates passed');
        """
        js_run = call(binary, "eval", js)
        assert js_run.returncode == 0 and "JS gates passed" in js_run.stdout, (js_run.stdout, js_run.stderr)

        repl = call(binary, "script", "repl", "--policy", "allow.json", "--sim", "battery=42",
                    input_text='.cap drone=sim\naction main():\n    return drone.telemetry("battery")\n\n.run main\n.exit\n')
        assert "simulation (testing): drone" in repl.stdout and "42" in repl.stdout, (repl.stdout, repl.stderr)

        for source in ("pure", "ai", "drone"):
            built = call(binary, "compile", source + ".kaze", "-o", source + "-agent")
            assert built.returncode == 0, (built.stdout, built.stderr)
        pure = call(root / "pure-agent")
        assert pure.returncode == 0 and "5" in pure.stdout, (pure.stdout, pure.stderr)
        for flags, code in (([], "E_CAP_MISSING"), (["--cap", "ai"], "E_CAP_DENIED"),
                            (["--cap", "ai", "--policy", "allow.json"], "E_CAP_MISSING"),
                            (["--cap", "ai", "--policy", "invalid.json"], "E_CAP_DENIED")):
            proc = call(root / "ai-agent", *flags)
            assert proc.returncode != 0 and code in proc.stderr, (proc.stdout, proc.stderr)
        drone = call(root / "drone-agent", "--cap", "drone=sim", "--policy", "allow.json",
                     "--sim", "battery=42")
        assert drone.returncode == 0 and "Simulation (testing): drone" in drone.stdout and "41" in drone.stdout, (drone.stdout, drone.stderr)
        if len(sys.argv) > 2:
            old_base = Path(sys.argv[2]).resolve()
            built = call(binary, "compile", "ai.kaze", "-o", "old-base-agent", "--base", old_base)
            assert built.returncode == 0, (built.stdout, built.stderr)
            stale = call(root / "old-base-agent")
            assert stale.returncode != 0 and "E_KAZE_RUNTIME" in stale.stderr, (stale.stdout, stale.stderr)

    print("Kaze fail-closed surface checks passed.")


if __name__ == "__main__":
    main()
