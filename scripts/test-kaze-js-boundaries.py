#!/usr/bin/env python3
"""Exercise Kaze's JS filesystem boundary and argument ownership with real files."""
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
    with tempfile.TemporaryDirectory(prefix="sofuu-kaze-boundary-") as directory:
        root = Path(directory).resolve()
        jail = root / "jail"
        jail.mkdir()
        script = "action main():\n    return 7\n"
        (jail / "allowed.kaze").write_text(script)
        (root / "outside.kaze").write_text(script)
        (jail / "escape.kaze").symlink_to(root / "outside.kaze")
        (root / "agent.kaze").write_text("action main(secret):\n    return secret\n")
        (root / "hooks.kaze").write_text('needs: log\nevery 100ms:\n    log "tick"\n')
        config_marker = "cfg-" + secrets.token_hex(8)
        env_marker = "env-" + secrets.token_hex(8)
        (root / "agent.config.json").write_text(json.dumps({"args": {"secret": config_marker}}))
        env = {k: v for k, v in os.environ.items() if not k.startswith(("AGENT_", "SOFUU_"))}
        env.update({"HOME": str(root), "AGENT_SECRET": env_marker})

        def call(*args):
            return subprocess.run([str(arg) for arg in args], cwd=root, env=env,
                                  capture_output=True, text=True, timeout=30)

        # Each JS operation is checked independently: a denied run must not
        # execute the file, and check/compile must not read it either.
        js = """
        const file = %s;
        const src = 'action main(secret):\\n    return secret\\n';
        sofuu.policy.set(JSON.stringify({fs:{read:'off'}}));
        for (const api of [sofuu.kaze, sofuu.sfs]) {
          for (const op of ['run', 'check', 'compile']) {
            const r = api[op](file);
            if (r.ok || r.code !== 'E_PERMISSION_DENIED') throw Error(op + ': ' + JSON.stringify(r));
          }
          const missing = api.run(src, {});
          if (missing.ok || missing.code !== 'E_KAZE_PARAM') throw Error('ambient config escaped: ' + JSON.stringify(missing));
          const passed = api.run(src, {secret: 'explicit'});
          if (!passed.ok || passed.result !== '"explicit"') throw Error(JSON.stringify(passed));
          for (const bad of [[], 4, 'text']) {
            const r = api.run(src, bad);
            if (r.ok || r.code !== 'E_KAZE_ARGS') throw Error('invalid args: ' + JSON.stringify(r));
          }
          const cycle = {}; cycle.self = cycle;
          if (api.run(src, cycle).code !== 'E_KAZE_ARGS') throw Error('cycle accepted');
        }
        const memory = 'needs: memory\\naction main():\\n    memory.save("key", "real")\\n    return memory.recall("key")\\n';
        const opts = {caps:{memory:'ephemeral'}, policy:{tools:{allow:['kaze.memory.*']}}};
        for (const api of [sofuu.kaze, sofuu.sfs]) {
          sofuu.policy.set(JSON.stringify({tools:{deny:['kaze.memory.*']}}));
          let r = api.run(memory, {}, opts);
          if (r.ok || r.code !== 'E_CAP_DENIED') throw Error('host deny bypassed: ' + JSON.stringify(r));
          sofuu.policy.set(JSON.stringify({approval:{mode:'always'}}));
          r = api.run(memory, {}, opts);
          if (r.ok || r.code !== 'E_CAP_APPROVAL_REQUIRED') throw Error('host approval bypassed: ' + JSON.stringify(r));
          sofuu.policy.set('{}');
          r = api.run(memory, {}, opts);
          if (!r.ok || r.result !== '["real"]') throw Error('real memory: ' + JSON.stringify(r));
        }
        console.log('JS file and argument boundaries passed');
        """ % json.dumps(str(root / "outside.kaze"))
        proc = call(binary, "eval", js)
        assert proc.returncode == 0 and "JS file and argument boundaries passed" in proc.stdout, (proc.stdout, proc.stderr)
        assert config_marker not in proc.stdout + proc.stderr and env_marker not in proc.stdout + proc.stderr

        root_js = """
        sofuu.fs.setRoot(%s);
        for (const api of [sofuu.kaze, sofuu.sfs]) {
          for (const op of ['run', 'check', 'compile']) {
            const good = api[op]('allowed.kaze');
            if (!good.ok) throw Error('allowed: ' + JSON.stringify(good));
            for (const path of [%s, 'escape.kaze']) {
              const bad = api[op](path);
              if (bad.ok || !['E_CAP_DENIED', 'E_PATH_OUTSIDE'].includes(bad.code)) {
                throw Error('escaped: ' + JSON.stringify(bad));
              }
            }
          }
        }
        console.log('JS native root boundary passed');
        """ % (json.dumps(str(jail)), json.dumps(str(root / "outside.kaze")))
        proc = call(binary, "eval", root_js)
        assert proc.returncode == 0 and "JS native root boundary passed" in proc.stdout, (proc.stdout, proc.stderr)

        # CLI and compiled agents retain their documented config hierarchy.
        proc = call(binary, "script", "run", "agent.kaze", "--json")
        assert proc.returncode == 0 and json.loads(proc.stdout)["result"] == json.dumps(env_marker), (proc.stdout, proc.stderr)
        proc = call(binary, "script", "run", "agent.kaze", "positional", "--json")
        assert proc.returncode == 0 and json.loads(proc.stdout)["result"] == '"positional"', (proc.stdout, proc.stderr)
        proc = call(binary, "script", "run", "agent.kaze", "positional", "--arg", "secret=named", "--json")
        assert proc.returncode == 0 and json.loads(proc.stdout)["result"] == '"named"', (proc.stdout, proc.stderr)
        (root / "selected.json").write_text('{"args":{"secret":"selected"}}')
        proc = call(binary, "script", "run", "agent.kaze", "--config", "selected.json", "--json")
        assert proc.returncode == 0 and json.loads(proc.stdout)["result"] == json.dumps(env_marker), (proc.stdout, proc.stderr)
        for bad_file, expected in (("missing.json", "E_KAZE_IO"), ("bad.json", "E_KAZE_ARGS"), ("bad-args.json", "E_KAZE_ARGS")):
            if bad_file == "bad.json":
                (root / bad_file).write_text('{"args":')
            if bad_file == "bad-args.json":
                (root / bad_file).write_text('{"args":[]}')
            proc = call(binary, "script", "run", "agent.kaze", "--config", bad_file, "--json")
            assert proc.returncode != 0 and json.loads(proc.stdout)["code"] == expected, (proc.stdout, proc.stderr)
        proc = call(binary, "script", "run", "hooks.kaze", "--config", "bad.json", "--ticks", "1", "--json")
        assert proc.returncode != 0 and json.loads(proc.stdout)["code"] == "E_KAZE_ARGS", (proc.stdout, proc.stderr)
        built = call(binary, "compile", "agent.kaze", "-o", "agent")
        assert built.returncode == 0, (built.stdout, built.stderr)
        proc = call(root / "agent")
        assert proc.returncode == 0 and env_marker in proc.stdout, (proc.stdout, proc.stderr)
        proc = call(root / "agent", "--arg", "secret=explicit")
        assert proc.returncode == 0 and 'explicit' in proc.stdout and env_marker not in proc.stdout, (proc.stdout, proc.stderr)
        env.pop("AGENT_SECRET")
        proc = call(binary, "eval", js)
        assert proc.returncode == 0 and "JS file and argument boundaries passed" in proc.stdout, (proc.stdout, proc.stderr)
        proc = call(binary, "script", "run", "agent.kaze", "--config=selected.json", "--json")
        assert proc.returncode == 0 and json.loads(proc.stdout)["result"] == '"selected"', (proc.stdout, proc.stderr)
        proc = call(root / "agent")
        assert proc.returncode == 0 and config_marker in proc.stdout, (proc.stdout, proc.stderr)

    print("PASS: embedded Kaze file policy, native root, argument isolation and CLI precedence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
