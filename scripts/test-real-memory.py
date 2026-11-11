#!/usr/bin/env python3
"""Exercise a real temporary QTSQ store across independent processes."""
import os
import hashlib
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent.parent
binary = Path(os.environ.get("SOFUU_BIN", root / "sofuu")).resolve()
if not binary.is_file() or not os.access(binary, os.X_OK):
    print("BLOCKED: set SOFUU_BIN to a built Sofuu executable", file=sys.stderr)
    raise SystemExit(2)
if os.geteuid() == 0:
    print("BLOCKED: the filesystem denial check requires a non-root user", file=sys.stderr)
    raise SystemExit(2)
print(f"Artifact: {binary}, sha256={hashlib.sha256(binary.read_bytes()).hexdigest()}", flush=True)

with tempfile.TemporaryDirectory(prefix="sofuu-real-memory-") as directory:
    temp = Path(directory)
    fail_dir = temp / "failure"
    fail_dir.mkdir()
    marker = "memory-" + secrets.token_hex(12)
    env = os.environ.copy()
    env["SOFUU_HOME"] = str(temp / "home")
    script = root / "scripts/contract/harness/real_memory_store.js"
    phases = [
        ("write", temp / "primary.qtsq"),
        ("read", temp / "primary.qtsq"),
        ("empty", temp / "separate.qtsq"),
        ("fail-flush", fail_dir / "failure.qtsq"),
    ]
    try:
        for phase, path in phases:
            result = subprocess.run(
                [str(binary), "run", str(script), phase, str(path), marker, str(fail_dir)],
                cwd=temp, env=env, timeout=30,
            )
            if result.returncode:
                raise SystemExit(result.returncode)
    finally:
        # Restore traversal permission even if the test process was interrupted.
        fail_dir.chmod(0o700)
    print("PASS: real memory write, separate-process reopen, store isolation, readonly and flush failure")
