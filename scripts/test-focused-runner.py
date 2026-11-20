#!/usr/bin/env python3
"""Regression checks for focused test selection using the real runners."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent


class FocusedRunnerTests(unittest.TestCase):
    def run_command(self, *command, extra_env=None):
        env = os.environ.copy()
        env.pop("ALLOW_BLOCKED", None)
        env.pop("SOFUU_CONTRACT_CASE", None)
        env.pop("SOFUU_CONTRACT_CASES", None)
        env.pop("SOFUU_CONTRACT_MODEL", None)
        env.pop("SOFUU_RUST_FILTER", None)
        env["SOFUU_BIN"] = str(ROOT / "sofuu")
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            command, cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
        )

    def test_unknown_area_is_rejected(self):
        result = self.run_command("bash", "scripts/test-area.sh", "not-an-area")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("usage:", result.stdout)

    def test_unmatched_rust_filter_is_blocked(self):
        result = self.run_command(
            "bash", "scripts/preflight.sh", "rust",
            extra_env={"SOFUU_RUST_FILTER": "nonexistent_test_filter_8b3e3e"},
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("no passing tests", result.stdout)
        self.assertIn("NOT GREEN", result.stdout)

    def test_exact_contract_case(self):
        result = self.run_command(
            "bash", "scripts/contract/run_contract_tests.sh",
            extra_env={"SOFUU_CONTRACT_CASE": "T1.1"},
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("summary: 1 cases", result.stdout)
        self.assertIn("full suite not run", result.stdout)

    def test_unknown_contract_case_is_rejected(self):
        result = self.run_command(
            "bash", "scripts/contract/run_contract_tests.sh",
            extra_env={"SOFUU_CONTRACT_CASE": "T1.nonexistent"},
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("matched 0 rows", result.stdout)

    def test_empty_contract_table_is_not_green(self):
        with tempfile.TemporaryDirectory(prefix="sofuu-test-selection-") as directory:
            table = Path(directory) / "cases.tsv"
            table.write_text("# no cases\n", encoding="utf-8")
            result = self.run_command(
                "bash", "scripts/contract/run_contract_tests.sh",
                extra_env={"SOFUU_CONTRACT_CASES": str(table)},
            )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("no cases were executed", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
