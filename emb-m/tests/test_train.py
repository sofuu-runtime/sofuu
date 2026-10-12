"""Training-path tests: losses, checkpointing, end-to-end CPU smoke."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from sofuu_arch import ModelConfig, SofuuEncoder, count_params
from sofuu_arch import data as data_mod
from sofuu_arch.losses import info_nce_gathered, matryoshka_loss
from sofuu_arch.train import evaluate_retrieval, linear_warmup_decay, save_checkpoint


class TestLRSchedule(unittest.TestCase):
    def test_fractions_and_decay_shape(self):
        self.assertAlmostEqual(linear_warmup_decay(0, 500, 20000), 1 / 500)
        self.assertEqual(linear_warmup_decay(499, 500, 20000), 1.0)
        self.assertEqual(linear_warmup_decay(20000, 500, 20000), 0.0)
        prev = 2.0
        for t in range(500, 20000):
            v = linear_warmup_decay(t, 500, 20000)
            self.assertLessEqual(v, prev)  # monotone non-increasing decay.
            self.assertGreaterEqual(v, 0.0)
            prev = v

    def test_effective_lr_never_squared(self):
        # The audited bug: schedule returned absolute lrs, so LambdaLR
        # applied base_lr twice (peak 4e-8 instead of 2e-4). Pin the
        # end-to-end effective rate through a real optimizer + scheduler.
        base = 2e-4
        opt = torch.optim.SGD([torch.zeros(1)], lr=base)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda t: linear_warmup_decay(t, 500, 20000))
        peak_seen = 0.0
        for _ in range(20000):
            opt.step()
            sched.step()
            peak_seen = max(peak_seen, sched.get_last_lr()[0])
        self.assertAlmostEqual(peak_seen, base, places=12)
        self.assertGreater(peak_seen, base / 2)  # sane magnitude, not lr^2.


class TestLosses(unittest.TestCase):
    def test_info_nce_perfect_match_low_loss(self):
        torch.manual_seed(0)
        B, d = 8, 32
        q = torch.nn.functional.normalize(torch.randn(B, d), dim=-1)
        loss, n_negs = info_nce_gathered(q, q.clone(), None, tau=0.05)
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(loss.item(), 0.5)  # identical pairs separate easily.
        self.assertEqual(n_negs, 0)  # single process: no gathered extras.

    def test_info_nce_hard_negatives_expand_pool(self):
        torch.manual_seed(1)
        B, H, d = 4, 3, 16
        q = torch.nn.functional.normalize(torch.randn(B, d), dim=-1)
        p = torch.nn.functional.normalize(torch.randn(B, d), dim=-1)
        h = torch.nn.functional.normalize(torch.randn(B, H, d), dim=-1)
        loss, n_negs = info_nce_gathered(q, p, h, tau=0.05)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(n_negs, B * H)  # all hard negs join the pool.

    def test_matryoshka_weights(self):
        torch.manual_seed(2)
        B, d = 6, 64
        q = torch.nn.functional.normalize(torch.randn(B, d), dim=-1)
        p = torch.nn.functional.normalize(torch.randn(B, d), dim=-1)
        full, _ = matryoshka_loss(q, p, None, (64,), (1.0,), 0.05)
        weighted, _ = matryoshka_loss(q, p, None, (64, 32), (1.0, 0.5), 0.05)
        single, _ = info_nce_gathered(q, p, None, 0.05)
        self.assertAlmostEqual(full.item(), single.item(), places=5)
        self.assertGreater(weighted.item(), 0.0)
        self.assertTrue(torch.isfinite(weighted))


class TestCheckpoint(unittest.TestCase):
    def test_save_load_roundtrip(self):
        torch.manual_seed(3)
        cfg = ModelConfig(name="ckpt-test", hidden_dim=64, num_heads=2,
                          num_layers=2, ffn_dim=128, output_dim=64,
                          matryoshka_dims=(64,), embed_style="direct",
                          vocab_size=512, moe_pattern="none")
        m1, m2 = SofuuEncoder(cfg), SofuuEncoder(cfg)
        opt = torch.optim.AdamW(m1.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ckpt.pt"
            save_checkpoint(p, 7, 5, m1, opt, sched, None, 0.42,
                            torch.get_rng_state(), None)
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m2.load_state_dict(ck["model"])
            self.assertEqual(ck["step"], 7)
            self.assertEqual(ck["opt_step"], 5)
            self.assertAlmostEqual(ck["best"], 0.42)
        for a, b in zip(m1.parameters(), m2.parameters()):
            self.assertTrue(torch.equal(a, b))


class TestData(unittest.TestCase):
    def test_pair_dataset_and_collate(self):
        recs = [
            {"query": [1, 2, 3], "positive": [4, 5],
             "hard_negatives": [[6], [7, 8]]},
            {"query": [9], "positive": [10]},
        ]
        ds = data_mod.PairDataset(recs, max_seq_len=8)
        batch = data_mod.collate_pairs([ds[0], ds[1]])
        q_ids, q_mask, d_ids, d_mask = batch
        self.assertEqual(q_ids.shape, (2, 8))
        self.assertEqual(d_ids.shape, (2, 3, 8))  # H padded to batch max (3).
        self.assertTrue(q_mask[0, :3].all() and not q_mask[0, 3:].any())
        # Padded hard-negative slots are fully masked.
        self.assertFalse(d_mask[1, 1:].any())

    def test_load_jsonl_rejects_bad_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.jsonl"
            p.write_text('{"query": [1]}\n')
            with self.assertRaises(ValueError):
                data_mod.load_jsonl(str(p))


class TestTrainSmoke(unittest.TestCase):
    def test_smoke_train_saves_and_resumes(self):
        from sofuu_arch import train as train_mod
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["train", "--smoke", "--preset", "s-mem-10m",
                    "--steps", "6", "--micro-batch", "4", "--accum", "1",
                    "--max-seq-len", "32", "--workers", "0",
                    "--out", tmp, "--save-every", "2", "--eval-every", "1000"]
            old = sys.argv
            sys.argv = argv
            try:
                train_mod.main()
            finally:
                sys.argv = old
            self.assertTrue((Path(tmp) / "last.pt").exists())
            self.assertTrue((Path(tmp) / "train.log.jsonl").exists())
            # Resume continues past the saved step.
            sys.argv = argv + ["--resume", str(Path(tmp) / "last.pt"),
                               "--steps", "8"]
            try:
                train_mod.main()
            finally:
                sys.argv = old
            ck = torch.load(Path(tmp) / "last.pt", map_location="cpu",
                            weights_only=False)
            self.assertGreaterEqual(ck["step"], 8)

    def test_module_entrypoint_trains(self):
        # Audited bug: train.py had no __main__ guard, so the documented
        # `torchrun -m sofuu_arch.train` exited 0 doing nothing. Pin the
        # real entrypoint: subprocess must print step logs.
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-m", "sofuu_arch.train", "--smoke",
                 "--preset", "s-mem-10m", "--steps", "21",
                 "--micro-batch", "4", "--accum", "1",
                 "--max-seq-len", "16", "--workers", "0",
                 "--out", tmp, "--save-every", "1000",
                 "--eval-every", "1000"],
                capture_output=True, text=True, cwd=Path(__file__).parent.parent,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            step_lines = [l for l in proc.stdout.splitlines()
                          if l.startswith("step ")]
            self.assertGreaterEqual(len(step_lines), 2)  # opt_steps 1 and 20.
            self.assertIn("loss", step_lines[0])

    def test_evaluate_retrieval_metrics(self):
        torch.manual_seed(4)
        cfg = ModelConfig(name="eval-test", hidden_dim=64, num_heads=2,
                          num_layers=2, ffn_dim=128, output_dim=64,
                          matryoshka_dims=(64,), embed_style="direct",
                          vocab_size=64, moe_pattern="none")
        model = SofuuEncoder(cfg).eval()
        args = type("A", (), {"max_seq_len": 16})()
        eval_obj = {
            "queries": [[1, 2, 3], [4, 5]],
            "pool": [[1, 2, 3], [4, 5], [6, 7]],
            "positives": [[0], [1]],
        }
        m = evaluate_retrieval(model, eval_obj, cfg, torch.device("cpu"), args)
        for k in ("recall@1", "recall@5", "mrr"):
            self.assertTrue(0.0 <= m[k] <= 1.0)


if __name__ == "__main__":
    unittest.main()
