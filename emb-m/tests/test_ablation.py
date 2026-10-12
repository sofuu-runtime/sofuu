"""Ablation harness smoke test: tiny grid, real training, fast."""

import json
import unittest

from sofuu_arch.ablate import VARIANTS, evaluate, proxy_config, summarize, train_variant
from sofuu_arch.synth import SynthSpec, eval_set, train_pairs


class TestSynthTask(unittest.TestCase):
    def test_pairs_share_motifs_and_eval_splits(self):
        spec = SynthSpec()
        q, d, m = train_pairs(spec, 16, seed=0)
        self.assertEqual(q.shape, (16, spec.seq_len))
        self.assertEqual(d.shape, (16, spec.seq_len))
        self.assertEqual(m.shape, (16,))
        eq, pool, eq_m, pool_m = eval_set(spec, 8, 32, seed=1)
        self.assertEqual(eq.shape, (8, spec.seq_len))
        self.assertEqual(pool.shape, (32, spec.seq_len))
        # Every query has at least one same-motif doc in the pool.
        for i in range(8):
            self.assertTrue((pool_m == eq_m[i]).any())


class TestAblationSmoke(unittest.TestCase):
    def test_fast_grid_runs_and_reports(self):
        spec = SynthSpec()
        results = []
        for variant in ["full", "no-moe"]:
            r = train_variant(variant, seed=99, spec=spec, steps=12,
                              batch_size=8, eval_every=6)
            self.assertIn(variant, VARIANTS)
            for key in ("recall@1", "recall@5", "mrr", "final_aux"):
                self.assertTrue(0.0 <= r[key] <= 2.0 or key == "final_aux",
                                f"{variant}.{key}={r[key]}")
                self.assertTrue(r[key] == r[key], "NaN metric")  # finite
            results.append(r)
        table = summarize(results)
        self.assertIn("full", table)
        self.assertIn("no-moe", table)
        json.dumps(results)  # report-serializable

    def test_proxy_configs_build_for_all_variants(self):
        spec = SynthSpec()
        for name, overrides in VARIANTS.items():
            cfg = proxy_config(overrides, spec)
            from sofuu_arch import SofuuEncoder, count_params
            model = SofuuEncoder(cfg)
            actual = sum(p.numel() for p in model.parameters())
            self.assertEqual(actual, count_params(cfg)["total"], name)

    def test_evaluate_restores_training_mode(self):
        # Audited bug: evaluate() left the model in eval mode, silently
        # disabling stochastic depth for the rest of training.
        import torch
        from sofuu_arch import SofuuEncoder
        spec = SynthSpec()
        cfg = proxy_config({}, spec)
        model = SofuuEncoder(cfg)
        model.train()
        eq, pool, eq_m, pool_m = eval_set(spec, 4, 16, seed=3)
        evaluate(model, eq, pool, eq_m, pool_m)
        self.assertTrue(model.training)
        model.eval()
        evaluate(model, eq, pool, eq_m, pool_m)
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
