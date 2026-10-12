"""Architecture tests: budgets, shapes, masks, MoE, kernel parity."""

import unittest

import torch

from sofuu_arch import (
    ModelConfig,
    SofuuEncoder,
    SparseMoEFFN,
    count_active_params,
    count_params,
    get_preset,
)
from sofuu_arch import PRESETS
from sofuu_arch.block import DisentangledAttention


def run(model, ids, mask=None):
    vec, aux, z = model(ids, mask)
    return vec, aux, z


class TestParamCounts(unittest.TestCase):
    def test_counter_matches_torch(self):
        for name in PRESETS:
            cfg = get_preset(name)
            model = SofuuEncoder(cfg)
            actual = sum(p.numel() for p in model.parameters())
            expected = count_params(cfg)["total"]
            self.assertEqual(actual, expected, f"{name}: torch={actual} counter={expected}")

    def test_within_budget(self):
        for name in PRESETS:
            cfg = get_preset(name)
            total = count_params(cfg)["total"]
            self.assertLessEqual(total, cfg.param_budget, name)
        # Exact pinned totals (change only with a deliberate spec revision).
        self.assertEqual(count_params(get_preset("s-code-20m"))["total"], 18_731_008)
        self.assertEqual(count_params(get_preset("s-docs-15m"))["total"], 14_569_472)
        self.assertEqual(count_params(get_preset("s-mem-10m"))["total"], 9_549_312)

    def test_moe_costs_no_extra_trunk_params(self):
        # Experts are sized so MoE layers cost what dense layers cost, plus
        # router + qk-norm only.
        cfg = get_preset("s-code-20m")
        c = count_params(cfg)
        self.assertEqual(c["per_moe_layer"]["ffn"], c["per_layer"]["ffn"])
        self.assertEqual(c["n_moe"], 6)
        self.assertEqual(c["n_dense"], 6)

    def test_active_params_below_total(self):
        for name in PRESETS:
            cfg = get_preset(name)
            active = count_active_params(cfg)
            total = count_params(cfg)["total"]
            self.assertLess(active, total, name)
        # s-code: 6 MoE layers each activate 1/8 of the FFN.
        cfg = get_preset("s-code-20m")
        dense_equiv = count_params(cfg)["total"]
        self.assertLess(count_active_params(cfg), dense_equiv - 2_000_000)

    def test_dense_pattern_has_no_moe(self):
        cfg = ModelConfig(name="dense-test", moe_pattern="none", num_layers=4)
        model = SofuuEncoder(cfg)
        n_moe = sum(1 for b in model.blocks if not hasattr(b.ffn, "experts"))
        self.assertEqual(n_moe, 4)  # all blocks dense
        self.assertEqual(count_params(cfg)["n_moe"], 0)

    def test_invalid_configs_rejected(self):
        with self.assertRaises(ValueError):
            ModelConfig(name="bad", hidden_dim=250)  # not a multiple of 64
        with self.assertRaises(ValueError):
            ModelConfig(name="bad", hidden_dim=256, num_heads=3)  # does not divide
        with self.assertRaises(ValueError):
            ModelConfig(name="bad", ffn_dim=700)  # not divisible by num_experts
        with self.assertRaises(ValueError):
            ModelConfig(name="bad", attn_impl="flash")  # unknown kernel


class TestEncoderBehavior(unittest.TestCase):
    def test_output_shape_and_unit_norm(self):
        torch.manual_seed(0)
        cfg = get_preset("s-mem-10m")
        model = SofuuEncoder(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        out, aux, z = run(model, ids)
        self.assertEqual(out.shape, (2, cfg.output_dim))
        self.assertTrue(torch.allclose(out.norm(p=2, dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.isfinite(aux) and torch.isfinite(z))
        self.assertGreaterEqual(aux.item(), 1.0 - 1e-5)  # balance loss floor.

    def test_padding_does_not_leak(self):
        torch.manual_seed(1)
        cfg = get_preset("s-mem-10m")
        model = SofuuEncoder(cfg).eval()
        real = torch.randint(0, cfg.vocab_size, (1, 10))
        mask = torch.ones(1, 10, dtype=torch.bool)
        out_clean, _, _ = run(model, real, mask)
        junk = torch.randint(0, cfg.vocab_size, (1, 6))
        padded = torch.cat([real, junk], dim=1)
        pad_mask = torch.cat([mask, torch.zeros(1, 6, dtype=torch.bool)], dim=1)
        out_padded, _, _ = run(model, padded, pad_mask)
        self.assertTrue(torch.allclose(out_clean, out_padded, atol=1e-5))

    def test_deterministic_eval(self):
        torch.manual_seed(2)
        cfg = get_preset("s-mem-10m")
        model = SofuuEncoder(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 12))
        a, _, _ = run(model, ids)
        b, _, _ = run(model, ids)
        self.assertTrue(torch.equal(a, b))

    def test_matryoshka_prefixes_unit_norm(self):
        torch.manual_seed(3)
        cfg = get_preset("s-mem-10m")
        model = SofuuEncoder(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        vec, _, _ = run(model, ids)
        for dim in cfg.matryoshka_dims:
            sub = model.matryoshka(vec, dim)
            self.assertEqual(sub.shape, (2, dim))
            self.assertTrue(torch.allclose(sub.norm(p=2, dim=-1), torch.ones(2), atol=1e-5))

    def test_local_window_masks_distant_tokens(self):
        attn = DisentangledAttention(256, 4, 32, 256).eval()
        mask = attn._build_mask(1, 300, torch.device("cpu"), None, False, 128)
        neg = torch.finfo(torch.float32).min
        self.assertEqual(mask[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(mask[0, 0, 0, 128].item(), 0.0)
        self.assertEqual(mask[0, 0, 0, 129].item(), neg)
        self.assertEqual(mask[0, 0, 0, 299].item(), neg)


class TestMoE(unittest.TestCase):
    def test_padding_gets_zero_delta(self):
        torch.manual_seed(5)
        moe = SparseMoEFFN(256, 32, num_experts=4).eval()
        x = torch.randn(1, 8, 256)
        mask = torch.tensor([[True] * 5 + [False] * 3])
        out, _, _ = moe(x, mask)
        self.assertTrue(torch.equal(out[:, 5:, :], torch.zeros(1, 3, 256)))

    def test_overflow_tokens_pass_through(self):
        torch.manual_seed(6)
        # 8 tokens, 2 experts, capacity 1 each -> at most 2 served.
        moe = SparseMoEFFN(256, 32, num_experts=2, capacity_factor=0.25).eval()
        x = torch.randn(1, 8, 256)
        out, aux, z = moe(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(aux) and torch.isfinite(z))

    def test_routing_covers_all_tokens_or_skips_cleanly(self):
        torch.manual_seed(7)
        moe = SparseMoEFFN(256, 32, num_experts=8).eval()
        x = torch.randn(2, 16, 256)
        out, aux, z = moe(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())


class TestKernelParity(unittest.TestCase):
    def test_sdpa_matches_eager(self):
        torch.manual_seed(8)
        cfg = get_preset("s-mem-10m")
        eager_cfg = ModelConfig(
            name="eager", embed_style=cfg.embed_style, num_layers=4,
            ffn_dim=512, attn_impl="eager", moe_pattern="none",
            qk_norm=True, dropout=0.0,
        )
        sdpa_cfg = ModelConfig(
            name="sdpa", embed_style=cfg.embed_style, num_layers=4,
            ffn_dim=512, attn_impl="sdpa", moe_pattern="none",
            qk_norm=True, dropout=0.0,
        )
        eager = SofuuEncoder(eager_cfg).eval()
        sdpa = SofuuEncoder(sdpa_cfg).eval()
        sdpa.load_state_dict(eager.state_dict())
        ids = torch.randint(0, cfg.vocab_size, (2, 24))
        mask = torch.ones(2, 24, dtype=torch.bool)
        mask[0, 20:] = False
        a, _, _ = run(eager, ids, mask)
        b, _, _ = run(sdpa, ids, mask)
        self.assertTrue(torch.allclose(a, b, atol=1e-5),
                        f"max diff {(a - b).abs().max().item()}")


if __name__ == "__main__":
    unittest.main()
