"""CPU regression: E08 layer caches may use checkpoint rather than runtime module prefixes."""
import tempfile
import unittest
from pathlib import Path

import torch
from htr.quantize import LayerDirHessians


class LayerCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.runtime = "model.layers.0.linear_attn.in_proj_qkv"
        self.checkpoint = "model.language_model.layers.0.linear_attn.in_proj_qkv"
        self.H = torch.arange(16, dtype=torch.float64).reshape(4, 4)

    def store(self, tensors, layer=0):
        torch.save(tensors, self.d / f"layer{layer:03d}.pt")
        return LayerDirHessians(str(self.d), "cpu")

    def test_exact_name_preserves_values_and_dtype(self):
        h = self.store({self.runtime: self.H})
        self.assertIn(self.runtime, h)
        self.assertTrue(torch.equal(h[self.runtime], self.H))
        self.assertEqual(h[self.runtime].dtype, self.H.dtype)

    def test_checkpoint_prefix_resolves_runtime_name(self):
        h = self.store({self.checkpoint: self.H})
        self.assertIn(self.runtime, h)
        try:
            got = h[self.runtime]
        except KeyError:
            self.fail("checkpoint-prefix Hessian was not resolved to its unique runtime module")
        self.assertTrue(torch.equal(got, self.H))
        self.assertEqual(got.dtype, self.H.dtype)
        self.assertEqual(got.shape, self.H.shape)

    def test_missing_module_is_not_present_merely_because_layer_exists(self):
        h = self.store({self.runtime: self.H})
        absent = "model.layers.0.linear_attn.out_proj"
        self.assertNotIn(absent, h)
        with self.assertRaises(KeyError):
            h[absent]

    def test_missing_layer_is_not_present(self):
        h = self.store({self.runtime: self.H})
        absent = self.runtime.replace("layers.0.", "layers.1.")
        self.assertNotIn(absent, h)
        with self.assertRaises(KeyError):
            h[absent]

    def test_ambiguous_canonical_names_fail_closed(self):
        h = self.store({self.runtime: self.H, self.checkpoint: self.H + 1})
        with self.assertRaises(ValueError):
            _ = self.runtime in h
        with self.assertRaises(ValueError):
            h[self.runtime]

    def test_layer_switch_retains_correct_values(self):
        self.store({self.checkpoint: self.H})
        other = self.checkpoint.replace("layers.0.", "layers.1.")
        h = self.store({other: self.H + 100}, layer=1)
        for n, want in ((self.runtime, self.H), (self.runtime.replace("layers.0.", "layers.1."), self.H + 100),
                        (self.runtime, self.H)):
            try:
                got = h[n]
            except KeyError:
                self.fail("layer cache did not resolve a unique canonical module")
            self.assertTrue(torch.equal(got, want))

    def test_weight_suffix_is_canonicalized(self):
        h = self.store({self.checkpoint + ".weight": self.H})
        try:
            got = h[self.runtime]
        except KeyError:
            self.fail("existing canon() weight suffix contract was not applied")
        self.assertTrue(torch.equal(got, self.H))


if __name__ == "__main__":
    unittest.main()
