"""CPU tests for the wave-cooperative grouped Viterbi wiring (HTR encoder track step 2).

The GPU kernels themselves are qualified by step2/parity.py (bitwise against the shipped k_group binary). These tests
pin the source contract: the shipped kernels and host code are byte-preserved, the coop path is opt-in, and the
admitted-domain guard rejects inputs on which the butterfly can differ from the serial scan.
"""
import importlib.util
import pathlib
import unittest

import torch

from htr import vit_v2

ROOT = pathlib.Path(__file__).resolve().parents[1]


def pinned():
    spec = importlib.util.spec_from_file_location("htr._pinned_vit_v2_t", ROOT / "experiments/pinned/vit_v2_bd0b92c.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SourceContract(unittest.TestCase):
    def test_shipped_kernels_and_host_are_a_byte_prefix(self):
        p = pinned()
        self.assertTrue(vit_v2._SRC.startswith(p._SRC), "k_init/k_group/run must stay byte-identical to bd0b92c")
        self.assertTrue(vit_v2._CPP.startswith(p._CPP))
        self.assertIn("run_coop", vit_v2._SRC[len(p._SRC):])

    def test_coop_code_is_rocm_guarded(self):
        tail = vit_v2._SRC[len(pinned()._SRC):]
        self.assertIn("#if defined(__HIP_PLATFORM_AMD__)", tail)
        self.assertIn("coop Viterbi is ROCm (wave32 gfx1151) only", tail)

    def test_new_extension_name_and_build_dir(self):
        import inspect
        src = inspect.getsource(vit_v2._get)
        self.assertIn('"htr_viterbi_ext3"', src)
        self.assertIn('"vit3_"', src)

    def test_default_impl_is_shipped_kernel(self):
        self.assertEqual(vit_v2.IMPL, "group")
        self.assertEqual(sorted(vit_v2.COOP_VARIANT.values()), [0, 1, 2, 3])


class Domain(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(5)
        self.x = torch.randn(4, 256, generator=g)
        self.w = 0.5 + torch.rand(4, 256, generator=g)
        self.cb = torch.randn(1 << 16, generator=g)

    def ok(self, x=None, w=None, cb=None, K=3, th=512):
        return vit_v2.coop_admissible(self.x if x is None else x, self.w if w is None else w,
                                      self.cb if cb is None else cb, K, th)

    @unittest.skipUnless(torch.version.hip, "admission is ROCm-only")
    def test_encoder_domain_admitted(self):
        for K in (2, 3, 4, 5):
            self.assertTrue(self.ok(K=K))
        huge = self.x.clone(); huge[-1, ::2] = 1.5e19       # finite input whose path costs overflow to +inf
        self.assertTrue(self.ok(x=huge))

    def test_rejections(self):
        nan_x = self.x.clone(); nan_x[0, 3] = float("nan")
        inf_cb = self.cb.clone(); inf_cb[7] = float("inf")
        zero_w = self.w.clone(); zero_w[1, 9] = 0.0
        neg_w = self.w.clone(); neg_w[2, 0] = -1.0
        inf_w = self.w.clone(); inf_w[3, 5] = float("inf")
        for kw in (dict(x=nan_x), dict(cb=inf_cb), dict(w=zero_w), dict(w=neg_w), dict(w=inf_w),
                   dict(K=1), dict(K=6), dict(K=8), dict(th=48)):
            self.assertFalse(self.ok(**kw), kw)

    def test_ug4_tail_requires_whole_wave(self):
        self.assertFalse(vit_v2.coop_geometry(2, 1, 256, 2))
        self.assertFalse(vit_v2.coop_geometry(2, 2, 256, 3))
        self.assertTrue(vit_v2.coop_geometry(2, 4, 256, 3))
        self.assertTrue(vit_v2.coop_geometry(2, 1, 256, 0))
        self.assertTrue(vit_v2.coop_geometry(3, 1, 256, 2))

    def test_unknown_impl_refused(self):
        with self.assertRaises(ValueError):
            vit_v2.viterbi(self.x, self.w, self.cb, 3, impl="fast")


if __name__ == "__main__":
    unittest.main()
