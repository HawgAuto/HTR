"""Standalone unit tests for D3 automatic gfx1151 cooperative Viterbi dispatch and backtrack optimization."""
import importlib.util
import os
import pathlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

VIT_PATH = pathlib.Path(__file__).resolve().parents[1] / "htr" / "vit_v2.py"
GROUP = {2: (4, 256), 3: (3, 512), 4: (2, 512), 5: (2, 512), 6: (1, 512), 8: (1, 512)}
EXPECTED = {
    2: ("coop", 2, 5, 512, False, 3),
    3: ("coop", 3, 3, 512, False, 1),
    4: ("coop", 4, 2, 256, False, 1),
    5: ("coop", 5, 2, 512, False, 3),
}


class _Hit(Exception):
    pass


class _Native:
    def run(self, x, w, cb, j, K, pin, r, th):
        raise _Hit(("group", K, r, th, pin))
    def run_coop(self, x, w, cb, j, K, pin, r, th, var, bplo):
        raise _Hit(("coop", K, r, th, pin, var, bplo))


class _FakeCUDA:
    def __init__(self, t, device):
        self.t = t
        self.shape = t.shape
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
    def float(self):
        return self.t.float()


def _load(env=None):
    patch = {"HTR_VITERBI_IMPL": env} if env is not None else {}
    with mock.patch.dict(os.environ, patch):
        if env is None:
            os.environ.pop("HTR_VITERBI_IMPL", None)
        spec = importlib.util.spec_from_file_location("htr._test_vit_v2", VIT_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._get = lambda: _Native()
        return mod


def _observe(m, K=4, B=256, T=256, seam="qtip", tail=True, impl=None,
             arch="gfx1151:sramecc-:xnack-", device="cuda:0", bad=None, hip=True, auto=True):
    x, w, cb = torch.zeros(B, T), torch.ones(B, T), torch.ones(1 << 16)
    if bad == "cb":
        cb[0] = float("inf")
    if bad == "x":
        x[0, 0] = float("nan")
    if bad == "w":
        w[0, 0] = 0.0
    args = (_FakeCUDA(x, device), _FakeCUDA(w, device), _FakeCUDA(cb, device), K)
    with mock.patch.object(torch.version, "hip", "6.4" if hip else None), \
         mock.patch.object(torch.cuda, "get_device_properties",
                           lambda d: SimpleNamespace(**({"gcnArchName": arch} if arch is not None else {}))):
        try:
            (m.viterbi_auto if auto else m.viterbi)(*args, tailbite=tail, seam=seam, impl=impl)
        except _Hit as h:
            return h.args[0]
    raise AssertionError("did not intercept native extension call")


class TestAutoDispatch(unittest.TestCase):
    def test_verified_widths_dispatch_on_gfx1151_both_seams(self):
        m = _load()
        for K, (impl, r, th, batches) in m._AUTO_ROUTES.items():
            for seam, bplo in (("qtip", 128), ("htr", 256)):
                for B in batches:
                    with self.subTest(K=K, seam=seam, B=B):
                        self.assertEqual(_observe(m, K=K, B=B, seam=seam), (*EXPECTED[K], bplo))

    def test_unmeasured_shapes_widths_and_arches_stay_group(self):
        m = _load()
        for kw in (
            dict(K=6, B=256), dict(K=8, B=256),
            dict(K=4, B=640), dict(K=3, B=128), dict(K=5, B=1025),
            dict(K=3, B=256, T=255), dict(K=3, B=256, seam="htr", tail=False),
            dict(K=4, B=256, arch="gfx1100"), dict(K=4, B=256, arch=None),
            dict(K=4, B=256, device="cpu"),
        ):
            with self.subTest(kw=kw):
                K = kw["K"]
                self.assertEqual(_observe(m, **kw), ("group", K, *GROUP[K], False))

    def test_invalid_inputs_restore_original_group_geometry(self):
        m = _load()
        for bad in ("x", "w", "cb"):
            self.assertEqual(_observe(m, K=4, B=256, bad=bad), ("group", 4, 2, 512, False))

    def test_env_and_explicit_overrides_bypass_auto_table(self):
        for env, arg, expected in (
            ("group", None, ("group", 4, 2, 512, False)),
            ("coop", None, ("coop", 4, 2, 512, False, 0, 128)),
            (None, "group", ("group", 4, 2, 512, False)),
            ("group", "coop_dpp", ("coop", 4, 2, 512, False, 1, 128)),
        ):
            m = _load(env)
            self.assertEqual(_observe(m, K=4, B=256, impl=arg), expected)

    def test_direct_and_cuda_paths_preserved(self):
        m = _load()
        self.assertEqual(_observe(m, K=4, B=256, auto=False), ("group", 4, m.RMAX[4], m.TH, False))
        self.assertEqual(_observe(m, K=4, B=256, hip=False), ("group", 4, 2, 256, False))


if __name__ == "__main__":
    unittest.main()
