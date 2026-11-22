"""CPU tests for the opt-in QTIP tail-biting seam (rounding-only; HTR-W1 wire and decoders unchanged).

Oracle: tests/oracle_refvit.py is a byte-identical copy of the codebook-screen numpy reference
(htr-codebook-screen-20261121/refvit.py, sha256 6bce76f5...), whose two seam rules were proven pinned-optimal by
brute force on small trellises and matched the gfx1151 kernel in three GPU attempts.
"""
import importlib.util
import pathlib
import subprocess
import sys
import types
import unittest

import numpy as np
import torch

from htr import encode, reference, wire

torch.set_num_threads(2)
ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = "bd0b92c4f748f696c525932d2db0a75cd8e409f6"
sys.path.insert(0, str(ROOT / "tests"))
import oracle_refvit  # noqa: E402


def pinned_module(relpath, name):
    """Load the pinned bd0b92c version of a package module (its relative imports resolve to the live package)."""
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{PIN}:{relpath}"], check=True,
                         capture_output=True, text=True).stdout
    mod = types.ModuleType(name)
    mod.__package__ = "htr"
    exec(compile(src, f"{PIN}:{relpath}", "exec"), mod.__dict__)
    return mod


def cycle_ok(st, K):
    Sj = (1 << 16) >> K
    prev = torch.roll(st, 1, dims=1)
    return bool(((st >> K) == (prev & (Sj - 1))).all())


def fixture(B, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((B, 256), generator=g)
    w = 0.5 + torch.rand((B, 256), generator=g)
    return x, w


class ReferenceSeamTests(unittest.TestCase):
    def test_default_reference_is_bitwise_the_pinned_reference(self):
        old = pinned_module("htr/reference.py", "htr._pinned_reference")
        cb = wire.codebook_f32()
        for K in (2, 3):
            x, w = fixture(3, K)
            self.assertTrue(torch.equal(reference.viterbi(x, w, cb, K), old.viterbi(x, w, cb, K)))
            self.assertTrue(torch.equal(reference.viterbi(x, w, cb, K, seam="htr"), old.viterbi(x, w, cb, K)))

    def test_qtip_seam_matches_the_bruteforce_proven_oracle(self):
        cb = wire.codebook_f32()
        cbn = cb.numpy()
        for K in (2, 3, 4):
            x, w = fixture(3, 10 + K)
            st = reference.viterbi(x, w, cb, K, seam="qtip")
            self.assertTrue(cycle_ok(st, K))
            ref = oracle_refvit.viterbi(x.numpy(), w.numpy(), cbn, K, seam="qtip")
            c = (w.double() * (x.double() - cb.double()[st]) ** 2).sum(1)
            c_ref = (w.double() * (x.double() - torch.from_numpy(cbn).double()[torch.from_numpy(ref)]) ** 2).sum(1)
            # equal-cost paths may differ under float reordering; a seam bug moves cost by percent
            self.assertTrue(torch.allclose(c, c_ref, rtol=1e-6, atol=0), (K, c, c_ref))

    def test_qtip_differs_from_htr_somewhere(self):
        cb = wire.codebook_f32()
        x, w = fixture(8, 99)
        a = reference.viterbi(x, w, cb, 3, seam="htr")
        b = reference.viterbi(x, w, cb, 3, seam="qtip")
        self.assertFalse(torch.equal(a, b), "fixture too weak to distinguish the seam rules")

    def test_unknown_seam_rejected(self):
        x, w = fixture(1, 1)
        with self.assertRaises(ValueError):
            reference.viterbi(x, w, wire.codebook_f32(), 3, seam="bogus")
        with self.assertRaises(ValueError):
            reference.viterbi(x, w, wire.codebook_f32(), 3, tailbite=False, seam="qtip")


class EncoderSeamTests(unittest.TestCase):
    def jobs(self, seed):
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(128, 256, generator=g) * 0.02
        A = torch.randn(512, 256, generator=g)
        H = A.T @ A / 512
        return [dict(W=W, H=H), dict(W=W[:, :].flip(0).contiguous() * 1.5, H=H)]

    def run_encode(self, mod, **kw):
        old = encode.VITERBI
        try:
            encode.VITERBI = None
            mod.VITERBI = None
            return mod.encode_group(self.jobs(5), 3, max_batch=8, **kw)
        finally:
            encode.VITERBI = old

    def test_default_encode_is_bitwise_the_pinned_encoder(self):
        pinned = pinned_module("htr/encode.py", "htr._pinned_encode")
        a = self.run_encode(pinned)
        for kw in ({}, {"seam": "htr"}):
            b = self.run_encode(encode, **kw)
            for ra, rb in zip(a, b):
                for k in ("qweight", "scales", "su", "sv", "What", "Q"):
                    self.assertTrue(torch.equal(ra[k], rb[k]), (kw, k))

    def test_qtip_seam_changes_codes_only(self):
        a = self.run_encode(encode)
        b = self.run_encode(encode, seam="qtip")
        changed = False
        for ra, rb in zip(a, b):
            for k in ("scales", "su", "sv"):
                self.assertTrue(torch.equal(ra[k], rb[k]), k)
            self.assertEqual(ra["qweight"].shape, rb["qweight"].shape)
            self.assertEqual(ra["qweight"].dtype, rb["qweight"].dtype)
            st = wire.unpack_states(rb["qweight"].reshape(-1, 8 * 3), 3)
            self.assertTrue(cycle_ok(st, 3))
            # packed qtip codes decode through the unchanged W1 decoder to exactly the encoder's What
            self.assertTrue(torch.equal(wire.dequant(rb["qweight"], rb["scales"], rb["su"], rb["sv"]), rb["What"]))
            changed |= not torch.equal(ra["qweight"], rb["qweight"])
        self.assertTrue(changed, "fixture too weak to distinguish the seam rules")

    def test_unknown_seam_rejected(self):
        with self.assertRaises(ValueError):
            self.run_encode(encode, seam="bogus")

    def test_seam_is_forwarded_to_the_viterbi_callable(self):
        seen = []

        def fake(x, w, cb, K, seam="htr"):
            seen.append(seam)
            return reference.viterbi(x, w, cb, K, seam=seam)

        old = encode.VITERBI
        try:
            encode.VITERBI = fake
            encode.encode_group(self.jobs(6)[:1], 3, max_batch=8, seam="qtip")
            encode.VITERBI = lambda x, w, cb, K: reference.viterbi(x, w, cb, K)   # legacy 4-argument callable
            encode.encode_group(self.jobs(6)[:1], 3, max_batch=8)                # default must not pass seam
            with self.assertRaises(ValueError):
                encode.encode_group(self.jobs(6)[:1], 3, max_batch=8, seam="qtip")
        finally:
            encode.VITERBI = old
        self.assertEqual(set(seen), {"qtip"})


class GpuViterbiSignatureTests(unittest.TestCase):
    def test_vit_v2_exposes_seam_without_building_the_extension(self):
        import inspect
        spec = importlib.util.find_spec("htr.vit_v2")
        self.assertIsNotNone(spec)
        from htr import vit_v2
        for fn in (vit_v2.viterbi, vit_v2.viterbi_auto):
            p = inspect.signature(fn).parameters
            self.assertIn("seam", p)
            self.assertEqual(p["seam"].default, "htr")


if __name__ == "__main__":
    unittest.main()
