"""Regression against an actual saved MiMo calibration covariance."""
from pathlib import Path
import unittest
import torch
from safetensors.torch import load_file
from htr import wire
from htr.scale_refit import refit_packed
from test_scale_refit_packed import packed_fixture


class CapturedRoundoffTests(unittest.TestCase):
    def test_finite_capture_roundoff_is_accepted_without_mutating_input(self):
        h = load_file(str(Path(__file__).parent/'fixtures/captured-roundoff-128.safetensors'))['hessian']
        self.assertFalse(torch.allclose(h, h.T, rtol=1e-6, atol=1e-10), 'fixture must reproduce the old rejection')
        frozen = h.clone()
        packed = packed_fixture()
        desired = torch.full((128,), 1.5, dtype=torch.float16)
        w = wire.dequant(packed['qweight'], desired, packed['su'], packed['sv'])
        for multiplier in (1., 1e-4, 1e4):
            with self.subTest(multiplier=multiplier):
                try:
                    result = refit_packed(w, h*multiplier, packed)
                except ValueError as exc:
                    self.fail('valid captured covariance rejected: '+str(exc))
                self.assertTrue(torch.equal(result['scales'], desired))
                self.assertLessEqual(result['scale_refit']['max_row_delta'], 0.)
                self.assertGreater(result['scale_refit']['input_relative_skew_fro'], 0.)
                for key in ('qweight','su','sv'):
                    self.assertTrue(torch.equal(result[key], packed[key]))
        self.assertTrue(torch.equal(h, frozen))

    def test_materially_asymmetric_input_still_rejected(self):
        packed = packed_fixture()
        w = torch.zeros(128,128)
        h = torch.eye(128)
        h[0,1] = .01
        for multiplier in (1., 1e-4, 1e4):
            with self.subTest(multiplier=multiplier), self.assertRaisesRegex(ValueError, 'symmetric'):
                refit_packed(w, h*multiplier, packed)

    def test_quadratic_is_independent_of_the_antisymmetric_part(self):
        h = load_file(str(Path(__file__).parent/'fixtures/captured-roundoff-128.safetensors'))['hessian'].double()
        g = torch.Generator().manual_seed(741)
        e = torch.randn(16,128, generator=g, dtype=torch.float64)
        symmetric = .5*h + .5*h.T
        original = ((e@h)*e).sum(dim=1)
        canonical = ((e@symmetric)*e).sum(dim=1)
        torch.testing.assert_close(original, canonical, rtol=2e-14, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
