"""Fixed-path scale refit contract. CPU-only; no models/GPU/network."""
import importlib
import importlib.util
import unittest

import torch

torch.set_num_threads(2)


def implementation(case):
    # Outside assertRaises: missing implementation must not satisfy negatives.
    case.assertIsNotNone(importlib.util.find_spec('htr.scale_refit'),
                         'missing opt-in fixed-path scale-refit implementation')
    return importlib.import_module('htr.scale_refit').refit_row_scales


def errors(w, c, h, scales):
    # Independent, actual FP32 wire-product objective, accumulated in FP64.
    e = w.double() - (c.float() * scales.float()[:, None]).double()
    return torch.einsum('bi,ij,bj->b', e, h.double(), e)


class RowScaleRefitTests(unittest.TestCase):
    def test_known_gain_with_nondiagonal_hessian(self):
        fn = implementation(self)
        c = torch.tensor([[1., -2., 0.5], [-1., 0.25, 2.]])
        w = c * torch.tensor([1.25, 0.5])[:, None]
        h = torch.tensor([[2., 0.5, 0.], [0.5, 2., 0.25], [0., 0.25, 1.]])
        old = torch.tensor([0.75, 0.75], dtype=torch.float16)
        r = fn(w, c, h, old)
        self.assertTrue(torch.equal(r['scales'], torch.tensor([1.25, .5], dtype=torch.float16)))
        self.assertTrue(torch.equal(r['before'], errors(w, c, h, old)))
        self.assertTrue(torch.equal(r['after'], errors(w, c, h, r['scales'])))
        self.assertTrue((r['after'] <= r['before']).all())

    def test_fp16_neighbors_against_exhaustive_scale_grid(self):
        fn = implementation(self)
        # All nonnegative finite IEEE binary16 values, not a float32 grid.
        grid = torch.arange(0x7c00, dtype=torch.int16).view(torch.float16)
        c = torch.tensor([[.0312345, -1.312345, .444444],
                          [3.141592, .271828, -.111111]], dtype=torch.float32)
        w = c * torch.tensor([.10003, 4.0019])[:, None]
        h = torch.tensor([[1., .1, .2], [.1, 2., .3], [.2, .3, 1.]])
        old = torch.tensor([.125, 3.5], dtype=torch.float16)
        r = fn(w, c, h, old)
        for i in range(2):
            e = w[i].double()[None] - (c[i][None] * grid.float()[:, None]).double()
            scores = torch.einsum('bi,ij,bj->b', e, h.double(), e)
            self.assertEqual(float(r['after'][i]), float(scores.min()))

    def test_zero_curvature_and_zero_code_keep_incumbent(self):
        fn = implementation(self)
        w = torch.ones(2, 3)
        c = torch.tensor([[0., 0., 0.], [1., 2., 3.]])
        old = torch.tensor([0., 2.], dtype=torch.float16)
        for h in (torch.zeros(3, 3), torch.eye(3)):
            r = fn(w, c, h, old)
            self.assertEqual(r['scales'][0], old[0])
            if not h.any():
                self.assertTrue(torch.equal(r['scales'], old))

    def test_saturation_and_nonnegative_constraint(self):
        fn = implementation(self)
        c = torch.ones(2, 2)
        w = torch.tensor([[100000., 100000.], [-1., -1.]])
        old = torch.ones(2, dtype=torch.float16)
        r = fn(w, c, torch.eye(2), old)
        self.assertTrue(torch.equal(r['scales'], torch.tensor([65504., 0.], dtype=torch.float16)))
        self.assertTrue(torch.isfinite(r['scales']).all())
        self.assertTrue((r['after'] <= r['before']).all())

    def test_inputs_immutable_batch_independent_and_never_worse(self):
        fn = implementation(self)
        g = torch.Generator().manual_seed(431)
        c = torch.randn(13, 9, generator=g)
        w = torch.randn(13, 9, generator=g)
        a = torch.randn(9, 9, generator=g)
        h = a @ a.T + torch.eye(9)*.1
        old = torch.linspace(.01, 1., 13).half()
        inputs = [w, c, h, old]
        copies = [x.clone() for x in inputs]
        full = fn(*inputs, row_batch=13)
        chunked = fn(*inputs, row_batch=1)
        self.assertTrue(torch.equal(full['scales'], chunked['scales']))
        self.assertTrue(torch.allclose(full['before'], errors(w, c, h, old), rtol=1e-12, atol=1e-12))
        self.assertTrue(torch.allclose(full['after'], errors(w, c, h, full['scales']), rtol=1e-12, atol=1e-12))
        self.assertTrue((full['after'] <= full['before']).all())
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(inputs, copies)))

    def test_invalid_inputs_rejected_before_computation(self):
        fn = implementation(self)
        w, c, h, s = torch.ones(2, 3), torch.ones(2, 3), torch.eye(3), torch.ones(2).half()
        bad = [
            (w, c[:, :2], h, s),
            (w, c, h[:2], s),
            (w, c, h, s.float()),
            (w, c, h, torch.tensor([-1., 1.]).half()),
            (w, c, h, torch.tensor([float('nan'), 1.]).half()),
            (w*float('inf'), c, h, s),
            (w, c*float('nan'), h, s),
            (w, c, h*float('nan'), s),
            (w[:0], c[:0], h, s[:0]),
            (w[:, :0], c[:, :0], h[:0, :0], s),
            (w, c, torch.tensor([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]]), s),
        ]
        for args in bad:
            with self.subTest(shapes=[tuple(x.shape) for x in args]):
                with self.assertRaises((ValueError, TypeError)):
                    fn(*args)
        for batch in (0, -1, 1.5, True):
            with self.subTest(row_batch=batch):
                with self.assertRaises((ValueError, TypeError)):
                    fn(w, c, h, s, row_batch=batch)


if __name__ == '__main__':
    unittest.main()
