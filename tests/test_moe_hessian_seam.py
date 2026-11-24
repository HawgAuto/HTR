"""Unit tests for Item 5 softp/shrink router Hessians and opt-in --seam in htr.moe_encode."""
import unittest
import torch
import torch.nn.functional as F

from htr import moe_encode as me


class TestMoeHessianModes(unittest.TestCase):
    def test_soft_hessians_matches_full_batch_fp32_exact(self):
        g = torch.Generator().manual_seed(42)
        N, D, H, E = 64, 32, 16, 8
        X = torch.randn(N, D, generator=g, dtype=torch.bfloat16)
        W_gate = torch.randn(E, D, generator=g, dtype=torch.bfloat16)
        G = torch.randn(H, D, generator=g, dtype=torch.bfloat16)
        U = torch.randn(H, D, generator=g, dtype=torch.bfloat16)

        P = me.router_probs(X, W_gate, batch=17, dev=torch.device("cpu"))
        P_ref = F.softmax(X.float() @ W_gate.float().T, dim=-1)
        self.assertTrue(torch.allclose(P, P_ref, atol=1e-6))

        Hin_s, Hdn_s = me.soft_hessians(X, P[:, 3], G, U, batch=19)
        w = P[:, 3]
        denom = w.sum().clamp_min(1.0)
        Xf = X.float()
        Hin_ref = (Xf * w[:, None]).T @ Xf
        Hin_ref = (0.5 * (Hin_ref + Hin_ref.T)) / denom
        A = F.silu(X @ G.T) * (X @ U.T)
        Af = A.float()
        Hdn_ref = (Af * w[:, None]).T @ Af
        Hdn_ref = (0.5 * (Hdn_ref + Hdn_ref.T)) / denom

        self.assertTrue(torch.allclose(Hin_s, Hin_ref, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(Hdn_s, Hdn_ref, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.equal(Hin_s, Hin_s.T))
        self.assertTrue(torch.equal(Hdn_s, Hdn_s.T))

    def test_blend_hessians_extremes_and_weights(self):
        Hh = torch.eye(8) * 4.0
        Hs = torch.eye(8) * 1.0
        self.assertTrue(torch.equal(me.blend_hessians(Hh, Hs, n=0, n0=1024.0), Hs))
        mid = me.blend_hessians(Hh, Hs, n=1024, n0=1024.0)
        self.assertTrue(torch.allclose(mid, torch.eye(8) * 2.5))
        high = me.blend_hessians(Hh, Hs, n=3072, n0=1024.0)
        self.assertTrue(torch.allclose(high, torch.eye(8) * 3.25))


if __name__ == "__main__":
    unittest.main()
