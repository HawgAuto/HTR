"""HTR encoder (block-LDLQ + tail-biting bitshift-trellis Viterbi) emitting the HTR-W1 wire.
Algorithm identical to htrq.htr_trellis_multi (E08/E09, out_rot=True, gain=1, ldlq=True); differences:
returns the trellis states (packed) and supports a shared input-sign vector for a group of matrices."""
import torch
from . import wire

VITERBI = None          # callable(x[B,256], w[B,256], cb[S], K) -> states int64 [B,256]; set by caller


def _viterbi_ref(x, w, cb, K):
    from .reference import viterbi
    return viterbi(x, w, cb, K)


def blk_had_cols(W, bs=128):
    m, n = W.shape
    return (W.reshape(m, n // bs, bs) @ wire.hadamard(bs, W.device).T).reshape(m, n)


def make_signs(n, m, seed, ji, shared_su):
    """legacy (shared_su=False): exactly htrq.rht_forward(seed + 7919*ji) draws su then sv from one generator.
    shared: su from generator(seed) for the whole input group, sv from generator(seed + 7919*(ji+1))."""
    if not shared_su:
        g = torch.Generator().manual_seed(seed + 7919 * ji)
        su = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()
        sv = (torch.randint(0, 2, (m,), generator=g) * 2 - 1).float()
        return su, sv
    g = torch.Generator().manual_seed(seed)
    su = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()
    g2 = torch.Generator().manual_seed(seed + 7919 * (ji + 1))
    sv = (torch.randint(0, 2, (m,), generator=g2) * 2 - 1).float()
    return su, sv


def prep_H(H, W, damp=0.01):
    H = H.clone().double()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W = W.clone()
    W[:, dead] = 0
    H += damp * torch.diag(H).mean() * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    return H, W


@torch.no_grad()
def encode_group(jobs, K, seed=1234, shared_su=True, damp=0.01, max_batch=1024, log=None):
    """jobs: list of dict(W=[m,n], H=[n,n]) sharing n. Returns list of dict(qweight, scales, su, sv, What)."""
    vit = VITERBI or _viterbi_ref
    n = jobs[0]["W"].shape[1]
    assert n % 128 == 0
    nb = n // 16
    cb = wire.codebook_f32(jobs[0]["W"].device)
    P = []
    for ji, jb in enumerate(jobs):
        W = jb["W"].float()
        H = jb["H"].float().to(W.device)
        m = W.shape[0]
        assert m % 128 == 0 and W.shape[1] == n
        su, sv = make_signs(n, m, seed, ji, shared_su)
        su, sv = su.to(W.device), sv.to(W.device)
        Wt = blk_had_cols(W * sv[:, None] * su[None, :])
        Wt = blk_had_cols(Wt.T.contiguous()).T
        Ht = blk_had_cols(H * su[:, None] * su[None, :])
        Ht = blk_had_cols(Ht.T.contiguous()).T.contiguous()
        Ht, Wt = prep_H(Ht, Wt.contiguous(), damp)
        rs = Wt.pow(2).mean(dim=1).sqrt().half().float().clamp(min=1e-8)
        C = torch.linalg.cholesky(Ht)
        Cblk = C.view(nb, 16, nb, 16).diagonal(dim1=0, dim2=2).permute(2, 0, 1)
        Binv = torch.linalg.inv(Cblk)
        Lb = torch.einsum("ikj,kjl->ikl", C.view(n, nb, 16), Binv).reshape(n, n).float()
        P.append(dict(Wt=Wt.float(), rs=rs, Lb=Lb, su=su, sv=sv, Q=torch.zeros_like(Wt, dtype=torch.float32),
                      F=torch.zeros_like(Wt, dtype=torch.float32), S=torch.zeros((m // 16, nb, 256), dtype=torch.int64,
                                                                                device=W.device)))
    for k in range(nb - 1, -1, -1):
        sl = slice(16 * k, 16 * k + 16)
        xs, ws, sizes = [], [], []
        for p in P:
            tgt = p["Wt"][:, sl] + p["F"][:, sl]
            xs.append((tgt / p["rs"][:, None]).reshape(-1, 256))
            ws.append((p["rs"][:, None] ** 2).expand(-1, 16).contiguous().reshape(-1, 256))
            sizes.append(tgt.shape[0])
        X = torch.cat(xs)
        Wg = torch.cat(ws)
        Wg = Wg / Wg.mean(dim=1, keepdim=True)
        ST = torch.empty(X.shape, dtype=torch.int64, device=X.device)
        for b0 in range(0, X.shape[0], max_batch):
            ST[b0:b0 + max_batch] = vit(X[b0:b0 + max_batch].contiguous(), Wg[b0:b0 + max_batch].contiguous(), cb, K)
        R = cb[ST]
        off = 0
        for p, m in zip(P, sizes):
            nt = m // 16
            p["S"][:, k] = ST[off:off + nt]
            q = R[off:off + nt].reshape(m, 16) * p["rs"][:, None]
            off += nt
            p["Q"][:, sl] = q
            if k > 0:
                Ek = p["Wt"][:, sl] - q
                p["F"][:, :16 * k] += Ek @ p["Lb"][sl, :16 * k]
        if log and (k % 32 == 0):
            log(f"  block {k}/{nb}")
    outs = []
    for p in P:
        mt = p["S"].shape[0]
        qw = wire.pack_states(p["S"].reshape(-1, 256), K).reshape(mt, nb, 8 * K)
        W = wire.blk_had_cols_T(p["Q"].T.contiguous()).T
        W = wire.blk_had_cols_T(W.contiguous())
        What = (W * p["sv"][:, None] * p["su"][None, :]).contiguous()
        outs.append(dict(qweight=qw.cpu(), scales=p["rs"].half().cpu(), su=wire.pack_signs(p["su"]).cpu(),
                         sv=wire.pack_signs(p["sv"]).cpu(), What=What, Q=p["Q"]))
    return outs
