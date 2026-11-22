"""Numpy reference bitshift-trellis Viterbi (general L), mirroring htr/vit_v2.py `run` semantics, plus the two
tail-biting seam rules. Used for host brute-force tests and as the in-container cross-check of the GPU path.

run(): cost_0[s] = w0*(x0-cb[s])^2 (inf where (s>>K)!=j0 when pinned); cost_t[s] = min_top cost_{t-1}[top*Sj + (s>>K)]
       + w_t*(x_t-cb[s])^2, ties -> lowest top (vit_v2 uses strict <). bp[t-1][b, s>>K] = argmin top.
seam 'htr'  : free pass on x; j0 = argmin(end cost) & (Sj-1)                       (htr/vit_v2.py:132-139)
seam 'qtip' : free pass on x rolled by T/2; j0 = (state at rolled index T/2) >> K   (QTIP/YAQA bitshift.py quantize)
Both then run pass 2 pinned to j0 and keep only end states with (s & (Sj-1)) == j0.
"""
import numpy as np


def run(x, w, cb, K, j0=None, Lbits=16):
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    cb = np.asarray(cb, dtype=np.float32)
    B, T = x.shape
    S = 1 << Lbits
    nb, Sj = 1 << K, S >> K
    ar = np.arange(S)
    d = x[:, 0:1] - cb[None, :]
    cost = (w[:, 0:1] * d) * d
    if j0 is not None:
        cost = np.where((ar[None, :] >> K) == np.asarray(j0)[:, None], cost, np.float32(np.inf)).astype(np.float32)
    bps = np.empty((T - 1, B, Sj), dtype=np.uint8)
    cb2 = cb.reshape(Sj, nb)
    for t in range(1, T):
        c3 = cost.reshape(B, nb, Sj)
        arg = np.argmin(c3, axis=1)
        m = np.take_along_axis(c3, arg[:, None, :], axis=1)[:, 0, :]
        bps[t - 1] = arg.astype(np.uint8)
        dd = x[:, t, None, None] - cb2[None]
        cost = (m[:, :, None] + (w[:, t, None, None] * dd) * dd).reshape(B, S).astype(np.float32)
    return cost, bps


def backtrace(s_end, bps, K, Lbits=16, stop=0):
    """Return states[:, stop:T] given end states and backpointers; states[:, T-1] = s_end."""
    S = 1 << Lbits
    Sj = S >> K
    Tm1, B, _ = bps.shape
    T = Tm1 + 1
    st = np.zeros((B, T), dtype=np.int64)
    s = np.asarray(s_end, dtype=np.int64).copy()
    st[:, T - 1] = s
    rows = np.arange(B)
    for t in range(T - 1, stop, -1):
        low = s >> K
        top = bps[t - 1][rows, low].astype(np.int64)
        s = top * Sj + low
        st[:, t - 1] = s
    return st


def viterbi(x, w, cb, K, seam="htr", Lbits=16, return_j0=False):
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    B, T = x.shape
    S = 1 << Lbits
    Sj = S >> K
    ar = np.arange(S)
    if seam == "htr":
        cost, _ = run(x, w, cb, K, None, Lbits)
        j0 = np.argmin(cost, axis=1) & (Sj - 1)
    elif seam == "qtip":
        h = T // 2
        xr, wr = np.roll(x, h, axis=1), np.roll(w, h, axis=1)
        cost, bps = run(xr, wr, cb, K, None, Lbits)
        st = backtrace(np.argmin(cost, axis=1), bps, K, Lbits, stop=h)
        j0 = st[:, h] >> K
    else:
        raise ValueError(seam)
    cost, bps = run(x, w, cb, K, j0, Lbits)
    cost = np.where((ar[None, :] & (Sj - 1)) == j0[:, None], cost, np.float32(np.inf))
    st = backtrace(np.argmin(cost, axis=1), bps, K, Lbits)
    return (st, j0) if return_j0 else st


def valid_tailbiting(st, K, Lbits=16):
    """Every transition is a K-bit shift and the cycle closes: (s_t >> K) == (s_{t-1} & (Sj-1)) for all t incl. wrap."""
    Sj = (1 << Lbits) >> K
    prev = np.roll(st, 1, axis=1)
    return bool(np.all((st >> K) == (prev & (Sj - 1))))
