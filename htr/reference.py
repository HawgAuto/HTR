"""Portable reference Viterbi from the original HTR codec screen; no GPU extension required."""
import torch
L = 16
S = 1 << L

SEAMS = ("htr", "qtip")


@torch.no_grad()
def viterbi(x, w, cb, K, tailbite=True, seam="htr"):
    """Batched bitshift-trellis Viterbi.
    x, w: [B, T] targets and per-element weights; cb: [S] value of each 16-bit state.
    Minimises sum w*(x-cb[s_t])^2 with s_t = ((s_{t-1}<<K)&mask)|bits. Two-pass tail-biting
    makes the stream circular, so the rate is exactly K bits/weight. Returns states [B, T].
    seam picks pass 2's pinned state j0 (both rules then run the same pinned pass 2):
      'htr'  (default): j0 = low bits of the argmin end state of a free pass on the block;
      'qtip' (opt-in) : j0 = high bits of the state at original index 0 of a free pass on the block rolled by T/2
                        (QTIP/CPTC/EXL3 rule). Rounding-only: codes change, wire and decoders do not.
    """
    if seam not in SEAMS:
        raise ValueError(f"unknown seam {seam!r}")
    if seam != "htr" and not tailbite:
        raise ValueError("seam applies to tail-biting encodes only")
    B, T = x.shape
    nb = 1 << K
    Sj = S >> K
    dev = x.device
    cb2 = cb.view(Sj, nb)
    ar = torch.arange(S, device=dev)

    def run(j0):
        d0 = w[:, 0:1] * (x[:, 0:1] - cb[None, :]) ** 2
        if j0 is None:
            cost = d0
        else:
            ok = (ar[None, :] >> K) == j0[:, None]
            cost = torch.where(ok, d0, torch.full_like(d0, float("inf")))
        bps = torch.empty((T - 1, B, Sj), dtype=torch.uint8, device=dev)
        for t in range(1, T):
            m, idx = cost.view(B, nb, Sj).min(dim=1)
            bps[t - 1] = idx.to(torch.uint8)
            d = (x[:, t, None, None] - cb2[None]) ** 2
            cost = (m[:, :, None] + w[:, t, None, None] * d).view(B, S)
        return cost, bps

    rows = torch.arange(B, device=dev)
    if seam == "qtip":
        h = T // 2
        x0, w0 = x, w
        x, w = torch.roll(x0, h, 1), torch.roll(w0, h, 1)
        cost, bps = run(None)
        s = cost.argmin(dim=1)
        for t in range(T - 1, h, -1):           # back to rolled index h == original index 0
            s = bps[t - 1][rows, s >> K].long() * Sj + (s >> K)
        j0 = s >> K
        x, w = x0, w0
        del cost, bps
    else:
        cost, bps = run(None)
    if tailbite:
        if seam == "htr":
            j0 = cost.argmin(dim=1) & (Sj - 1)
        cost, bps = run(j0)
        ok = (ar[None, :] & (Sj - 1)) == j0[:, None]
        cost = torch.where(ok, cost, torch.full_like(cost, float("inf")))
    s = cost.argmin(dim=1)
    states = torch.empty((B, T), dtype=torch.int64, device=dev)
    states[:, T - 1] = s
    rows = torch.arange(B, device=dev)
    for t in range(T - 1, 0, -1):
        low = s >> K
        top = bps[t - 1][rows, low].long()
        s = top * Sj + low
        states[:, t - 1] = s
    return states
