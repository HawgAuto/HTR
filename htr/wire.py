"""HTR-W1 wire: bit packing, sign packing, reference decoder (see ../WIRE.md)."""
import math
import torch

L = 16
S = 1 << L
GEN_A, GEN_B = 0xDF6ADF, 0x34983


def gen_values(states):
    """Integer codebook value v(s) in [-510, 510] for int64 states (any shape)."""
    x = (states.long() * GEN_A + GEN_B) & 0xFFFFFFFF
    return ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)) - 510


_CB = {}


def codebook_f32(device="cpu"):
    """cb[s] exactly as the E08/E09 encoder (htrq.codebook('rdna24')): float32(v / rms64)."""
    key = str(device)
    if key not in _CB:
        v = gen_values(torch.arange(S, dtype=torch.int64)).double()
        _CB[key] = (v / v.pow(2).mean().sqrt()).float().to(device)
    return _CB[key]


def cbscale():
    v = gen_values(torch.arange(S, dtype=torch.int64)).double()
    return float(1.0 / v.pow(2).mean().sqrt())


def pack_states(states, K):
    """states int64 [B, 256] (tail-biting trellis path) -> int32 words [B, 8K]."""
    B, T = states.shape
    assert T == 256 and 2 <= K <= 8
    b = (states & ((1 << K) - 1)).long()                                     # [B,256] new K bits per step
    sh = torch.arange(K - 1, -1, -1, device=states.device)
    bits = ((b[:, :, None] >> sh) & 1).reshape(B, 256 * K)                  # MSB-first stream
    w = bits.view(B, 8 * K, 32)
    wsh = torch.arange(31, -1, -1, device=states.device)
    words = (w << wsh).sum(-1)                                               # uint32 value in int64
    return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)


def stream_bits(words):
    """int32 [B, 8K] -> bits int64 [B, 256K]."""
    B, W = words.shape
    u = words.long() & 0xFFFFFFFF
    wsh = torch.arange(31, -1, -1, device=words.device)
    return ((u[:, :, None] >> wsh) & 1).reshape(B, W * 32)


def unpack_states(words, K):
    """Reference state extraction from the circular stream: s_t = 16 bits ending at bit (t+1)K-1."""
    B = words.shape[0]
    bits = stream_bits(words)
    NB = 256 * K
    t = torch.arange(256, device=words.device)
    pos = ((t[:, None] + 1) * K - 16 + torch.arange(16, device=words.device)[None, :]) % NB   # [256,16]
    sel = bits[:, pos]                                                          # [B,256,16]
    wts = (1 << torch.arange(15, -1, -1, device=words.device))
    return (sel * wts).sum(-1)


def pack_signs(s):
    """±1 float vector [N] (N%8==0) -> uint8 [N/8], LSB-first, bit set = -1."""
    neg = (s < 0).to(torch.uint8).view(-1, 8)
    return (neg << torch.arange(8, dtype=torch.uint8, device=s.device)).sum(-1).to(torch.uint8)


def unpack_signs(u8, N):
    bits = (u8.view(-1, 1).long() >> torch.arange(8, device=u8.device)) & 1
    return (1.0 - 2.0 * bits.reshape(-1)[:N].float())


_HC = {}


def hadamard(n, device):
    key = (n, str(device))
    if key not in _HC:
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HC[key] = (h / math.sqrt(n)).float().to(device)
    return _HC[key]


def blk_had_cols_T(W, bs=128):
    m, n = W.shape
    return (W.reshape(m, n // bs, bs) @ hadamard(bs, W.device)).reshape(m, n)


def dequant_Q(qweight, scales):
    """Rotated-domain Q [m, n] float32 (reference)."""
    mt, nt, W = qweight.shape
    K = W // 8
    st = unpack_states(qweight.reshape(-1, W), K)                       # [mt*nt, 256]
    cb = codebook_f32(qweight.device)
    R = cb[st].reshape(mt, nt, 16, 16).permute(0, 2, 1, 3).reshape(mt * 16, nt * 16)
    return R * scales.float()[:, None]


def dequant(qweight, scales, su8, sv8):
    """Original-basis W [m, n] float32 (same op order as htrq.rht_inverse)."""
    Q = dequant_Q(qweight, scales)
    m, n = Q.shape
    su = unpack_signs(su8, n).to(Q.device)
    sv = unpack_signs(sv8, m).to(Q.device)
    W = blk_had_cols_T(Q.T.contiguous()).T
    W = blk_had_cols_T(W.contiguous())
    return (W * sv[:, None] * su[None, :]).contiguous()
