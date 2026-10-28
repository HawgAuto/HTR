"""Python entry points for the HTR-W1 GPU kernels (torch extension built on first use) + torch custom op htr::linear."""
import os
import torch
from . import wire

_EXT = None
CB = wire.cbscale()
_COUNTERS = {}


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
        bd = os.environ.get("HTR_BUILD_DIR") or os.path.join(d, "build_" + ("rocm" if torch.version.hip else "cuda"))
        os.makedirs(bd, exist_ok=True)
        flags = ["-O3"] + (["--offload-arch=" + os.environ.get("HTR_ARCH", "gfx1151")] if torch.version.hip else [])
        _EXT = load("htr_kernels", [os.path.join(d, "htr_kernels.cpp"), os.path.join(d, "htr_kernels.cu")],
                    build_directory=bd, extra_cuda_cflags=flags, verbose=bool(os.environ.get("HTR_VERBOSE")))
    return _EXT


def counters(device):
    key = str(device)
    if key not in _COUNTERS:
        _COUNTERS[key] = torch.zeros(1 << 16, dtype=torch.int32, device=device)
    return _COUNTERS[key]


def nsplit_for(mt, nt, M=1):
    # E10 T2 sweep (bench_kernels_K*.json): best split is ~nt/8 capped at 32; for M>=4 tall shapes want fewer splits
    # (mt=1536 -> 8). The old target_waves//mt rule chose ns=1-6 for tall/wide shapes (+90 us on 24576x4096).
    cap = 32 if M < 4 else max(4, min(32, 12288 // max(1, mt)))
    return max(1, min((nt + 3) // 4, cap, max(2, nt // 8)))


def use_fused(m, n, M):
    # E10 T2 bench_policy.json: the fused single-launch kernel wins only where launch overhead dominates
    # (2560x640 M<=4, 1280x2560 M=1); it is 1.1-2x slower than prologue+GEMV+epilogue on larger weights.
    return FUSED and ((M <= 4 and m * n <= FUSED_MAX_ELEMS_M4) or (M == 1 and m * n <= FUSED_MAX_ELEMS_M1))


def fused_nsplit(m, n, target_wgs=int(os.environ.get("HTR_FUSED_WGS", "240"))):
    return max(1, min(n // 128, -(-target_wgs // (m // 128))))


GEMV_MAX_M = int(os.environ.get("HTR_GEMV_MAX_M", "8"))
FUSED = os.environ.get("HTR_FUSED", "1") == "1"
FUSED_MAX_ELEMS_M1 = int(os.environ.get("HTR_FUSED_MAX_ELEMS_M1", str(1280 * 2560)))
FUSED_MAX_ELEMS_M4 = int(os.environ.get("HTR_FUSED_MAX_ELEMS_M4", str(2560 * 640)))
# M > GEMV_MAX_M (prefill): "w" = dequantize to the original basis (bf16 W, k_dequant_w) + stock bf16 GEMM (default, E10 19:30);
# "rot" = v1 path prologue -> dequant_f16 -> mm(f16, out fp32) -> epilogue (6x slower: hipBLASLt fp32-output GEMM, bench_prefill_K4.json)
PREFILL = os.environ.get("HTR_PREFILL", "w")
# v2 (2026-10-16): decode GEMV with the epilogue fused in (k_gemv_epi; bitwise == gemv_partials + epilogue). HTR_EPI=0 -> v1 path.
EPI = os.environ.get("HTR_EPI", "1") == "1"
# v3: in-WG split reduction (k_gemv_epi8) when the effective split count is a multiple of 8. HTR_EPI8=0 -> plain k_gemv_epi.
EPI8 = os.environ.get("HTR_EPI8", "1") == "1"
# v4: 8 < M <= GEMV_LOOP_MAX uses the chunked decode GEMV (dev/bench_smallm.json: 3-6x faster than dequant_w+GEMM at M 9-32,
# break-even ~M 96 on MiMo shapes). 0 disables.
GEMV_LOOP_MAX = int(os.environ.get("HTR_GEMV_LOOP_MAX", "64"))


def _ns_eff(mt, nt, nsplit):
    tps = ((nt + nsplit - 1) // nsplit + 3) // 4 * 4
    return (nt + tps - 1) // tps


def _linear_impl(x2, qweight, scales, su, sv, bias):
    E = ext()
    M = x2.shape[0]
    mt, nt = qweight.shape[0], qweight.shape[1]
    if M == 0:
        return x2.new_empty((0, mt * 16))
    if M <= GEMV_MAX_M and x2.dtype in (torch.bfloat16, torch.float16) and use_fused(mt * 16, nt * 16, M):
        return E.gemv_fused(qweight, x2, su, sv, scales, CB, bias, counters(x2.device), fused_nsplit(mt * 16, nt * 16))
    if GEMV_MAX_M < M <= GEMV_LOOP_MAX and EPI and x2.dtype in (torch.bfloat16, torch.float16, torch.float32):
        # v4: short prompts run the decode GEMV over 8-row chunks (reads W ceil(M/8) times) instead of k_dequant_w + GEMM
        od = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}[x2.dtype]
        xt, xs, tok = E.prologue(x2, su)
        cnt = counters(x2.device)
        outs = []
        for r0 in range(0, M, GEMV_MAX_M):
            r1 = min(M, r0 + GEMV_MAX_M)
            ns = nsplit_for(mt, nt, r1 - r0)
            outs.append(E.gemv_epi(qweight, xt[r0:r1], xs[r0:r1], tok[r0:r1], scales, CB, sv, bias, cnt, ns, od,
                                   int(EPI8 and _ns_eff(mt, nt, ns) % 8 == 0)))
        return torch.cat(outs)
    if M > GEMV_MAX_M and PREFILL == "w":
        W = E.dequant_w(qweight, scales, CB, su, sv)
        if x2.dtype != torch.bfloat16:
            W = W.to(x2.dtype)
        return torch.nn.functional.linear(x2, W, bias)
    od = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}[x2.dtype]
    xt, xs, tok = E.prologue(x2, su)
    if M <= GEMV_MAX_M and EPI:
        ns = nsplit_for(mt, nt, M)
        return E.gemv_epi(qweight, xt, xs, tok, scales, CB, sv, bias, counters(x2.device), ns, od, int(EPI8 and _ns_eff(mt, nt, ns) % 8 == 0))
    if M <= GEMV_MAX_M:
        part = E.gemv_partials(qweight, xt, nsplit_for(mt, nt, M))
        return E.epilogue(part, xs, tok, scales, CB, sv, bias, 0, od)
    Q = E.dequant_f16(qweight, scales, CB)
    Y = torch.mm(xt, Q.t(), out_dtype=torch.float32)
    return E.epilogue(Y, xs, tok, scales, CB, sv, bias, 1, od)


@torch.library.custom_op("htr::linear", mutates_args=())
def htr_linear_op(x2: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, su: torch.Tensor, sv: torch.Tensor,
                  bias: torch.Tensor | None) -> torch.Tensor:
    return _linear_impl(x2.contiguous(), qweight, scales, su, sv, bias)


@htr_linear_op.register_fake
def _(x2, qweight, scales, su, sv, bias):
    return x2.new_empty((x2.shape[0], qweight.shape[0] * 16))


def linear(x, qweight, scales, su, sv, bias=None):
    """y = x @ W^T (+bias) for an HTR-W1 weight. x [..., n] -> [..., m] in x.dtype."""
    shp = x.shape
    y = htr_linear_op(x.reshape(-1, shp[-1]), qweight, scales, su, sv, bias)
    return y.reshape(*shp[:-1], qweight.shape[0] * 16)
