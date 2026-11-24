"""Grouped-step CUDA bitshift-trellis Viterbi (L=16). Same semantics/outputs as htrq.viterbi.
r consecutive trellis steps are fused per launch: for fixed middle bits u (L-rK bits) the r-step sub-trellis is
closed (2^(rK) states), so each block keeps U=8 adjacent groups in shared memory and DRAM traffic per step drops ~r x.
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#define LBITS 16
#define NS (1 << LBITS)
#define UG 8

__global__ void k_init(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ cb,
                       const int64_t* __restrict__ j0, float* __restrict__ cost, int B, int T, int K) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (int64_t)B * NS) return;
  int b = i / NS, s = i % NS;
  float d = x[(int64_t)b * T] - cb[s];
  float c = w[(int64_t)b * T] * d * d;
  if (j0 != nullptr && (s >> K) != j0[b]) c = INFINITY;
  cost[i] = c;
}

__global__ void k_group(const float* __restrict__ cin, float* __restrict__ cout, uint8_t* __restrict__ bp,
                        const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ cb,
                        int B, int T, int t0, int r, int K) {
  extern __shared__ float sm[];
  const int R = r * K, NR = 1 << R, G = 1 << (LBITS - R), nb = 1 << K, Sj = NS >> K;
  const int bpt = G / UG;
  const int b = blockIdx.x / bpt;
  const int u0 = (blockIdx.x % bpt) * UG;
  float* A = sm;
  float* Bf = sm + UG * NR;
  const float* ci = cin + (int64_t)b * NS;
  for (int e = threadIdx.x; e < UG * NR; e += blockDim.x) {
    int a = e / UG, uu = e % UG;
    A[uu * NR + a] = ci[(a << (LBITS - R)) | (u0 + uu)];
  }
  __syncthreads();
  for (int j = 1; j <= r; ++j) {
    const int t = t0 + j;
    const float xt = x[(int64_t)b * T + t], wt = w[(int64_t)b * T + t];
    uint8_t* bpt_ = bp + ((int64_t)(t - 1) * B + b) * Sj;
    const int jk = j * K;
    for (int e = threadIdx.x; e < UG * NR; e += blockDim.x) {
      // thread order = global state order (lo fastest, then the 8 adjacent u, then hi) -> coalesced cb/bp access
      const int lo = e & ((1 << jk) - 1), uu = (e >> jk) & (UG - 1), hi = e >> (jk + 3);
      const int idx = (hi << jk) | lo;
      const float* Au = A + uu * NR;
      const int base = idx >> K;
      float m = Au[base]; int arg = 0;
      for (int top = 1; top < nb; ++top) {
        float v = Au[base | (top << (R - K))];
        if (v < m) { m = v; arg = top; }
      }
      const int sj = ((idx >> jk) << (LBITS - (r - j) * K)) | ((u0 + uu) << jk) | (idx & ((1 << jk) - 1));
      const float d = xt - cb[sj];
      Bf[uu * NR + idx] = m + wt * d * d;
      if ((idx & (nb - 1)) == 0) bpt_[sj >> K] = (uint8_t)arg;
    }
    __syncthreads();
    float* tmp = A; A = Bf; Bf = tmp;
  }
  float* co = cout + (int64_t)b * NS;
  for (int e = threadIdx.x; e < UG * NR; e += blockDim.x) {
    int uu = e >> R, idx = e & (NR - 1);
    co[((u0 + uu) << R) | idx] = A[uu * NR + idx];
  }
}

std::vector<torch::Tensor> run(torch::Tensor x, torch::Tensor w, torch::Tensor cb, torch::Tensor j0, int64_t K, bool use_j0, int64_t rmax, int64_t th2) {
  int B = x.size(0), T = x.size(1);
  int Sj = NS >> K;
  auto opt = x.options();
  auto c0 = torch::empty({B, NS}, opt);
  auto c1 = torch::empty({B, NS}, opt);
  auto bp = torch::empty({T - 1, B, Sj}, opt.dtype(torch::kUInt8));
  auto stream = at::cuda::getCurrentCUDAStream();
  int th = 256;
  int64_t n0 = (int64_t)B * NS;
  k_init<<<(n0 + th - 1) / th, th, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), cb.data_ptr<float>(),
       use_j0 ? j0.data_ptr<int64_t>() : nullptr, c0.data_ptr<float>(), B, T, (int)K);
  cudaFuncSetAttribute((const void*)k_group, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * UG * 1024 * 4);
  int t0 = 0;
  while (t0 < T - 1) {
    int r = (int)rmax; if (t0 + r > T - 1) r = T - 1 - t0;
    int R = r * (int)K;
    int G = 1 << (LBITS - R);
    int nblk = B * (G / UG);
    size_t smem = 2 * UG * (1 << R) * sizeof(float);
    k_group<<<nblk, th2, smem, stream>>>(c0.data_ptr<float>(), c1.data_ptr<float>(), bp.data_ptr<uint8_t>(),
        x.data_ptr<float>(), w.data_ptr<float>(), cb.data_ptr<float>(), B, T, t0, r, (int)K);
    std::swap(c0, c1);
    t0 += r;
  }
  return {c0, bp};
}

// ---------------------------------------------------------------------------------------------------------------
// Wave-cooperative grouped Viterbi (HTR encoder track, DESIGN v0.1 step 2). Same outputs as k_group, bit for bit, on
// the admitted domain (finite x/cb, finite w > 0; host-checked in Python): the 2^K siblings of a child group are
// adjacent lanes, lane c loads predecessor top=c, and a log2(2^K) xor butterfly on (cost, index) with k_group's
// strict '<' and lowest-index tie rule replaces the serial scan. A/Bf use the zero-storage swizzle
// PH(a) = a ^ (a >> (R-K)) (identity when R == K). The branch-cost expression is textually k_group's.
// ROCm/wave32 only. Variants: bit 0 = DPP row_xmask/permlanex16 exchange instead of __shfl_xor (ds_bpermute);
// bit 1 = 4 groups per block instead of 8. bplo: backpointers are written only for steps t-1 >= bplo (unread rows
// are skipped; 0 writes all, as k_group).
#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ int coop_ph(int a, int sh) { return sh ? (a ^ (a >> sh)) : a; }
template <int MSK, int DPP> __device__ __forceinline__ int coop_xchg(int v) {
  if constexpr (DPP == 0) return __shfl_xor(v, MSK);
  else if constexpr (MSK < 16) return __builtin_amdgcn_update_dpp(0, v, 0x160 | MSK, 0xF, 0xF, false);
  else return (int)__builtin_amdgcn_permlanex16((unsigned)v, (unsigned)v, 0x76543210u, 0xfedcba98u, false, false);
}
template <int MSK, int DPP> __device__ __forceinline__ void coop_step(float& m, int& arg) {
  const float pv = __builtin_bit_cast(float, coop_xchg<MSK, DPP>(__builtin_bit_cast(int, m)));
  const int pi = coop_xchg<MSK, DPP>(arg);
  const bool take = (pv < m) || (!(m < pv) && pi < arg);
  m = take ? pv : m; arg = take ? pi : arg;
}

template <int LOG2G, int DPP>
__global__ void k_group_coop(const float* __restrict__ cin, float* __restrict__ cout, uint8_t* __restrict__ bp,
                             const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ cb,
                             int B, int T, int t0, int r, int K, int bplo) {
  extern __shared__ float sm[];
  constexpr int UGC = 1 << LOG2G;
  const int R = r * K, NR = 1 << R, G = 1 << (LBITS - R), nb = 1 << K, Sj = NS >> K;
  const int sh = R - K;
  const int bpt = G / UGC;
  const int b = blockIdx.x / bpt;
  const int u0 = (blockIdx.x % bpt) * UGC;
  float* A = sm;
  float* Bf = sm + UGC * NR;
  const float* ci = cin + (int64_t)b * NS;
  for (int e = threadIdx.x; e < UGC * NR; e += blockDim.x) {
    int a = e / UGC, uu = e % UGC;
    A[uu * NR + coop_ph(a, sh)] = ci[(a << (LBITS - R)) | (u0 + uu)];
  }
  __syncthreads();
  for (int j = 1; j <= r; ++j) {
    const int t = t0 + j;
    const float xt = x[(int64_t)b * T + t], wt = w[(int64_t)b * T + t];
    uint8_t* bpt_ = bp + ((int64_t)(t - 1) * B + b) * Sj;
    const int jk = j * K;
    for (int e = threadIdx.x; e < UGC * NR; e += blockDim.x) {   // trip count uniform per wave (host-checked)
      const int lo = e & ((1 << jk) - 1), uu = (e >> jk) & (UGC - 1), hi = e >> (jk + LOG2G);
      const int idx = (hi << jk) | lo;
      const float* Au = A + uu * NR;
      const int base = idx >> K;
      const int c = idx & (nb - 1);
      float m = Au[coop_ph(base | (c << sh), sh)]; int arg = c;
      if (nb > 1) coop_step<1, DPP>(m, arg);
      if (nb > 2) coop_step<2, DPP>(m, arg);
      if (nb > 4) coop_step<4, DPP>(m, arg);
      if (nb > 8) coop_step<8, DPP>(m, arg);
      if (nb > 16) coop_step<16, DPP>(m, arg);
      const int sj = ((idx >> jk) << (LBITS - (r - j) * K)) | ((u0 + uu) << jk) | (idx & ((1 << jk) - 1));
      const float d = xt - cb[sj];
      Bf[uu * NR + coop_ph(idx, sh)] = m + wt * d * d;
      if ((idx & (nb - 1)) == 0 && t - 1 >= bplo) bpt_[sj >> K] = (uint8_t)arg;
    }
    __syncthreads();
    float* tmp = A; A = Bf; Bf = tmp;
  }
  float* co = cout + (int64_t)b * NS;
  for (int e = threadIdx.x; e < UGC * NR; e += blockDim.x) {
    int uu = e >> R, idx = e & (NR - 1);
    co[((u0 + uu) << R) | idx] = A[uu * NR + coop_ph(idx, sh)];
  }
}

template <int LOG2G, int DPP>
static void coop_launch(int nblk, int th2, size_t smem, cudaStream_t stream, float* c0, float* c1, uint8_t* bp,
                        const float* x, const float* w, const float* cb, int B, int T, int t0, int r, int K, int bplo) {
  cudaFuncSetAttribute((const void*)k_group_coop<LOG2G, DPP>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
  k_group_coop<LOG2G, DPP><<<nblk, th2, smem, stream>>>(c0, c1, bp, x, w, cb, B, T, t0, r, K, bplo);
}

std::vector<torch::Tensor> run_coop(torch::Tensor x, torch::Tensor w, torch::Tensor cb, torch::Tensor j0, int64_t K,
                                    bool use_j0, int64_t rmax, int64_t th2, int64_t variant, int64_t bplo) {
  TORCH_CHECK(K >= 2 && K <= 5, "coop Viterbi: K must be 2..5");
  TORCH_CHECK(th2 > 0 && th2 % 32 == 0, "coop Viterbi: threads must be a positive multiple of 32");
  TORCH_CHECK(variant >= 0 && variant <= 3, "coop Viterbi: variant must be 0..3");
  const int log2g = (variant & 2) ? 2 : 3, ugc = 1 << log2g;
  int B = x.size(0), T = x.size(1);
  int Sj = NS >> K;
  auto opt = x.options();
  auto c0 = torch::empty({B, NS}, opt);
  auto c1 = torch::empty({B, NS}, opt);
  auto bp = torch::empty({T - 1, B, Sj}, opt.dtype(torch::kUInt8));
  auto stream = at::cuda::getCurrentCUDAStream();
  int th = 256;
  int64_t n0 = (int64_t)B * NS;
  k_init<<<(n0 + th - 1) / th, th, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), cb.data_ptr<float>(),
       use_j0 ? j0.data_ptr<int64_t>() : nullptr, c0.data_ptr<float>(), B, T, (int)K);
  int t0 = 0;
  while (t0 < T - 1) {
    int r = (int)rmax; if (t0 + r > T - 1) r = T - 1 - t0;
    int R = r * (int)K;
    TORCH_CHECK(((ugc << R) % 32) == 0 && R <= 16 - log2g, "coop Viterbi: launch shape outside the wave-uniform domain");
    size_t smem = 2 * ugc * (1 << R) * sizeof(float);
    TORCH_CHECK(smem <= 65536, "coop Viterbi: shared memory exceeds 64 KiB");
    int G = 1 << (LBITS - R);
    int nblk = B * (G / ugc);
    switch (variant) {
      case 0: coop_launch<3, 0>(nblk, th2, smem, stream, c0.data_ptr<float>(), c1.data_ptr<float>(), bp.data_ptr<uint8_t>(), x.data_ptr<float>(), w.data_ptr<float>(), cb.data_ptr<float>(), B, T, t0, r, (int)K, (int)bplo); break;
      case 1: coop_launch<3, 1>(nblk, th2, smem, stream, c0.data_ptr<float>(), c1.data_ptr<float>(), bp.data_ptr<uint8_t>(), x.data_ptr<float>(), w.data_ptr<float>(), cb.data_ptr<float>(), B, T, t0, r, (int)K, (int)bplo); break;
      case 2: coop_launch<2, 0>(nblk, th2, smem, stream, c0.data_ptr<float>(), c1.data_ptr<float>(), bp.data_ptr<uint8_t>(), x.data_ptr<float>(), w.data_ptr<float>(), cb.data_ptr<float>(), B, T, t0, r, (int)K, (int)bplo); break;
      default: coop_launch<2, 1>(nblk, th2, smem, stream, c0.data_ptr<float>(), c1.data_ptr<float>(), bp.data_ptr<uint8_t>(), x.data_ptr<float>(), w.data_ptr<float>(), cb.data_ptr<float>(), B, T, t0, r, (int)K, (int)bplo); break;
    }
    std::swap(c0, c1);
    t0 += r;
  }
  return {c0, bp};
}
#else
std::vector<torch::Tensor> run_coop(torch::Tensor x, torch::Tensor w, torch::Tensor cb, torch::Tensor j0, int64_t K,
                                    bool use_j0, int64_t rmax, int64_t th2, int64_t variant, int64_t bplo) {
  TORCH_CHECK(false, "coop Viterbi is ROCm (wave32 gfx1151) only");
  return {};
}
#endif
"""
_CPP = "std::vector<torch::Tensor> run(torch::Tensor x, torch::Tensor w, torch::Tensor cb, torch::Tensor j0, int64_t K, bool use_j0, int64_t rmax, int64_t th2);"
_CPP += ("\nstd::vector<torch::Tensor> run_coop(torch::Tensor x, torch::Tensor w, torch::Tensor cb, torch::Tensor j0, "
         "int64_t K, bool use_j0, int64_t rmax, int64_t th2, int64_t variant, int64_t bplo);")
RMAX = {1: 10, 2: 5, 3: 3, 4: 2, 5: 2, 6: 1, 8: 1}
_mod = None


def _get():
    global _mod
    if _mod is None:
        bd = os.path.join(os.environ.get("HTR_BUILD_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc"), "vit3_" + ("rocm" if torch.version.hip else "cuda"))
        os.makedirs(bd, exist_ok=True)
        _mod = load_inline("htr_viterbi_ext3", cpp_sources=_CPP, cuda_sources=_SRC, functions=["run", "run_coop"],
                           build_directory=bd, extra_cuda_cflags=(["-O3", "--offload-arch=gfx1151"] if torch.version.hip else
                                                                  ["-O3", "-gencode=arch=compute_86,code=sm_86",
                                                                   "-gencode=arch=compute_120,code=sm_120"]), verbose=False)
    return _mod


TH = 256
# k_group (default, the shipped kernel) or a wave-cooperative variant (ROCm only; bit-identical on the admitted domain,
# otherwise the call falls back to k_group). HTR_VITERBI_IMPL sets the process default.
IMPL = os.environ.get("HTR_VITERBI_IMPL", "group")
_EXPLICIT_IMPL = "HTR_VITERBI_IMPL" in os.environ
# Verified gfx1151 full-encode winners (fullencode-001: 800/800 K3, 24/24 K4, 24/24 K5 byte-identical).
# Values: (impl, r, threads, eligible B sizes).
_AUTO_ROUTES = {
    3: ("coop_dpp", 3, 512, (256, 512, 640, 768, 1024)),
    4: ("coop_dpp", 2, 256, (256, 512, 768, 1024)),
    5: ("coop_dpp_ug4", 2, 512, (256, 512, 768, 1024)),
}
COOP_VARIANT = {"coop": 0, "coop_dpp": 1, "coop_ug4": 2, "coop_dpp_ug4": 3}


def coop_geometry(K, rmax, T, variant):
    """Every full/tail group must fill wave32, fit LDS and divide the global state space."""
    if K not in (2, 3, 4, 5) or rmax <= 0 or T < 1 or variant not in range(4):
        return False
    log2g = 2 if variant & 2 else 3
    ug = 1 << log2g
    for t0 in range(0, T-1, rmax):
        R = min(rmax, T-1-t0) * K
        if R > 16-log2g or (ug << R) % 32 or 2*ug*(1 << R)*4 > 65536:
            return False
    return True


def coop_admissible(x, w, cb, K, th):
    """The cooperative kernel equals k_group bit for bit only on this domain (DESIGN v0.1 section 2): ROCm, 2 <= K <= 5,
    whole waves, finite x and cb, finite w > 0. encode.py's inputs always satisfy it; anything else uses k_group."""
    if not torch.version.hip or not 2 <= K <= 5 or th % 32:
        return False
    return bool(torch.isfinite(x).all()) and bool(torch.isfinite(cb).all()) and bool((torch.isfinite(w) & (w > 0)).all())


def viterbi(x, w, cb, K, tailbite=True, rmax=None, th=None, seam="htr", impl=None, *, _fallback_geometry=None):
    """seam: 'htr' (default, unchanged) or opt-in 'qtip' (pass 1 on the block rolled by T/2, j0 taken mid-path);
    see htr/reference.py. Both rules run the same pinned pass 2; the wire and decoders are unchanged.
    impl: None (module IMPL), 'group' (k_group) or a COOP_VARIANT key. Every impl returns identical states."""
    if seam not in ("htr", "qtip"):
        raise ValueError(f"unknown seam {seam!r}")
    if seam != "htr" and not tailbite:
        raise ValueError("seam applies to tail-biting encodes only")
    impl = impl or IMPL
    if impl != "group" and impl not in COOP_VARIANT:
        raise ValueError(f"unknown Viterbi impl {impl!r}")
    S = 1 << 16
    Sj = S >> K
    rmax = rmax or RMAX[K]
    th = th or TH
    x = x.float().contiguous(); w = w.float().contiguous(); cb = cb.float().contiguous()
    B, T = x.shape
    dev = x.device
    m = _get()
    if (impl != "group" and coop_geometry(K, rmax, T, COOP_VARIANT[impl])
            and coop_admissible(x, w, cb, K, th)):
        variant = COOP_VARIANT[impl]

        def run(xx, ww, jj, use_j0, bplo):    # bplo: backpointer rows below it are never read and are not written
            return m.run_coop(xx, ww, cb, jj, K, use_j0, rmax, th, variant, bplo)
    else:
        if _fallback_geometry is not None:
            rmax, th = _fallback_geometry  # automatic proposal refused: restore the original group geometry
        def run(xx, ww, jj, use_j0, bplo):
            return m.run(xx, ww, cb, jj, K, use_j0, rmax, th)
    dummy = torch.zeros(1, dtype=torch.int64, device=dev)
    rows = torch.arange(B, device=dev)
    if seam == "qtip":
        h = T // 2
        cost, bps = run(torch.roll(x, h, 1).contiguous(), torch.roll(w, h, 1).contiguous(), dummy, False, h)
        s = cost.argmin(dim=1)
        for t in range(T - 1, h, -1):           # back to rolled index h == original index 0
            low = s >> K
            s = bps[t - 1][rows, low].long() * Sj + low
        j0 = (s >> K).contiguous()
    else:
        cost, bps = run(x, w, dummy, False, T if tailbite else 0)
    ar = torch.arange(S, device=dev)
    if tailbite:
        if seam == "htr":
            j0 = (cost.argmin(dim=1) & (Sj - 1)).contiguous()
        del cost, bps
        cost, bps = run(x, w, j0, True, 0)
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


def viterbi_auto(x, w, cb, K, tailbite=True, seam="htr", impl=None):
    """Fastest measured config on the RTX 3090 (sweep_v2, 1024 tiles, idle GPU):
    K2: grouped r=4/256 thr 0.256 s (v1 0.344); K3: v1 0.340 s (grouped r=3/512 0.358); K4: grouped r=2/256 0.568 s (v1 0.700).
    seam='qtip' (opt-in) always uses the grouped kernel; the default path is unchanged. impl applies on ROCm only."""
    with torch.no_grad():
        if torch.version.hip:   # gfx1151 sweep (sweep_rocm.py, 1024 tiles): K2 r4/256 1.16 s, K3 r3/512 1.24 s, K4 r2/512 1.77 s; v1 2.35/2.81/4.33
            r, th = {2: (4, 256), 3: (3, 512), 4: (2, 512), 5: (2, 512), 6: (1, 512), 8: (1, 512)}[K]
            if (K in _AUTO_ROUTES and impl is None and not _EXPLICIT_IMPL and IMPL == "group"
                    and seam == "qtip" and tailbite and x.is_cuda and w.device == x.device == cb.device):
                proposal, pr, pt, batches = _AUTO_ROUTES[K]
                if (len(x.shape) == 2 and x.shape[0] in batches and x.shape[1] == 256
                        and w.shape == x.shape and tuple(cb.shape) == (1 << 16,)):
                    props = torch.cuda.get_device_properties(x.device)
                    if getattr(props, "gcnArchName", "").partition(":")[0] == "gfx1151":
                        return viterbi(x, w, cb, K, tailbite, rmax=pr, th=pt, seam=seam,
                                       impl=proposal, _fallback_geometry=(r, th))
            return viterbi(x, w, cb, K, tailbite, rmax=r, th=th, seam=seam, impl=impl)
        if K == 3 and seam == "htr":
            from . import vit_v1
            return vit_v1.viterbi(x, w, cb, K, tailbite)
        return viterbi(x, w, cb, K, tailbite, rmax={2: 4, 3: 3, 4: 2, 5: 2, 6: 1, 8: 1}.get(K), th=256, seam=seam)
