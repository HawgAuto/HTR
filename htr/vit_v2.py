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
"""
_CPP = "std::vector<torch::Tensor> run(torch::Tensor x, torch::Tensor w, torch::Tensor cb, torch::Tensor j0, int64_t K, bool use_j0, int64_t rmax, int64_t th2);"
RMAX = {1: 10, 2: 5, 3: 3, 4: 2, 5: 2, 6: 1, 8: 1}
_mod = None


def _get():
    global _mod
    if _mod is None:
        bd = os.path.join(os.environ.get("HTR_BUILD_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc"), "vit2_" + ("rocm" if torch.version.hip else "cuda"))
        os.makedirs(bd, exist_ok=True)
        _mod = load_inline("htr_viterbi_ext2c", cpp_sources=_CPP, cuda_sources=_SRC, functions=["run"],
                           build_directory=bd, extra_cuda_cflags=(["-O3", "--offload-arch=gfx1151"] if torch.version.hip else
                                                                  ["-O3", "-gencode=arch=compute_86,code=sm_86",
                                                                   "-gencode=arch=compute_120,code=sm_120"]), verbose=False)
    return _mod


TH = 256


def viterbi(x, w, cb, K, tailbite=True, rmax=None, th=None):
    S = 1 << 16
    Sj = S >> K
    rmax = rmax or RMAX[K]
    x = x.float().contiguous(); w = w.float().contiguous(); cb = cb.float().contiguous()
    B, T = x.shape
    dev = x.device
    m = _get()
    dummy = torch.zeros(1, dtype=torch.int64, device=dev)
    cost, bps = m.run(x, w, cb, dummy, K, False, rmax, th or TH)
    ar = torch.arange(S, device=dev)
    if tailbite:
        j0 = (cost.argmin(dim=1) & (Sj - 1)).contiguous()
        del cost, bps
        cost, bps = m.run(x, w, cb, j0, K, True, rmax, th or TH)
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


def viterbi_auto(x, w, cb, K, tailbite=True):
    """Fastest measured config on the RTX 3090 (sweep_v2, 1024 tiles, idle GPU):
    K2: grouped r=4/256 thr 0.256 s (v1 0.344); K3: v1 0.340 s (grouped r=3/512 0.358); K4: grouped r=2/256 0.568 s (v1 0.700)."""
    with torch.no_grad():
        if torch.version.hip:   # gfx1151 sweep (sweep_rocm.py, 1024 tiles): K2 r4/256 1.16 s, K3 r3/512 1.24 s, K4 r2/512 1.77 s; v1 2.35/2.81/4.33
            r, th = {2: (4, 256), 3: (3, 512), 4: (2, 512), 5: (2, 512), 6: (1, 512), 8: (1, 512)}[K]
            return viterbi(x, w, cb, K, tailbite, rmax=r, th=th)
        if K == 3:
            from . import vit_v1
            return vit_v1.viterbi(x, w, cb, K, tailbite)
        return viterbi(x, w, cb, K, tailbite, rmax={2: 4, 4: 2, 5: 2, 6: 1, 8: 1}.get(K), th=256)
