"""Fused CUDA bitshift-trellis Viterbi (L=16) for the E08 encoder. Same semantics as htrq.viterbi."""
import os
import torch
from torch.utils.cpp_extension import load_inline

_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#define LBITS 16
#define NS (1 << LBITS)

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

__global__ void k_step(const float* __restrict__ cin, float* __restrict__ cout, uint8_t* __restrict__ bp,
                       const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ cb,
                       int B, int T, int t, int K) {
  const int Sj = NS >> K;
  const int nb = 1 << K;
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (int64_t)B * Sj) return;
  int b = i / Sj, low = i % Sj;
  const float* cb_ = cin + (int64_t)b * NS;
  float m = cb_[low]; int arg = 0;
  for (int top = 1; top < nb; ++top) {
    float v = cb_[top * Sj + low];
    if (v < m) { m = v; arg = top; }
  }
  bp[i] = (uint8_t)arg;
  float xt = x[(int64_t)b * T + t], wt = w[(int64_t)b * T + t];
  float* co = cout + (int64_t)b * NS + (int64_t)low * nb;
  const float* cbv = cb + (int64_t)low * nb;
  for (int bit = 0; bit < nb; ++bit) { float d = xt - cbv[bit]; co[bit] = m + wt * d * d; }
}

// returns final cost [B,NS] and backpointers [T-1,B,Sj]
std::vector<torch::Tensor> run(torch::Tensor x, torch::Tensor w, torch::Tensor cb, torch::Tensor j0, int64_t K, bool use_j0) {
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
  int64_t n1 = (int64_t)B * Sj;
  for (int t = 1; t < T; ++t) {
    k_step<<<(n1 + th - 1) / th, th, 0, stream>>>(c0.data_ptr<float>(), c1.data_ptr<float>(),
        bp.data_ptr<uint8_t>() + (int64_t)(t - 1) * B * Sj, x.data_ptr<float>(), w.data_ptr<float>(), cb.data_ptr<float>(), B, T, t, (int)K);
    std::swap(c0, c1);
  }
  return {c0, bp};
}
"""
_CPP = "std::vector<torch::Tensor> run(torch::Tensor x, torch::Tensor w, torch::Tensor cb, torch::Tensor j0, int64_t K, bool use_j0);"

_mod = None


def _get():
    global _mod
    if _mod is None:
        bd = os.path.join(os.environ.get("HTR_BUILD_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc"), "vit1_" + ("rocm" if torch.version.hip else "cuda"))
        os.makedirs(bd, exist_ok=True)
        _mod = load_inline("htr_viterbi_ext", cpp_sources=_CPP, cuda_sources=_SRC, functions=["run"],
                           build_directory=bd, extra_cuda_cflags=(["-O3", "--offload-arch=gfx1151"] if torch.version.hip else
                                                                  ["-O3", "-gencode=arch=compute_86,code=sm_86",
                                                                   "-gencode=arch=compute_120,code=sm_120"]), verbose=False)
    return _mod


@torch.no_grad()
def viterbi(x, w, cb, K, tailbite=True):
    S = 1 << 16
    Sj = S >> K
    x = x.float().contiguous(); w = w.float().contiguous(); cb = cb.float().contiguous()
    B, T = x.shape
    dev = x.device
    m = _get()
    dummy = torch.zeros(1, dtype=torch.int64, device=dev)
    cost, bps = m.run(x, w, cb, dummy, K, False)
    ar = torch.arange(S, device=dev)
    if tailbite:
        j0 = (cost.argmin(dim=1) & (Sj - 1)).contiguous()
        del cost, bps
        cost, bps = m.run(x, w, cb, j0, K, True)
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
