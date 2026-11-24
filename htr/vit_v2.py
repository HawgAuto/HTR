"""Grouped-step CUDA bitshift-trellis Viterbi (L=16). Same semantics/outputs as htrq.viterbi.
r consecutive trellis steps are fused per launch: for fixed middle bits u (L-rK bits) the r-step sub-trellis is
closed (2^(rK) states), so each block keeps U=8 adjacent groups in shared memory and DRAM traffic per step drops ~r x.
"""
import hashlib
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
// Wave-cooperative grouped Viterbi (HTR encoder track, DESIGN v0.1 step 2 + backtrack/template optimization).
// Same outputs as k_group, bit for bit, on the admitted domain (finite x/cb, finite w > 0; host-checked in Python).
// Additions beyond DESIGN v0.1:
// 1. Compile-time specialized k_group_coop_spec<LOG2G, DPP, K_CONST, R_CONST> for the production (K, r) shapes,
//    eliminating runtime integer division, variable shifts, and dead butterfly stages; tails/other r use generic.
// 2. Trimmed pass-1 backpointer allocation: bp stores only rows [bplo, T-1) (half memory for QTIP pass 1, 0 for HTR).
// 3. Single-launch GPU backtrack_qtip_j0, backtrack_full, and in-place mask_tailbite_inplace (replacing 382 Python
//    indexing launches and ~576 MiB of per-call temporary mask tensors).

__global__ void k_mask_tailbite(float* __restrict__ cost, const int64_t* __restrict__ j0, int B, int K) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (int64_t)B * NS) return;
  int b = i >> LBITS;
  int s = i & (NS - 1);
  int mask = (NS >> K) - 1;
  if ((s & mask) != j0[b]) cost[i] = INFINITY;
}

__global__ void k_backtrack_qtip_j0(const uint8_t* __restrict__ bp, const int64_t* __restrict__ s_end,
                                    int64_t* __restrict__ j0, int B, int T, int K, int bplo) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  const int64_t Sj = NS >> K;
  const int h = T / 2;
  int64_t s = s_end[b];
  for (int t = T - 1; t > h; --t) {
    int64_t low = s >> K;
    int64_t top = (int64_t)bp[((int64_t)(t - 1 - bplo) * B + b) * Sj + low];
    s = top * Sj + low;
  }
  j0[b] = s >> K;
}

__global__ void k_backtrack_full(const uint8_t* __restrict__ bp, const int64_t* __restrict__ s_end,
                                 int64_t* __restrict__ states, int B, int T, int K, int bplo) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  const int64_t Sj = NS >> K;
  int64_t s = s_end[b];
  states[(int64_t)b * T + (T - 1)] = s;
  for (int t = T - 1; t > 0; --t) {
    int64_t low = s >> K;
    int64_t top = (int64_t)bp[((int64_t)(t - 1 - bplo) * B + b) * Sj + low];
    s = top * Sj + low;
    states[(int64_t)b * T + (t - 1)] = s;
  }
}

void mask_tailbite_inplace(torch::Tensor cost, torch::Tensor j0, int64_t K) {
  int B = cost.size(0);
  int th = 256;
  int64_t n = (int64_t)B * NS;
  auto stream = at::cuda::getCurrentCUDAStream();
  k_mask_tailbite<<<(n + th - 1) / th, th, 0, stream>>>(
      cost.data_ptr<float>(), j0.data_ptr<int64_t>(), B, (int)K);
}

torch::Tensor backtrack_qtip_j0(torch::Tensor bp, torch::Tensor s_end, int64_t T, int64_t K, int64_t bplo) {
  int B = s_end.size(0);
  auto j0 = torch::empty({B}, s_end.options());
  int th = 64;
  auto stream = at::cuda::getCurrentCUDAStream();
  k_backtrack_qtip_j0<<<(B + th - 1) / th, th, 0, stream>>>(
      bp.data_ptr<uint8_t>(), s_end.data_ptr<int64_t>(), j0.data_ptr<int64_t>(), B, (int)T, (int)K, (int)bplo);
  return j0;
}

torch::Tensor backtrack_full(torch::Tensor bp, torch::Tensor s_end, int64_t T, int64_t K, int64_t bplo) {
  int B = s_end.size(0);
  auto states = torch::empty({B, T}, s_end.options());
  int th = 64;
  auto stream = at::cuda::getCurrentCUDAStream();
  k_backtrack_full<<<(B + th - 1) / th, th, 0, stream>>>(
      bp.data_ptr<uint8_t>(), s_end.data_ptr<int64_t>(), states.data_ptr<int64_t>(), B, (int)T, (int)K, (int)bplo);
  return states;
}

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
    int a = e >> LOG2G, uu = e & (UGC - 1);
    A[uu * NR + coop_ph(a, sh)] = ci[(a << (LBITS - R)) | (u0 + uu)];
  }
  __syncthreads();
  for (int j = 1; j <= r; ++j) {
    const int t = t0 + j;
    const float xt = x[(int64_t)b * T + t], wt = w[(int64_t)b * T + t];
    uint8_t* bpt_ = (t - 1 >= bplo) ? (bp + ((int64_t)(t - 1 - bplo) * B + b) * Sj) : nullptr;
    const int jk = j * K;
    for (int e = threadIdx.x; e < UGC * NR; e += blockDim.x) {
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
      if ((idx & (nb - 1)) == 0 && bpt_ != nullptr) bpt_[sj >> K] = (uint8_t)arg;
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

template <int LOG2G, int DPP, int K_C, int R_C>
__global__ void k_group_coop_spec(const float* __restrict__ cin, float* __restrict__ cout, uint8_t* __restrict__ bp,
                                  const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ cb,
                                  int B, int T, int t0, int bplo) {
  extern __shared__ float sm[];
  constexpr int UGC = 1 << LOG2G;
  constexpr int R = R_C * K_C;
  constexpr int NR = 1 << R;
  constexpr int G = 1 << (LBITS - R);
  constexpr int nb = 1 << K_C;
  constexpr int Sj = NS >> K_C;
  constexpr int sh = R - K_C;
  constexpr int LOG2_BPT = LBITS - R - LOG2G;
  constexpr int BPT = 1 << LOG2_BPT;
  const int b = blockIdx.x >> LOG2_BPT;
  const int u0 = (blockIdx.x & (BPT - 1)) << LOG2G;
  float* A = sm;
  float* Bf = sm + UGC * NR;
  const float* ci = cin + (int64_t)b * NS;
  for (int e = threadIdx.x; e < UGC * NR; e += blockDim.x) {
    int a = e >> LOG2G, uu = e & (UGC - 1);
    A[(uu << R) + (sh ? (a ^ (a >> sh)) : a)] = ci[(a << (LBITS - R)) | (u0 + uu)];
  }
  __syncthreads();
  #pragma unroll
  for (int j = 1; j <= R_C; ++j) {
    const int t = t0 + j;
    const float xt = x[(int64_t)b * T + t], wt = w[(int64_t)b * T + t];
    uint8_t* bpt_ = (t - 1 >= bplo) ? (bp + ((int64_t)(t - 1 - bplo) * B + b) * Sj) : nullptr;
    const int jk = j * K_C;
    const int lo_mask = (1 << jk) - 1;
    const int hi_sh = LBITS - (R_C - j) * K_C;
    for (int e = threadIdx.x; e < UGC * NR; e += blockDim.x) {
      const int lo = e & lo_mask, uu = (e >> jk) & (UGC - 1), hi = e >> (jk + LOG2G);
      const int idx = (hi << jk) | lo;
      const float* Au = A + (uu << R);
      const int base = idx >> K_C;
      const int c = idx & (nb - 1);
      const int src_i = base | (c << sh);
      float m = Au[sh ? (src_i ^ (src_i >> sh)) : src_i]; int arg = c;
      if constexpr (K_C >= 1) coop_step<1, DPP>(m, arg);
      if constexpr (K_C >= 2) coop_step<2, DPP>(m, arg);
      if constexpr (K_C >= 3) coop_step<4, DPP>(m, arg);
      if constexpr (K_C >= 4) coop_step<8, DPP>(m, arg);
      if constexpr (K_C >= 5) coop_step<16, DPP>(m, arg);
      const int sj = ((idx >> jk) << hi_sh) | ((u0 + uu) << jk) | lo;
      const float d = xt - cb[sj];
      Bf[(uu << R) + (sh ? (idx ^ (idx >> sh)) : idx)] = m + wt * d * d;
      if ((idx & (nb - 1)) == 0 && bpt_ != nullptr) bpt_[sj >> K_C] = (uint8_t)arg;
    }
    __syncthreads();
    float* tmp = A; A = Bf; Bf = tmp;
  }
  float* co = cout + (int64_t)b * NS;
  for (int e = threadIdx.x; e < UGC * NR; e += blockDim.x) {
    int uu = e >> R, idx = e & (NR - 1);
    co[((u0 + uu) << R) | idx] = A[(uu << R) + (sh ? (idx ^ (idx >> sh)) : idx)];
  }
}

template <int LOG2G, int DPP, int K_C, int R_C>
static void coop_launch_spec(int nblk, int th2, size_t smem, cudaStream_t stream, float* c0, float* c1, uint8_t* bp,
                             const float* x, const float* w, const float* cb, int B, int T, int t0, int bplo) {
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute((const void*)k_group_coop_spec<LOG2G, DPP, K_C, R_C>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    attr_set = true;
  }
  k_group_coop_spec<LOG2G, DPP, K_C, R_C><<<nblk, th2, smem, stream>>>(c0, c1, bp, x, w, cb, B, T, t0, bplo);
}

template <int LOG2G, int DPP>
static void coop_launch(int nblk, int th2, size_t smem, cudaStream_t stream, float* c0, float* c1, uint8_t* bp,
                        const float* x, const float* w, const float* cb, int B, int T, int t0, int r, int K, int bplo) {
  if (K == 3 && r == 3) { coop_launch_spec<LOG2G, DPP, 3, 3>(nblk, th2, smem, stream, c0, c1, bp, x, w, cb, B, T, t0, bplo); return; }
  if (K == 4 && r == 2) { coop_launch_spec<LOG2G, DPP, 4, 2>(nblk, th2, smem, stream, c0, c1, bp, x, w, cb, B, T, t0, bplo); return; }
  if (K == 5 && r == 2) { coop_launch_spec<LOG2G, DPP, 5, 2>(nblk, th2, smem, stream, c0, c1, bp, x, w, cb, B, T, t0, bplo); return; }
  if (K == 2 && r == 4) { coop_launch_spec<LOG2G, DPP, 2, 4>(nblk, th2, smem, stream, c0, c1, bp, x, w, cb, B, T, t0, bplo); return; }
  if (K == 2 && r == 5) { coop_launch_spec<LOG2G, DPP, 2, 5>(nblk, th2, smem, stream, c0, c1, bp, x, w, cb, B, T, t0, bplo); return; }
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute((const void*)k_group_coop<LOG2G, DPP>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    attr_set = true;
  }
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
  int64_t bp_lo = std::max<int64_t>(0, bplo);
  int64_t bp_rows = std::max<int64_t>(0, (T - 1) - bp_lo);
  auto bp = torch::empty({bp_rows, B, Sj}, opt.dtype(torch::kUInt8));
  auto stream = at::cuda::getCurrentCUDAStream();
  int th = 256;
  int64_t n0 = (int64_t)B * NS;
  k_init<<<(n0 + th - 1) / th, th, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), cb.data_ptr<float>(),
       use_j0 ? j0.data_ptr<int64_t>() : nullptr, c0.data_ptr<float>(), B, T, (int)K);
  int t0 = 0;
  uint8_t* bp_ptr = bp_rows > 0 ? bp.data_ptr<uint8_t>() : nullptr;
  while (t0 < T - 1) {
    int r = (int)rmax; if (t0 + r > T - 1) r = T - 1 - t0;
    int R = r * (int)K;
    TORCH_CHECK(((ugc << R) % 32) == 0 && R <= 16 - log2g, "coop Viterbi: launch shape outside the wave-uniform domain");
    size_t smem = 2 * ugc * (1 << R) * sizeof(float);
    TORCH_CHECK(smem <= 65536, "coop Viterbi: shared memory exceeds 64 KiB");
    int G = 1 << (LBITS - R);
    int nblk = B * (G / ugc);
    switch (variant) {
      case 0: coop_launch<3, 0>(nblk, th2, smem, stream, c0.data_ptr<float>(), c1.data_ptr<float>(), bp_ptr, x.data_ptr<float>(), w.data_ptr<float>(), cb.data_ptr<float>(), B, T, t0, r, (int)K, (int)bp_lo); break;
      case 1: coop_launch<3, 1>(nblk, th2, smem, stream, c0.data_ptr<float>(), c1.data_ptr<float>(), bp_ptr, x.data_ptr<float>(), w.data_ptr<float>(), cb.data_ptr<float>(), B, T, t0, r, (int)K, (int)bp_lo); break;
      case 2: coop_launch<2, 0>(nblk, th2, smem, stream, c0.data_ptr<float>(), c1.data_ptr<float>(), bp_ptr, x.data_ptr<float>(), w.data_ptr<float>(), cb.data_ptr<float>(), B, T, t0, r, (int)K, (int)bp_lo); break;
      default: coop_launch<2, 1>(nblk, th2, smem, stream, c0.data_ptr<float>(), c1.data_ptr<float>(), bp_ptr, x.data_ptr<float>(), w.data_ptr<float>(), cb.data_ptr<float>(), B, T, t0, r, (int)K, (int)bp_lo); break;
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
         "int64_t K, bool use_j0, int64_t rmax, int64_t th2, int64_t variant, int64_t bplo);"
         "\nvoid mask_tailbite_inplace(torch::Tensor cost, torch::Tensor j0, int64_t K);"
         "\ntorch::Tensor backtrack_qtip_j0(torch::Tensor bp, torch::Tensor s_end, int64_t T, int64_t K, int64_t bplo);"
         "\ntorch::Tensor backtrack_full(torch::Tensor bp, torch::Tensor s_end, int64_t T, int64_t K, int64_t bplo);")
RMAX = {1: 10, 2: 5, 3: 3, 4: 2, 5: 2, 6: 1, 8: 1}
_mod = None


def _get():
    global _mod
    if _mod is None:
        tag = hashlib.sha256((_CPP + _SRC).encode()).hexdigest()[:8]
        bd = os.path.join(os.environ.get("HTR_BUILD_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc"), "vit3_" + ("rocm_" if torch.version.hip else "cuda_") + tag)
        os.makedirs(bd, exist_ok=True)
        _mod = load_inline("htr_viterbi_ext3", cpp_sources=_CPP, cuda_sources=_SRC,
                           functions=["run", "run_coop", "mask_tailbite_inplace", "backtrack_qtip_j0", "backtrack_full"],
                           build_directory=bd, extra_cuda_cflags=(["-O3", "--offload-arch=gfx1151"] if torch.version.hip else
                                                                  ["-O3", "-gencode=arch=compute_86,code=sm_86",
                                                                   "-gencode=arch=compute_120,code=sm_120"]), verbose=False)
    return _mod


TH = 256
IMPL = os.environ.get("HTR_VITERBI_IMPL", "group")
_EXPLICIT_IMPL = "HTR_VITERBI_IMPL" in os.environ
# Verified gfx1151 cooperative routes: K3/K4/K5 fullencode-001 (800/800 K3, 24/24 K4, 24/24 K5 byte-identical)
# + backtrack-opt-001 (72 stress cases, 32 microbench cells K2-K5 x qtip/htr x B=80..1024, MiMo L1 K3/K4/K5 real encode).
_AUTO_ROUTES = {
    2: ("coop_dpp_ug4", 5, 512, (80, 256, 512, 768, 1024)),
    3: ("coop_dpp", 3, 512, (80, 256, 512, 640, 768, 1024)),
    4: ("coop_dpp", 2, 256, (80, 256, 512, 768, 1024)),
    5: ("coop_dpp_ug4", 2, 512, (80, 256, 512, 768, 1024)),
}
COOP_VARIANT = {"coop": 0, "coop_dpp": 1, "coop_ug4": 2, "coop_dpp_ug4": 3}


def coop_geometry(K, rmax, T, variant):
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
    if not torch.version.hip or not 2 <= K <= 5 or th % 32:
        return False
    return bool((torch.isfinite(x).all() & torch.isfinite(cb).all() & (torch.isfinite(w) & (w > 0)).all()).item())


def viterbi(x, w, cb, K, tailbite=True, rmax=None, th=None, seam="htr", impl=None, *, _fallback_geometry=None):
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
    use_coop = (impl != "group" and coop_geometry(K, rmax, T, COOP_VARIANT[impl])
                and coop_admissible(x, w, cb, K, th))
    if use_coop:
        variant = COOP_VARIANT[impl]
        def run(xx, ww, jj, use_j0, bplo):
            return m.run_coop(xx, ww, cb, jj, K, use_j0, rmax, th, variant, bplo)
    else:
        if _fallback_geometry is not None:
            rmax, th = _fallback_geometry
        def run(xx, ww, jj, use_j0, bplo):
            return m.run(xx, ww, cb, jj, K, use_j0, rmax, th)
    dummy = torch.zeros(1, dtype=torch.int64, device=dev)
    if seam == "qtip":
        h = T // 2
        cost, bps = run(torch.roll(x, h, 1).contiguous(), torch.roll(w, h, 1).contiguous(), dummy, False, h)
        s = cost.argmin(dim=1)
        bp_off = h if use_coop else 0
        j0 = m.backtrack_qtip_j0(bps, s, T, K, bp_off)
    else:
        cost, bps = run(x, w, dummy, False, T if tailbite else 0)
    if tailbite:
        if seam == "htr":
            j0 = (cost.argmin(dim=1) & (Sj - 1)).contiguous()
        del cost, bps
        cost, bps = run(x, w, j0, True, 0)
        m.mask_tailbite_inplace(cost, j0, K)
    s = cost.argmin(dim=1)
    return m.backtrack_full(bps, s, T, K, 0)


def viterbi_auto(x, w, cb, K, tailbite=True, seam="htr", impl=None):
    with torch.no_grad():
        if torch.version.hip:
            r, th = {2: (4, 256), 3: (3, 512), 4: (2, 512), 5: (2, 512), 6: (1, 512), 8: (1, 512)}[K]
            if (K in _AUTO_ROUTES and impl is None and not _EXPLICIT_IMPL and IMPL == "group"
                    and seam in ("qtip", "htr") and tailbite and x.is_cuda and w.device == x.device == cb.device):
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
