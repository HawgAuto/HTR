// HTR-W1 GPU kernels (gfx1151 first; written in CUDA dialect, hipified by torch cpp_extension on ROCm).
// y = sv . H128( Q . H128(su . x) ),  Q[i,j] = v(s_ij) * scale_i * CB,  v(s) = bytesum(s*A+B) - 510
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cstdlib>

#define GEN_A 0xDF6ADFu
#define GEN_B 0x34983u
typedef _Float16 h2v __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint32_t sad_bytesum(uint32_t x, uint32_t acc) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC__)
  return __builtin_amdgcn_sad_u8(x, 0u, acc);
#else
  return (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24) + acc;
#endif
}
__device__ __forceinline__ uint32_t alignbit(uint32_t hi, uint32_t lo, uint32_t sh) {
  return __builtin_amdgcn_alignbit(hi, lo, sh);
}
// 16-bit window of the virtual MSB-first bit array V[] starting at compile-time bit b
template <int B>
__device__ __forceinline__ uint32_t ext16(const uint32_t* V) {
  constexpr int w = B >> 5, off = B & 31;
  if constexpr (off <= 16) return (V[w] >> (16 - off)) & 0xFFFFu;
  else return alignbit(V[w], V[w + 1], 48 - off) & 0xFFFFu;
}
// f16 bits of (1024 + bytesum(gen(s))) i.e. 1534 + v(s)
__device__ __forceinline__ uint32_t val_f16bits(uint32_t s) {
  return sad_bytesum(s * GEN_A + GEN_B, 0x6400u);
}
__device__ __forceinline__ h2v as_h2(uint32_t u) { return __builtin_bit_cast(h2v, u); }
__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
// v6 (2026-10-16): wave32 lane exchange v -> v[lane ^ MSK] without LDS. __shfl_xor lowers to ds_bpermute_b32 (an LDS round trip;
// k_dequant_w ran ~640 per wave). xor 1..15 = DPP16 row_xmask (0x160|MSK), xor 16 = v_permlanex16 with identity selects.
// Same values as __shfl_xor when the whole wave is active (all call sites are wave-uniform) -> results bitwise unchanged.
template <int MSK> __device__ __forceinline__ float shx(float v) {
  const int iv = __builtin_bit_cast(int, v);
  if constexpr (MSK < 16) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, iv, 0x160 | MSK, 0xF, 0xF, false));
  } else {
    return __builtin_bit_cast(float, (int)__builtin_amdgcn_permlanex16((unsigned)iv, (unsigned)iv, 0x76543210u, 0xfedcba98u, false, false));
  }
}
__device__ __forceinline__ float wave_max(float a) {
  a = fmaxf(a, shx<16>(a)); a = fmaxf(a, shx<8>(a)); a = fmaxf(a, shx<4>(a)); a = fmaxf(a, shx<2>(a)); a = fmaxf(a, shx<1>(a));
  return a;
}
__device__ __forceinline__ float wave_sum(float a) {
  a += shx<16>(a); a += shx<8>(a); a += shx<4>(a); a += shx<2>(a); a += shx<1>(a);
  return a;
}

// ---------------------------------------------------------------- GEMV (M <= 8 activation rows)
// wave = 16-row strip i x tile range of split sp; lane = (g = lane>>3 tile in group of 4, q = lane&7 row pair)
template <int K, int M>
__global__ __launch_bounds__(256) void k_gemv(const uint32_t* __restrict__ qw, const _Float16* __restrict__ xt,
                                              float* __restrict__ part, int mt, int nt, int nsplit, int tps) {
  const int lane = threadIdx.x & 31;
  const int wave = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (wave >= mt * nsplit) return;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7;
  const int n = nt * 16;
  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }
  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  for (int j = j0 + g; j < j1; j += 4) {
    const uint32_t* tp = qw + ((size_t)i * nt + j) * (8 * K);
    uint32_t V[K + 1];
    V[0] = tp[(q * K + 8 * K - 1) % (8 * K)];
#pragma unroll
    for (int k = 0; k < K; ++k) V[k + 1] = tp[q * K + k];
    h2v xv[M][8];
#pragma unroll
    for (int r = 0; r < M; ++r) {
      const uint4* xp = reinterpret_cast<const uint4*>(xt + (size_t)r * n + j * 16);
      uint4 u0 = xp[0], u1 = xp[1];
      xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w);
      xv[r][4] = as_h2(u1.x); xv[r][5] = as_h2(u1.y); xv[r][6] = as_h2(u1.z); xv[r][7] = as_h2(u1.w);
    }
#define HTR_PAIR(U)                                                                        \
    {                                                                                      \
      uint32_t lo = val_f16bits(ext16<16 + ((U) + 1) * K>(V));                             \
      uint32_t hi = val_f16bits(ext16<16 + ((U) + 2) * K>(V));                             \
      h2v w2 = as_h2(lo | (hi << 16));                                                     \
      _Pragma("unroll") for (int r = 0; r < M; ++r) {                                      \
        if ((U) < 16) a0[r] = __builtin_amdgcn_fdot2(w2, xv[r][((U) & 15) >> 1], a0[r], false); \
        else a1[r] = __builtin_amdgcn_fdot2(w2, xv[r][((U) & 15) >> 1], a1[r], false);      \
      }                                                                                    \
    }
    HTR_PAIR(0) HTR_PAIR(2) HTR_PAIR(4) HTR_PAIR(6) HTR_PAIR(8) HTR_PAIR(10) HTR_PAIR(12) HTR_PAIR(14)
    HTR_PAIR(16) HTR_PAIR(18) HTR_PAIR(20) HTR_PAIR(22) HTR_PAIR(24) HTR_PAIR(26) HTR_PAIR(28) HTR_PAIR(30)
#undef HTR_PAIR
  }
#pragma unroll
  for (int r = 0; r < M; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
    const int m = mt * 16;
#pragma unroll
    for (int r = 0; r < M; ++r) {
      float* pp = part + ((size_t)sp * M + r) * m + i * 16 + 2 * q;
      pp[0] = a0[r]; pp[1] = a1[r];
    }
  }
}

// ---------------------------------------------------------------- dequant Q -> f16 [m, n] (rotated basis, scaled)
template <int K>
__global__ __launch_bounds__(256) void k_dequant(const uint32_t* __restrict__ qw, const __half* __restrict__ scales,
                                                 float cb, _Float16* __restrict__ out, int mt, int nt) {
  const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (size_t)mt * nt * 8) return;
  const int q = gid & 7;
  const size_t tile = gid >> 3;
  const int i = tile / nt, j = tile % nt;
  const uint32_t* tp = qw + tile * (8 * K);
  uint32_t V[K + 1];
  V[0] = tp[(q * K + 8 * K - 1) % (8 * K)];
#pragma unroll
  for (int k = 0; k < K; ++k) V[k + 1] = tp[q * K + k];
  const int n = nt * 16;
  const int r0 = i * 16 + 2 * q;
  const float s0 = __half2float(scales[r0]) * cb, s1 = __half2float(scales[r0 + 1]) * cb;
  union alignas(16) { _Float16 h[32]; uint4 u[4]; } o;
#define HTR_DQ(U) { uint32_t fb = val_f16bits(ext16<16 + ((U) + 1) * K>(V));                           \
                    float v = (float)__builtin_bit_cast(_Float16, (uint16_t)fb) - 1534.0f;            \
                    o.h[U] = (_Float16)(v * ((U) < 16 ? s0 : s1)); }
  HTR_DQ(0) HTR_DQ(1) HTR_DQ(2) HTR_DQ(3) HTR_DQ(4) HTR_DQ(5) HTR_DQ(6) HTR_DQ(7)
  HTR_DQ(8) HTR_DQ(9) HTR_DQ(10) HTR_DQ(11) HTR_DQ(12) HTR_DQ(13) HTR_DQ(14) HTR_DQ(15)
  HTR_DQ(16) HTR_DQ(17) HTR_DQ(18) HTR_DQ(19) HTR_DQ(20) HTR_DQ(21) HTR_DQ(22) HTR_DQ(23)
  HTR_DQ(24) HTR_DQ(25) HTR_DQ(26) HTR_DQ(27) HTR_DQ(28) HTR_DQ(29) HTR_DQ(30) HTR_DQ(31)
#undef HTR_DQ
  uint4* d0 = reinterpret_cast<uint4*>(out + (size_t)r0 * n + j * 16);
  uint4* d1 = reinterpret_cast<uint4*>(out + (size_t)(r0 + 1) * n + j * 16);
  d0[0] = o.u[0]; d0[1] = o.u[1]; d1[0] = o.u[2]; d1[1] = o.u[3];
}

// ---------------------------------------------------------------- raw integer decode (bitwise oracle hook)
template <int K>
__global__ void k_decode_raw(const uint32_t* __restrict__ qw, int16_t* __restrict__ out, int ntiles) {
  const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (size_t)ntiles * 8) return;
  const int q = gid & 7;
  const size_t tile = gid >> 3;
  const uint32_t* tp = qw + tile * (8 * K);
  uint32_t V[K + 1];
  V[0] = tp[(q * K + 8 * K - 1) % (8 * K)];
#pragma unroll
  for (int k = 0; k < K; ++k) V[k + 1] = tp[q * K + k];
  int16_t* o = out + tile * 256 + q * 32;
#define HTR_RAW(U) o[U] = (int16_t)((int)(val_f16bits(ext16<16 + ((U) + 1) * K>(V)) & 0x3FF) - 510);
  HTR_RAW(0) HTR_RAW(1) HTR_RAW(2) HTR_RAW(3) HTR_RAW(4) HTR_RAW(5) HTR_RAW(6) HTR_RAW(7)
  HTR_RAW(8) HTR_RAW(9) HTR_RAW(10) HTR_RAW(11) HTR_RAW(12) HTR_RAW(13) HTR_RAW(14) HTR_RAW(15)
  HTR_RAW(16) HTR_RAW(17) HTR_RAW(18) HTR_RAW(19) HTR_RAW(20) HTR_RAW(21) HTR_RAW(22) HTR_RAW(23)
  HTR_RAW(24) HTR_RAW(25) HTR_RAW(26) HTR_RAW(27) HTR_RAW(28) HTR_RAW(29) HTR_RAW(30) HTR_RAW(31)
#undef HTR_RAW
}

// ---------------------------------------------------------------- prologue: xt = f16( H128(su . x) / tokscale )
// one block (256 thr = 8 waves) per token row; each wave transforms 128-blocks (4 values / lane)
template <int MSK> __device__ __forceinline__ void fwht_stage(float v[4], int lane) {
  const bool up = (lane & MSK) != 0;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float o = shx<MSK>(v[e]);
    v[e] = up ? (o - v[e]) : (v[e] + o);
  }
}
__device__ __forceinline__ void fwht128_wave(float v[4], int lane) {
  float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
  v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
  fwht_stage<1>(v, lane); fwht_stage<2>(v, lane); fwht_stage<4>(v, lane); fwht_stage<8>(v, lane); fwht_stage<16>(v, lane);
}
template <typename T>
__global__ __launch_bounds__(256) void k_prologue(const T* __restrict__ x, const uint8_t* __restrict__ su,
                                                  _Float16* __restrict__ xt, float* __restrict__ xs,
                                                  float* __restrict__ tok, int n) {
  __shared__ float red[8];
  __shared__ float s_scale;
  const int row = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const T* xr = x + (size_t)row * n;
  float amax = 0.f;
  for (int e = threadIdx.x; e < n; e += 256) amax = fmaxf(amax, fabsf(to_f(xr[e])));
  amax = wave_max(amax);
  if (lane == 0) red[w] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = 0.f;
    for (int k = 0; k < 8; ++k) m = fmaxf(m, red[k]);
    const float bound = m * 11.4f;          // |H128 x| <= sqrt(128) * amax
    s_scale = bound > 30000.f ? exp2f(ceilf(log2f(bound / 30000.f))) : 1.f;
    tok[row] = s_scale;
  }
  __syncthreads();
  const float inv = (1.0f / 11.313708498984761f) / s_scale;   // 1/sqrt(128)/tokscale
  float sum = 0.f;
  for (int blk = w; blk < n / 128; blk += 8) {
    const int base = blk * 128 + lane * 4;
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int idx = base + e;
      const float sg = ((su[idx >> 3] >> (idx & 7)) & 1) ? -1.f : 1.f;
      v[e] = to_f(xr[idx]) * sg;
    }
    fwht128_wave(v, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      _Float16 h = (_Float16)(v[e] * inv);
      xt[(size_t)row * n + base + e] = h;
      sum += (float)h;
    }
  }
  sum = wave_sum(sum);
  __syncthreads();
  if (lane == 0) red[w] = sum;
  __syncthreads();
  if (threadIdx.x == 0) { float s = 0.f; for (int k = 0; k < 8; ++k) s += red[k]; xs[row] = s; }
}

// v4 (2026-10-16): single-pass prologue. x is read once into registers (NB 128-blocks per wave), then amax -> tokscale -> H128 ->
// f16. Bitwise identical to k_prologue (max is order-independent; the f16 sum keeps k_prologue's per-thread block order and the
// same shfl/LDS reduction). Removes the second global read pass (the M=1 decode prologue is latency-bound, ~8 us per launch).
template <typename T, int NB>
__global__ __launch_bounds__(256) void k_prologue1p(const T* __restrict__ x, const uint8_t* __restrict__ su,
                                                    _Float16* __restrict__ xt, float* __restrict__ xs,
                                                    float* __restrict__ tok, int n) {
  __shared__ float red[8];
  __shared__ float s_scale;
  const int row = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const T* xr = x + (size_t)row * n;
  const int nblk = n >> 7;
  float v[NB][4];
  float amax = 0.f;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      const int base = blk * 128 + lane * 4;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int idx = base + e;
        const float xv = to_f(xr[idx]);
        amax = fmaxf(amax, fabsf(xv));
        const float sg = ((su[idx >> 3] >> (idx & 7)) & 1) ? -1.f : 1.f;
        v[t][e] = xv * sg;
      }
    }
  }
  amax = wave_max(amax);
  if (lane == 0) red[w] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = 0.f;
    for (int k = 0; k < 8; ++k) m = fmaxf(m, red[k]);
    const float bound = m * 11.4f;
    s_scale = bound > 30000.f ? exp2f(ceilf(log2f(bound / 30000.f))) : 1.f;
    tok[row] = s_scale;
  }
  __syncthreads();
  const float inv = (1.0f / 11.313708498984761f) / s_scale;
  float sum = 0.f;
#pragma unroll
  for (int t = 0; t < NB; ++t) {
    const int blk = w + 8 * t;
    if (blk < nblk) {
      fwht128_wave(v[t], lane);
      const int base = blk * 128 + lane * 4;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        _Float16 h = (_Float16)(v[t][e] * inv);
        xt[(size_t)row * n + base + e] = h;
        sum += (float)h;
      }
    }
  }
  sum = wave_sum(sum);
  __syncthreads();
  if (lane == 0) red[w] = sum;
  __syncthreads();
  if (threadIdx.x == 0) { float s = 0.f; for (int k = 0; k < 8; ++k) s += red[k]; xs[row] = s; }
}

// ---------------------------------------------------------------- epilogue: y = sv . H128(ytilde) (+bias), bf16 out
// mode 0: ytilde_i = (sum_sp part - 1534*xs) * scale_i * cb * tok   (GEMV partials)
// mode 1: ytilde_i = Y[r, i] * tok                                   (dense GEMM output, Q already scaled)
template <typename TO>
__global__ __launch_bounds__(128) void k_epilogue(const float* __restrict__ part, int nsplit, int M, const float* __restrict__ xs,
                                                  const float* __restrict__ tok, const __half* __restrict__ scales, float cb,
                                                  const uint8_t* __restrict__ sv, const TO* __restrict__ bias,
                                                  TO* __restrict__ y, int m, int mode) {
  __shared__ float buf[128];
  const int r = blockIdx.y, i = blockIdx.x * 128 + threadIdx.x;
  float v;
  if (mode == 0) {
    float s = 0.f;
    for (int sp = 0; sp < nsplit; ++sp) s += part[((size_t)sp * M + r) * m + i];
    v = (s - 1534.0f * xs[r]) * __half2float(scales[i]) * cb * tok[r];
  } else {
    v = part[(size_t)r * m + i] * tok[r];
  }
  buf[threadIdx.x] = v;
  __syncthreads();
  for (int h = 1; h < 128; h <<= 1) {
    const int t = threadIdx.x;
    float a = buf[t], b = buf[t ^ h];
    __syncthreads();
    buf[t] = (t & h) ? (b - a) : (a + b);
    __syncthreads();
  }
  float o = buf[threadIdx.x] * (1.0f / 11.313708498984761f);
  if ((sv[i >> 3] >> (i & 7)) & 1) o = -o;
  if (bias != nullptr) o += to_f(bias[i]);
  y[(size_t)r * m + i] = from_f<TO>(o);
}


// ---------------------------------------------------------------- v2 (2026-10-16): GEMV with the epilogue fused in (decode M <= 8)
// Same split-K GEMV as k_gemv (identical partial values and layout), then per 128-row block a self-resetting arrival counter:
// the last of the block's 8*nsplit waves sums the partials (nontemporal loads after the fence, as k_gemv_fused), applies
// (s - 1534*xs)*scale*cb*tok, the wave-level H128 (fwht128_wave), sv and bias, and writes y. Removes the k_epilogue launch.
template <int K, int M, typename TO>
__global__ __launch_bounds__(256) void k_gemv_epi(const uint32_t* __restrict__ qw, const _Float16* __restrict__ xt,
    float* __restrict__ part, unsigned* __restrict__ counters, const float* __restrict__ xs, const float* __restrict__ tok,
    const __half* __restrict__ scales, float cb, const uint8_t* __restrict__ sv, const TO* __restrict__ bias,
    TO* __restrict__ y, int mt, int nt, int nsplit, int tps) {
  const int lane = threadIdx.x & 31;
  const int wave = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (wave >= mt * nsplit) return;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7;
  const int n = nt * 16, m = mt * 16;
  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }
  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  for (int j = j0 + g; j < j1; j += 4) {
    const uint32_t* tp = qw + ((size_t)i * nt + j) * (8 * K);
    uint32_t V[K + 1];
    V[0] = tp[(q * K + 8 * K - 1) % (8 * K)];
#pragma unroll
    for (int k = 0; k < K; ++k) V[k + 1] = tp[q * K + k];
    h2v xv[M][8];
#pragma unroll
    for (int r = 0; r < M; ++r) {
      const uint4* xp = reinterpret_cast<const uint4*>(xt + (size_t)r * n + j * 16);
      uint4 u0 = xp[0], u1 = xp[1];
      xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w);
      xv[r][4] = as_h2(u1.x); xv[r][5] = as_h2(u1.y); xv[r][6] = as_h2(u1.z); xv[r][7] = as_h2(u1.w);
    }
#define HTR_PAIR(U)                                                                        \
    {                                                                                      \
      uint32_t lo = val_f16bits(ext16<16 + ((U) + 1) * K>(V));                             \
      uint32_t hi = val_f16bits(ext16<16 + ((U) + 2) * K>(V));                             \
      h2v w2 = as_h2(lo | (hi << 16));                                                     \
      _Pragma("unroll") for (int r = 0; r < M; ++r) {                                      \
        if ((U) < 16) a0[r] = __builtin_amdgcn_fdot2(w2, xv[r][((U) & 15) >> 1], a0[r], false); \
        else a1[r] = __builtin_amdgcn_fdot2(w2, xv[r][((U) & 15) >> 1], a1[r], false);      \
      }                                                                                    \
    }
    HTR_PAIR(0) HTR_PAIR(2) HTR_PAIR(4) HTR_PAIR(6) HTR_PAIR(8) HTR_PAIR(10) HTR_PAIR(12) HTR_PAIR(14)
    HTR_PAIR(16) HTR_PAIR(18) HTR_PAIR(20) HTR_PAIR(22) HTR_PAIR(24) HTR_PAIR(26) HTR_PAIR(28) HTR_PAIR(30)
#undef HTR_PAIR
  }
#pragma unroll
  for (int r = 0; r < M; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      float* pp = part + ((size_t)sp * M + r) * m + i * 16 + 2 * q;
      pp[0] = a0[r]; pp[1] = a1[r];
    }
  }
  const int b = i >> 3;                       // 128-row block
  // v5: release-only fence before the arrival (a full __threadfence() is seq_cst = also an ACQUIRE, which invalidates the CU's
  // vector L0/L1 and makes still-running waves re-fetch xt); only the last arriver needs the acquire.
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  unsigned old = 0;
  if (lane == 0) old = atomicAdd(&counters[b], 1u);
  old = __shfl(old, 0);
  if (old != (unsigned)(8 * nsplit - 1)) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  const int base = b * 128 + lane * 4;
  for (int r = 0; r < M; ++r) {
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < nsplit; ++k) {
      const float* pp = part + ((size_t)k * M + r) * m + base;
#pragma unroll
      for (int e = 0; e < 4; ++e) v[e] += __builtin_nontemporal_load(pp + e);
    }
    const float c = 1534.0f * xs[r], t = tok[r];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = (v[e] - c) * __half2float(scales[base + e]) * cb * t;   // same association as k_epilogue
    fwht128_wave(v, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = base + e;
      float o = v[e] * (1.0f / 11.313708498984761f);
      if ((sv[row >> 3] >> (row & 7)) & 1) o = -o;
      if (bias != nullptr) o += to_f(bias[row]);
      y[(size_t)r * m + row] = from_f<TO>(o);
    }
  }
  if (lane == 0) counters[b] = 0u;
}


// ---------------------------------------------------------------- v3 (2026-10-16): k_gemv_epi8 = k_gemv_epi + in-WG split reduction
// Requires nsplit % 8 == 0: the 8 waves of a WG then share one 16-row strip i and cover splits sp0..sp0+7; their partials are
// summed in LDS (fixed order w=0..7) and ONE partial per WG is written (partial traffic /8). Arrival counter per 128-row block
// counts WGs (8 strips x nsplit/8 = nsplit); the last WG runs the epilogue with rows r spread over its waves.
// Not bitwise vs v1 (different fp32 summation order of the split partials); deterministic.
template <int K, int M, typename TO>
__global__ __launch_bounds__(256) void k_gemv_epi8(const uint32_t* __restrict__ qw, const _Float16* __restrict__ xt,
    float* __restrict__ part, unsigned* __restrict__ counters, const float* __restrict__ xs, const float* __restrict__ tok,
    const __half* __restrict__ scales, float cb, const uint8_t* __restrict__ sv, const TO* __restrict__ bias,
    TO* __restrict__ y, int mt, int nt, int nsplit, int tps) {
  __shared__ float red[8][M][16];
  __shared__ unsigned is_last;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int wave = blockIdx.x * 8 + w;
  const int i = wave / nsplit, sp = wave % nsplit;
  const int g = lane >> 3, q = lane & 7;
  const int n = nt * 16, m = mt * 16, ns8 = nsplit >> 3;
  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }
  const int j0 = sp * tps, j1 = min(nt, j0 + tps);
  for (int j = j0 + g; j < j1; j += 4) {
    const uint32_t* tp = qw + ((size_t)i * nt + j) * (8 * K);
    uint32_t V[K + 1];
    V[0] = tp[(q * K + 8 * K - 1) % (8 * K)];
#pragma unroll
    for (int k = 0; k < K; ++k) V[k + 1] = tp[q * K + k];
    h2v xv[M][8];
#pragma unroll
    for (int r = 0; r < M; ++r) {
      const uint4* xp = reinterpret_cast<const uint4*>(xt + (size_t)r * n + j * 16);
      uint4 u0 = xp[0], u1 = xp[1];
      xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w);
      xv[r][4] = as_h2(u1.x); xv[r][5] = as_h2(u1.y); xv[r][6] = as_h2(u1.z); xv[r][7] = as_h2(u1.w);
    }
#define HTR_PAIR(U)                                                                        \
    {                                                                                      \
      uint32_t lo = val_f16bits(ext16<16 + ((U) + 1) * K>(V));                             \
      uint32_t hi = val_f16bits(ext16<16 + ((U) + 2) * K>(V));                             \
      h2v w2 = as_h2(lo | (hi << 16));                                                     \
      _Pragma("unroll") for (int r = 0; r < M; ++r) {                                      \
        if ((U) < 16) a0[r] = __builtin_amdgcn_fdot2(w2, xv[r][((U) & 15) >> 1], a0[r], false); \
        else a1[r] = __builtin_amdgcn_fdot2(w2, xv[r][((U) & 15) >> 1], a1[r], false);      \
      }                                                                                    \
    }
    HTR_PAIR(0) HTR_PAIR(2) HTR_PAIR(4) HTR_PAIR(6) HTR_PAIR(8) HTR_PAIR(10) HTR_PAIR(12) HTR_PAIR(14)
    HTR_PAIR(16) HTR_PAIR(18) HTR_PAIR(20) HTR_PAIR(22) HTR_PAIR(24) HTR_PAIR(26) HTR_PAIR(28) HTR_PAIR(30)
#undef HTR_PAIR
  }
#pragma unroll
  for (int r = 0; r < M; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) { red[w][r][2 * q] = a0[r]; red[w][r][2 * q + 1] = a1[r]; }
  }
  __syncthreads();
  if (threadIdx.x < M * 16) {
    const int r = threadIdx.x >> 4, c = threadIdx.x & 15;
    float s = 0.f;
#pragma unroll
    for (int ww = 0; ww < 8; ++ww) s += red[ww][r][c];
    part[((size_t)(sp >> 3) * M + r) * m + i * 16 + c] = s;
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");   // v5: release-only (see k_gemv_epi)
  __syncthreads();
  const int b = i >> 3;
  if (threadIdx.x == 0) is_last = (atomicAdd(&counters[b], 1u) == (unsigned)(nsplit - 1));
  __syncthreads();
  if (!is_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  const int base = b * 128 + lane * 4;
  for (int r = w; r < M; r += 8) {
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < ns8; ++k) {
      const float* pp = part + ((size_t)k * M + r) * m + base;
#pragma unroll
      for (int e = 0; e < 4; ++e) v[e] += __builtin_nontemporal_load(pp + e);
    }
    const float c = 1534.0f * xs[r], t = tok[r];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = (v[e] - c) * __half2float(scales[base + e]) * cb * t;
    fwht128_wave(v, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = base + e;
      float o = v[e] * (1.0f / 11.313708498984761f);
      if ((sv[row >> 3] >> (row & 7)) & 1) o = -o;
      if (bias != nullptr) o += to_f(bias[row]);
      y[(size_t)r * m + row] = from_f<TO>(o);
    }
  }
  if (threadIdx.x == 0) counters[b] = 0u;
}


// ---------------------------------------------------------------- fused single-launch GEMV (prologue + GEMV + epilogue)
// grid (m/128 strip-blocks, nsplit); block 256 = 8 waves; wave w -> strip 8*bx + w; WG chunk = 128-blocks [b0, b1).
// x~ per (row r, 128-block) is stored f16 in LDS with its own power-of-2 scale; per-tile dots are rescaled into acc.
// The last WG of a strip-block (self-resetting atomic counter) reduces the split partials and applies
// scale*CB, the output H128, sv and bias.
#define FUSED_MAXB 16
template <int K, int M, typename T>
__global__ __launch_bounds__(256) void k_gemv_fused(const uint32_t* __restrict__ qw, const T* __restrict__ x,
    const uint8_t* __restrict__ su, const uint8_t* __restrict__ sv, const __half* __restrict__ scales, float cb,
    const T* __restrict__ bias, float* __restrict__ part, unsigned* __restrict__ counters, T* __restrict__ y,
    int m, int n, int bps, int mr) {
  __shared__ __attribute__((aligned(16))) _Float16 xs_l[M][FUSED_MAXB * 128];
  __shared__ float bscale[M][FUSED_MAXB];
  __shared__ float bsum[M][FUSED_MAXB];
  __shared__ float ep[128];
  __shared__ unsigned is_last;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int nt = n >> 4, sb = blockIdx.x, sp = blockIdx.y, nsplit = gridDim.y;
  const int b0 = sp * bps, b1 = min(n >> 7, b0 + bps), nb = b1 - b0;
  for (int it = w; it < M * nb; it += 8) {
    const int r = it / nb, bl = it % nb, base = (b0 + bl) * 128 + lane * 4;
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int idx = base + e;
      const float sg = ((su[idx >> 3] >> (idx & 7)) & 1) ? -1.f : 1.f;
      v[e] = r < mr ? to_f(x[(size_t)r * n + idx]) * sg : 0.f;
    }
    fwht128_wave(v, lane);
    float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    am = wave_max(am);
    am *= (1.0f / 11.313708498984761f);
    const float sc = am > 30000.f ? exp2f(ceilf(log2f(am / 30000.f))) : 1.f;
    const float inv = (1.0f / 11.313708498984761f) / sc;
    float sum = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      _Float16 h = (_Float16)(v[e] * inv);
      xs_l[r][bl * 128 + lane * 4 + e] = h;
      sum += (float)h;
    }
    sum = wave_sum(sum);
    if (lane == 0) { bscale[r][bl] = sc; bsum[r][bl] = sum; }
  }
  __syncthreads();
  const int i = sb * 8 + w;
  const int g = lane >> 3, q = lane & 7;
  float a0[M], a1[M];
#pragma unroll
  for (int r = 0; r < M; ++r) { a0[r] = 0.f; a1[r] = 0.f; }
  for (int jl = g; jl < nb * 8; jl += 4) {
    const int j = b0 * 8 + jl, bl = jl >> 3;
    const uint32_t* tp = qw + ((size_t)i * nt + j) * (8 * K);
    uint32_t V[K + 1];
    V[0] = tp[(q * K + 8 * K - 1) % (8 * K)];
#pragma unroll
    for (int k = 0; k < K; ++k) V[k + 1] = tp[q * K + k];
    h2v xv[M][8];
    float d0[M], d1[M];
#pragma unroll
    for (int r = 0; r < M; ++r) {
      const uint4* xp = reinterpret_cast<const uint4*>(&xs_l[r][jl * 16]);
      uint4 u0 = xp[0], u1 = xp[1];
      xv[r][0] = as_h2(u0.x); xv[r][1] = as_h2(u0.y); xv[r][2] = as_h2(u0.z); xv[r][3] = as_h2(u0.w);
      xv[r][4] = as_h2(u1.x); xv[r][5] = as_h2(u1.y); xv[r][6] = as_h2(u1.z); xv[r][7] = as_h2(u1.w);
      d0[r] = 0.f; d1[r] = 0.f;
    }
#define HTR_PAIRF(U)                                                                       \
    {                                                                                      \
      uint32_t lo = val_f16bits(ext16<16 + ((U) + 1) * K>(V));                             \
      uint32_t hi = val_f16bits(ext16<16 + ((U) + 2) * K>(V));                             \
      h2v w2 = as_h2(lo | (hi << 16));                                                     \
      _Pragma("unroll") for (int r = 0; r < M; ++r) {                                      \
        if ((U) < 16) d0[r] = __builtin_amdgcn_fdot2(w2, xv[r][((U) & 15) >> 1], d0[r], false); \
        else d1[r] = __builtin_amdgcn_fdot2(w2, xv[r][((U) & 15) >> 1], d1[r], false);      \
      }                                                                                    \
    }
    HTR_PAIRF(0) HTR_PAIRF(2) HTR_PAIRF(4) HTR_PAIRF(6) HTR_PAIRF(8) HTR_PAIRF(10) HTR_PAIRF(12) HTR_PAIRF(14)
    HTR_PAIRF(16) HTR_PAIRF(18) HTR_PAIRF(20) HTR_PAIRF(22) HTR_PAIRF(24) HTR_PAIRF(26) HTR_PAIRF(28) HTR_PAIRF(30)
#undef HTR_PAIRF
#pragma unroll
    for (int r = 0; r < M; ++r) { const float s = bscale[r][bl]; a0[r] += s * d0[r]; a1[r] += s * d1[r]; }
  }
#pragma unroll
  for (int r = 0; r < M; ++r) {
    a0[r] += shx<8>(a0[r]); a0[r] += shx<16>(a0[r]);
    a1[r] += shx<8>(a1[r]); a1[r] += shx<16>(a1[r]);
  }
  if (g == 0) {
#pragma unroll
    for (int r = 0; r < M; ++r) {
      float off = 0.f;
      for (int bl = 0; bl < nb; ++bl) off += bscale[r][bl] * bsum[r][bl];
      off *= 1534.0f;
      float* pp = part + ((size_t)sp * M + r) * m + i * 16 + 2 * q;
      pp[0] = a0[r] - off; pp[1] = a1[r] - off;
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) is_last = (atomicAdd(&counters[sb], 1u) == (unsigned)(nsplit - 1));
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  const int t = threadIdx.x, row = sb * 128 + (t & 127);
  for (int r = 0; r < mr; ++r) {
    if (t < 128) {
      float s = 0.f;
      for (int k = 0; k < nsplit; ++k) s += __builtin_nontemporal_load(part + ((size_t)k * M + r) * m + row);
      ep[t] = s * __half2float(scales[row]) * cb;
    }
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
      float a = 0.f, b = 0.f;
      if (t < 128) { a = ep[t]; b = ep[t ^ h]; }
      __syncthreads();
      if (t < 128) ep[t] = (t & h) ? (b - a) : (a + b);
      __syncthreads();
    }
    if (t < 128) {
      float o = ep[t] * (1.0f / 11.313708498984761f);
      if ((sv[row >> 3] >> (row & 7)) & 1) o = -o;
      if (bias != nullptr) o += to_f(bias[row]);
      y[(size_t)r * m + row] = from_f<T>(o);
    }
    __syncthreads();
  }
  if (t == 0) counters[sb] = 0u;
}

// ---------------------------------------------------------------- dequant to the ORIGINAL basis: W = Sv Ho Q Hi Su -> bf16 [m, n]
// Prefill path (M > GEMV_MAX_M): W then feeds the stock bf16 GEMM, so there is no x-prologue / fp32 GEMM output / y-epilogue.
// One WG (8 waves) per 128x128 block: decode its 8x8 tiles into LDS (f16, padded rows), column FWHT (Ho), row FWHT (Hi), signs, bf16.
#define DQW_LD 130
template <int K>
__global__ __launch_bounds__(256) void k_dequant_w(const uint32_t* __restrict__ qw, const __half* __restrict__ scales, float cb,
                                                   const uint8_t* __restrict__ su, const uint8_t* __restrict__ sv,
                                                   __nv_bfloat16* __restrict__ W, int nt) {
  __shared__ _Float16 L[128 * DQW_LD];
  const int bx = blockIdx.x, by = blockIdx.y, t = threadIdx.x, lane = t & 31, w = t >> 5;
  const int n = nt * 16;
  for (int task = t; task < 512; task += 256) {
    const int tile = task >> 3, q = task & 7, ti = tile >> 3, tj = tile & 7;
    const uint32_t* tp = qw + ((size_t)(by * 8 + ti) * nt + (bx * 8 + tj)) * (8 * K);
    uint32_t V[K + 1];
    V[0] = tp[(q * K + 8 * K - 1) % (8 * K)];
#pragma unroll
    for (int k = 0; k < K; ++k) V[k + 1] = tp[q * K + k];
    const int lr = ti * 16 + 2 * q;
    const float s0 = __half2float(scales[by * 128 + lr]) * cb, s1 = __half2float(scales[by * 128 + lr + 1]) * cb;
    _Float16* d0 = L + lr * DQW_LD + tj * 16;
    _Float16* d1 = d0 + DQW_LD;
#define HTR_DW(U) { uint32_t fb = val_f16bits(ext16<16 + ((U) + 1) * K>(V));                                \
                    float v = (float)__builtin_bit_cast(_Float16, (uint16_t)fb) - 1534.0f;                  \
                    if ((U) < 16) d0[(U) & 15] = (_Float16)(v * s0); else d1[(U) & 15] = (_Float16)(v * s1); }
    HTR_DW(0) HTR_DW(1) HTR_DW(2) HTR_DW(3) HTR_DW(4) HTR_DW(5) HTR_DW(6) HTR_DW(7)
    HTR_DW(8) HTR_DW(9) HTR_DW(10) HTR_DW(11) HTR_DW(12) HTR_DW(13) HTR_DW(14) HTR_DW(15)
    HTR_DW(16) HTR_DW(17) HTR_DW(18) HTR_DW(19) HTR_DW(20) HTR_DW(21) HTR_DW(22) HTR_DW(23)
    HTR_DW(24) HTR_DW(25) HTR_DW(26) HTR_DW(27) HTR_DW(28) HTR_DW(29) HTR_DW(30) HTR_DW(31)
#undef HTR_DW
  }
  __syncthreads();
  const float inv = 1.0f / 11.313708498984761f;
  for (int c = w; c < 128; c += 8) {                       // Ho: transform each column along the 128 rows
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = (float)L[(lane * 4 + e) * DQW_LD + c];
    fwht128_wave(v, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) L[(lane * 4 + e) * DQW_LD + c] = (_Float16)(v[e] * inv);
  }
  __syncthreads();
  for (int r = w; r < 128; r += 8) {                       // Hi: transform each row along the 128 columns, then signs
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = (float)L[r * DQW_LD + lane * 4 + e];
    fwht128_wave(v, lane);
    const int gr = by * 128 + r;
    const float sr = ((sv[gr >> 3] >> (gr & 7)) & 1) ? -inv : inv;
    __nv_bfloat16* dst = W + (size_t)gr * n + bx * 128 + lane * 4;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int gc = bx * 128 + lane * 4 + e;
      const float sc = ((su[gc >> 3] >> (gc & 7)) & 1) ? -sr : sr;
      dst[e] = __float2bfloat16(v[e] * sc);
    }
  }
}

// v5 (2026-10-16): k_dequant_w2 = k_dequant_w with packed stores (bitwise identical math): phase-1 LDS writes as f16 pairs,
// final global writes as one 8-byte store of 4 bf16 per lane (was 4 scalar 2-byte stores; the bf16 W write is 4x the qweight read).
template <int K>
__global__ __launch_bounds__(256) void k_dequant_w2(const uint32_t* __restrict__ qw, const __half* __restrict__ scales, float cb,
                                                    const uint8_t* __restrict__ su, const uint8_t* __restrict__ sv,
                                                    __nv_bfloat16* __restrict__ W, int nt) {
  __shared__ __attribute__((aligned(16))) _Float16 L[128 * DQW_LD];
  const int bx = blockIdx.x, by = blockIdx.y, t = threadIdx.x, lane = t & 31, w = t >> 5;
  const int n = nt * 16;
  for (int task = t; task < 512; task += 256) {
    const int tile = task >> 3, q = task & 7, ti = tile >> 3, tj = tile & 7;
    const uint32_t* tp = qw + ((size_t)(by * 8 + ti) * nt + (bx * 8 + tj)) * (8 * K);
    uint32_t V[K + 1];
    V[0] = tp[(q * K + 8 * K - 1) % (8 * K)];
#pragma unroll
    for (int k = 0; k < K; ++k) V[k + 1] = tp[q * K + k];
    const int lr = ti * 16 + 2 * q;
    const float s0 = __half2float(scales[by * 128 + lr]) * cb, s1 = __half2float(scales[by * 128 + lr + 1]) * cb;
    uint32_t* d0 = reinterpret_cast<uint32_t*>(L + lr * DQW_LD + tj * 16);
    uint32_t* d1 = reinterpret_cast<uint32_t*>(L + (lr + 1) * DQW_LD + tj * 16);
#define HTR_DW2(U) { uint32_t fa = val_f16bits(ext16<16 + ((U) + 1) * K>(V));                                \
                     uint32_t fb = val_f16bits(ext16<16 + ((U) + 2) * K>(V));                                \
                     const float s = (U) < 16 ? s0 : s1;                                                     \
                     _Float16 ha = (_Float16)(((float)__builtin_bit_cast(_Float16, (uint16_t)fa) - 1534.0f) * s); \
                     _Float16 hb = (_Float16)(((float)__builtin_bit_cast(_Float16, (uint16_t)fb) - 1534.0f) * s); \
                     const uint32_t pk = (uint32_t)__builtin_bit_cast(uint16_t, ha) | ((uint32_t)__builtin_bit_cast(uint16_t, hb) << 16); \
                     if ((U) < 16) d0[((U) & 15) >> 1] = pk; else d1[((U) & 15) >> 1] = pk; }
    HTR_DW2(0) HTR_DW2(2) HTR_DW2(4) HTR_DW2(6) HTR_DW2(8) HTR_DW2(10) HTR_DW2(12) HTR_DW2(14)
    HTR_DW2(16) HTR_DW2(18) HTR_DW2(20) HTR_DW2(22) HTR_DW2(24) HTR_DW2(26) HTR_DW2(28) HTR_DW2(30)
#undef HTR_DW2
  }
  __syncthreads();
  const float inv = 1.0f / 11.313708498984761f;
  for (int c = w; c < 128; c += 8) {
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = (float)L[(lane * 4 + e) * DQW_LD + c];
    fwht128_wave(v, lane);
#pragma unroll
    for (int e = 0; e < 4; ++e) L[(lane * 4 + e) * DQW_LD + c] = (_Float16)(v[e] * inv);
  }
  __syncthreads();
  for (int r = w; r < 128; r += 8) {
    float v[4];
    const uint32_t* lr32 = reinterpret_cast<const uint32_t*>(L + r * DQW_LD + lane * 4);  // 4-byte aligned (260*r + 8*lane)
    uint2 raw; raw.x = lr32[0]; raw.y = lr32[1];
    v[0] = (float)__builtin_bit_cast(_Float16, (uint16_t)(raw.x & 0xFFFF)); v[1] = (float)__builtin_bit_cast(_Float16, (uint16_t)(raw.x >> 16));
    v[2] = (float)__builtin_bit_cast(_Float16, (uint16_t)(raw.y & 0xFFFF)); v[3] = (float)__builtin_bit_cast(_Float16, (uint16_t)(raw.y >> 16));
    fwht128_wave(v, lane);
    const int gr = by * 128 + r;
    const float sr = ((sv[gr >> 3] >> (gr & 7)) & 1) ? -inv : inv;
    uint16_t o[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int gc = bx * 128 + lane * 4 + e;
      const float sc = ((su[gc >> 3] >> (gc & 7)) & 1) ? -sr : sr;
      o[e] = __builtin_bit_cast(uint16_t, __float2bfloat16(v[e] * sc));
    }
    uint2 pk; pk.x = (uint32_t)o[0] | ((uint32_t)o[1] << 16); pk.y = (uint32_t)o[2] | ((uint32_t)o[3] << 16);
    *reinterpret_cast<uint2*>(W + (size_t)gr * n + bx * 128 + lane * 4) = pk;
  }
}

// v6 (2026-10-16): k_dequant_w3 = register-blocked 2D FWHT. Same W = Sv Ho Q Hi Su as k_dequant_w (NOT bitwise: fp32 math in
// registers, different f16 rounding points, exact power-of-2 normalisation instead of 2x 1/sqrt(128)). Block row r = ti*16+2q+b,
// col c = tj*16+u; H128 (Sylvester) = H2 on every index bit, so the bits can be split over passes:
//  pass 1 (regs, after decode): u (4 bits), b, ti bit 2 (the thread's 2 tasks are ti and ti+4)  -> x1/8 -> f16 LDS (b128 stores)
//  pass 2 (regs, in place):     tj (3 bits), q (3 bits); thread = (ti, b, u), 64 values
//  pass 3 (regs):               ti bits 0-1; thread = 4 rows x 8 cols -> x1/16, signs -> one 16-B bf16 store per row (coalesced)
// k_dequant_w used ~640 ds_bpermute per wave (wave-shuffle FWHT) + 16-bit LDS traffic; this uses ~140 LDS instructions per wave.
#define DQW3_LD 144                      // halves; 72 dwords -> rows b=0/1 land on disjoint banks in pass 2; 16-B aligned rows
template <int N> __device__ __forceinline__ void fwht_reg(float* v) {      // in-register Sylvester FWHT over N (stride 1)
#pragma unroll
  for (int h = 1; h < N; h <<= 1)
#pragma unroll
    for (int i = 0; i < N; ++i)
      if ((i & h) == 0) { const float a = v[i], b = v[i + h]; v[i] = a + b; v[i + h] = a - b; }
}
template <int K>
__global__ __launch_bounds__(256) void k_dequant_w3(const uint32_t* __restrict__ qw, const __half* __restrict__ scales, float cb,
                                                    const uint8_t* __restrict__ su, const uint8_t* __restrict__ sv,
                                                    __nv_bfloat16* __restrict__ W, int nt) {
  __shared__ __attribute__((aligned(16))) _Float16 L[128 * DQW3_LD];
  const int bx = blockIdx.x, by = blockIdx.y, t = threadIdx.x;
  const int n = nt * 16;
  {  // ---- pass 1
    const int tile = t >> 3, q = t & 7, ti = tile >> 3, tj = tile & 7;     // second task: ti + 4 (task t + 256)
    float x[2][2][16];                                                      // [ti hi bit][b][u]
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const uint32_t* tp = qw + ((size_t)(by * 8 + ti + 4 * j) * nt + (bx * 8 + tj)) * (8 * K);
      uint32_t V[K + 1];
      V[0] = tp[(q * K + 8 * K - 1) % (8 * K)];
#pragma unroll
      for (int k = 0; k < K; ++k) V[k + 1] = tp[q * K + k];
      const int lr = (ti + 4 * j) * 16 + 2 * q;
      const float s0 = __half2float(scales[by * 128 + lr]) * cb, s1 = __half2float(scales[by * 128 + lr + 1]) * cb;
#define HTR_DW3(U) { uint32_t fb = val_f16bits(ext16<16 + ((U) + 1) * K>(V));                                \
                     x[j][(U) >> 4][(U) & 15] = ((float)__builtin_bit_cast(_Float16, (uint16_t)fb) - 1534.0f) * ((U) < 16 ? s0 : s1); }
      HTR_DW3(0) HTR_DW3(1) HTR_DW3(2) HTR_DW3(3) HTR_DW3(4) HTR_DW3(5) HTR_DW3(6) HTR_DW3(7)
      HTR_DW3(8) HTR_DW3(9) HTR_DW3(10) HTR_DW3(11) HTR_DW3(12) HTR_DW3(13) HTR_DW3(14) HTR_DW3(15)
      HTR_DW3(16) HTR_DW3(17) HTR_DW3(18) HTR_DW3(19) HTR_DW3(20) HTR_DW3(21) HTR_DW3(22) HTR_DW3(23)
      HTR_DW3(24) HTR_DW3(25) HTR_DW3(26) HTR_DW3(27) HTR_DW3(28) HTR_DW3(29) HTR_DW3(30) HTR_DW3(31)
#undef HTR_DW3
    }
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int b = 0; b < 2; ++b) fwht_reg<16>(x[j][b]);
#pragma unroll
    for (int u = 0; u < 16; ++u) {
      const float a00 = x[0][0][u], a01 = x[0][1][u], a10 = x[1][0][u], a11 = x[1][1][u];
      const float p0 = a00 + a01, m0 = a00 - a01, p1 = a10 + a11, m1 = a10 - a11;     // H2 over b
      x[0][0][u] = (p0 + p1) * 0.125f; x[1][0][u] = (p0 - p1) * 0.125f;                 // H2 over ti bit 2; 1/sqrt(64)
      x[0][1][u] = (m0 + m1) * 0.125f; x[1][1][u] = (m0 - m1) * 0.125f;
    }
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        union alignas(16) { _Float16 h[16]; uint4 u4[2]; } o;
#pragma unroll
        for (int u = 0; u < 16; ++u) o.h[u] = (_Float16)x[j][b][u];
        uint4* d = reinterpret_cast<uint4*>(L + ((ti + 4 * j) * 16 + 2 * q + b) * DQW3_LD + tj * 16);
        d[0] = o.u4[0]; d[1] = o.u4[1];
      }
  }
  __syncthreads();
  {  // ---- pass 2: thread (ti = wave, b, u); values [q][tj]
    const int ti = t >> 5, b = (t >> 4) & 1, u = t & 15;
    _Float16* base = L + (ti * 16 + b) * DQW3_LD + u;
    float y[8][8];
#pragma unroll
    for (int q = 0; q < 8; ++q)
#pragma unroll
      for (int tj = 0; tj < 8; ++tj) y[q][tj] = (float)base[2 * q * DQW3_LD + tj * 16];
#pragma unroll
    for (int q = 0; q < 8; ++q) fwht_reg<8>(y[q]);                         // tj
#pragma unroll
    for (int h = 1; h < 8; h <<= 1)                                         // q
#pragma unroll
      for (int q = 0; q < 8; ++q)
        if ((q & h) == 0)
#pragma unroll
          for (int tj = 0; tj < 8; ++tj) { const float a = y[q][tj], c = y[q + h][tj]; y[q][tj] = a + c; y[q + h][tj] = a - c; }
#pragma unroll
    for (int q = 0; q < 8; ++q)
#pragma unroll
      for (int tj = 0; tj < 8; ++tj) base[2 * q * DQW3_LD + tj * 16] = (_Float16)y[q][tj];
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 2; ++j) {  // ---- pass 3: task = (rowset rs = (q, b, ti bit 2), column octet co); rows ti = 4*ti2 + tl
    const int k = t + 256 * j, co = k & 15, rs = k >> 4, q = rs & 7, b = (rs >> 3) & 1, ti2 = rs >> 4;
    float z[4][8];
#pragma unroll
    for (int tl = 0; tl < 4; ++tl) {
      const uint4 raw = *reinterpret_cast<const uint4*>(L + ((4 * ti2 + tl) * 16 + 2 * q + b) * DQW3_LD + co * 8);
      const uint32_t w4[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        z[tl][2 * e] = (float)__builtin_bit_cast(_Float16, (uint16_t)(w4[e] & 0xFFFF));
        z[tl][2 * e + 1] = (float)__builtin_bit_cast(_Float16, (uint16_t)(w4[e] >> 16));
      }
    }
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const float a0 = z[0][c], a1 = z[1][c], a2 = z[2][c], a3 = z[3][c];
      const float p0 = a0 + a1, m0 = a0 - a1, p1 = a2 + a3, m1 = a2 - a3;
      z[0][c] = p0 + p1; z[1][c] = m0 + m1; z[2][c] = p0 - p1; z[3][c] = m0 - m1;
    }
    const int gc0 = bx * 128 + co * 8;
    const uint32_t sub = su[gc0 >> 3];                                      // 8 column sign bits (co*8 is byte aligned)
#pragma unroll
    for (int tl = 0; tl < 4; ++tl) {
      const int gr = by * 128 + (4 * ti2 + tl) * 16 + 2 * q + b;
      const uint32_t sr = ((sv[gr >> 3] >> (gr & 7)) & 1) ? 0xFFu : 0u;
      const uint32_t sg = sub ^ sr;
      uint32_t o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float v0 = z[tl][2 * e] * 0.0625f, v1 = z[tl][2 * e + 1] * 0.0625f;   // 1/sqrt(256)
        if ((sg >> (2 * e)) & 1) v0 = -v0;
        if ((sg >> (2 * e + 1)) & 1) v1 = -v1;
        o[e] = (uint32_t)__builtin_bit_cast(uint16_t, __float2bfloat16(v0)) | ((uint32_t)__builtin_bit_cast(uint16_t, __float2bfloat16(v1)) << 16);
      }
      uint4 pk; pk.x = o[0]; pk.y = o[1]; pk.z = o[2]; pk.w = o[3];
      *reinterpret_cast<uint4*>(W + (size_t)gr * n + gc0) = pk;
    }
  }
}

// ================================================================ host wrappers
#define DISPATCH_K(K, ...) switch (K) { \
  case 2: { constexpr int KK = 2; __VA_ARGS__; } break; case 3: { constexpr int KK = 3; __VA_ARGS__; } break; \
  case 4: { constexpr int KK = 4; __VA_ARGS__; } break; case 5: { constexpr int KK = 5; __VA_ARGS__; } break; \
  case 6: { constexpr int KK = 6; __VA_ARGS__; } break; case 8: { constexpr int KK = 8; __VA_ARGS__; } break; \
  default: TORCH_CHECK(false, "unsupported K ", K); }

torch::Tensor gemv_partials(torch::Tensor qw, torch::Tensor xt, int64_t nsplit) {
  TORCH_CHECK(qw.dim() == 3 && qw.scalar_type() == torch::kInt32 && xt.scalar_type() == torch::kHalf);
  const int mt = qw.size(0), nt = qw.size(1), K = qw.size(2) / 8, M = xt.size(0);
  TORCH_CHECK(xt.size(1) == nt * 16 && M >= 1 && M <= 8 && xt.is_contiguous() && qw.is_contiguous());
  const int tps = ((nt + nsplit - 1) / nsplit + 3) / 4 * 4;
  const int ns = (nt + tps - 1) / tps;
  auto part = torch::empty({ns, M, mt * 16}, xt.options().dtype(torch::kFloat32));
  const int waves = mt * ns, blocks = (waves + 7) / 8;
  auto st = at::cuda::getCurrentCUDAStream();
  const uint32_t* q = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  const _Float16* x = reinterpret_cast<const _Float16*>(xt.data_ptr<at::Half>());
  float* p = part.data_ptr<float>();
#define LAUNCH_M(MM) k_gemv<KK, MM><<<blocks, 256, 0, st>>>(q, x, p, mt, nt, ns, tps)
  DISPATCH_K(K, switch (M) { case 1: LAUNCH_M(1); break; case 2: LAUNCH_M(2); break; case 3: LAUNCH_M(3); break;
                             case 4: LAUNCH_M(4); break; case 5: LAUNCH_M(5); break; case 6: LAUNCH_M(6); break;
                             case 7: LAUNCH_M(7); break; default: LAUNCH_M(8); })
#undef LAUNCH_M
  return part;
}

torch::Tensor dequant_f16(torch::Tensor qw, torch::Tensor scales, double cb) {
  const int mt = qw.size(0), nt = qw.size(1), K = qw.size(2) / 8;
  auto out = torch::empty({mt * 16, nt * 16}, qw.options().dtype(torch::kHalf));
  const size_t thr = (size_t)mt * nt * 8;
  auto st = at::cuda::getCurrentCUDAStream();
  const uint32_t* q = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  DISPATCH_K(K, k_dequant<KK><<<(thr + 255) / 256, 256, 0, st>>>(q, reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
                                                                 (float)cb, reinterpret_cast<_Float16*>(out.data_ptr<at::Half>()), mt, nt))
  return out;
}

torch::Tensor decode_raw(torch::Tensor qw) {
  const int K = qw.size(-1) / 8;
  const int ntiles = qw.numel() / (8 * K);
  auto out = torch::empty({ntiles, 256}, qw.options().dtype(torch::kInt16));
  auto st = at::cuda::getCurrentCUDAStream();
  const uint32_t* q = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  DISPATCH_K(K, k_decode_raw<KK><<<((size_t)ntiles * 8 + 255) / 256, 256, 0, st>>>(q, out.data_ptr<int16_t>(), ntiles))
  return out;
}

template <typename T>
static void launch_pro(int M, cudaStream_t st, const T* x, const uint8_t* su, _Float16* o, float* xs, float* tok, int n) {
  const int nb = ((n >> 7) + 7) / 8;     // 128-blocks per wave
  static const bool one_pass = !(std::getenv("HTR_PRO1P") && std::getenv("HTR_PRO1P")[0] == '0');
  if (one_pass && nb <= 1) {
    k_prologue1p<T, 1><<<M, 256, 0, st>>>(x, su, o, xs, tok, n);
  } else if (one_pass && nb <= 2) {
    k_prologue1p<T, 2><<<M, 256, 0, st>>>(x, su, o, xs, tok, n);
  } else if (one_pass && nb <= 4) {
    k_prologue1p<T, 4><<<M, 256, 0, st>>>(x, su, o, xs, tok, n);
  } else if (one_pass && nb <= 8) {
    k_prologue1p<T, 8><<<M, 256, 0, st>>>(x, su, o, xs, tok, n);
  } else if (one_pass && nb <= 12) {
    k_prologue1p<T, 12><<<M, 256, 0, st>>>(x, su, o, xs, tok, n);
  } else {
    k_prologue<T><<<M, 256, 0, st>>>(x, su, o, xs, tok, n);
  }
}

std::vector<torch::Tensor> prologue(torch::Tensor x, torch::Tensor su) {
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && x.size(1) % 128 == 0);
  const int M = x.size(0), n = x.size(1);
  auto xt = torch::empty({M, n}, x.options().dtype(torch::kHalf));
  auto xs = torch::empty({M}, x.options().dtype(torch::kFloat32));
  auto tok = torch::empty({M}, x.options().dtype(torch::kFloat32));
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  _Float16* o = reinterpret_cast<_Float16*>(xt.data_ptr<at::Half>());
  if (x.scalar_type() == torch::kBFloat16)
    launch_pro<__nv_bfloat16>(M, st, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), su.data_ptr<uint8_t>(), o, xs.data_ptr<float>(), tok.data_ptr<float>(), n);
  else if (x.scalar_type() == torch::kHalf)
    launch_pro<__half>(M, st, reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), su.data_ptr<uint8_t>(), o, xs.data_ptr<float>(), tok.data_ptr<float>(), n);
  else
    launch_pro<float>(M, st, x.data_ptr<float>(), su.data_ptr<uint8_t>(), o, xs.data_ptr<float>(), tok.data_ptr<float>(), n);
  return {xt, xs, tok};
}

torch::Tensor epilogue(torch::Tensor part, torch::Tensor xs, torch::Tensor tok, torch::Tensor scales, double cb, torch::Tensor sv,
                       c10::optional<torch::Tensor> bias, int64_t mode, int64_t out_dtype) {
  int nsplit, M, m;
  if (mode == 0) { nsplit = part.size(0); M = part.size(1); m = part.size(2); }
  else { nsplit = 1; M = part.size(0); m = part.size(1); }
  TORCH_CHECK(m % 128 == 0);
  auto st = at::cuda::getCurrentCUDAStream();
  dim3 grid(m / 128, M);
  const __half* sc = mode == 0 ? reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()) : nullptr;
  const float* xsp = mode == 0 ? xs.data_ptr<float>() : nullptr;
  if (out_dtype == 0) {
    auto y = torch::empty({M, m}, part.options().dtype(torch::kBFloat16));
    const __nv_bfloat16* b = bias.has_value() ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr<at::BFloat16>()) : nullptr;
    k_epilogue<__nv_bfloat16><<<grid, 128, 0, st>>>(part.data_ptr<float>(), nsplit, M, xsp, tok.data_ptr<float>(), sc, (float)cb,
        sv.data_ptr<uint8_t>(), b, reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), m, (int)mode);
    return y;
  } else if (out_dtype == 1) {
    auto y = torch::empty({M, m}, part.options().dtype(torch::kHalf));
    const __half* b = bias.has_value() ? reinterpret_cast<const __half*>(bias->data_ptr<at::Half>()) : nullptr;
    k_epilogue<__half><<<grid, 128, 0, st>>>(part.data_ptr<float>(), nsplit, M, xsp, tok.data_ptr<float>(), sc, (float)cb,
        sv.data_ptr<uint8_t>(), b, reinterpret_cast<__half*>(y.data_ptr<at::Half>()), m, (int)mode);
    return y;
  }
  auto y = torch::empty({M, m}, part.options().dtype(torch::kFloat32));
  const float* b = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  k_epilogue<float><<<grid, 128, 0, st>>>(part.data_ptr<float>(), nsplit, M, xsp, tok.data_ptr<float>(), sc, (float)cb,
      sv.data_ptr<uint8_t>(), b, y.data_ptr<float>(), m, (int)mode);
  return y;
}

// plain (non-macro) launchers: hipify cannot wrap <<<>>> with template commas inside a macro body
template <int KK, int MM, typename T>
static void launch_fused(dim3 grid, cudaStream_t st, const uint32_t* q, const T* x, const uint8_t* su, const uint8_t* sv,
                         const __half* sc, float cb, const T* bias, float* part, unsigned* cnt, T* y, int m, int n, int bps, int mr) {
  k_gemv_fused<KK, MM, T><<<grid, 256, 0, st>>>(q, x, su, sv, sc, cb, bias, part, cnt, y, m, n, bps, mr);
}
template <typename T>
static void fused_dispatch(int K, int M, dim3 grid, cudaStream_t st, const uint32_t* q, const T* x, const uint8_t* su, const uint8_t* sv,
                           const __half* sc, float cb, const T* bias, float* part, unsigned* cnt, T* y, int m, int n, int bps) {
  if (M == 1) { DISPATCH_K(K, launch_fused<KK, 1, T>(grid, st, q, x, su, sv, sc, cb, bias, part, cnt, y, m, n, bps, M)) }
  else if (M == 2) { DISPATCH_K(K, launch_fused<KK, 2, T>(grid, st, q, x, su, sv, sc, cb, bias, part, cnt, y, m, n, bps, M)) }
  else if (M <= 4) { DISPATCH_K(K, launch_fused<KK, 4, T>(grid, st, q, x, su, sv, sc, cb, bias, part, cnt, y, m, n, bps, M)) }
  else { DISPATCH_K(K, launch_fused<KK, 8, T>(grid, st, q, x, su, sv, sc, cb, bias, part, cnt, y, m, n, bps, M)) }
}

torch::Tensor gemv_fused(torch::Tensor qw, torch::Tensor x, torch::Tensor su, torch::Tensor sv, torch::Tensor scales, double cb,
                         c10::optional<torch::Tensor> bias, torch::Tensor counters, int64_t nsplit) {
  TORCH_CHECK(qw.dim() == 3 && qw.is_contiguous() && x.is_contiguous() && x.dim() == 2);
  const int mt = qw.size(0), nt = qw.size(1), K = qw.size(2) / 8, M = x.size(0), m = mt * 16, n = nt * 16;
  TORCH_CHECK(x.size(1) == n && M >= 1 && M <= 8 && m % 128 == 0 && n % 128 == 0 && counters.numel() >= m / 128);
  const int nbt = n / 128;
  int ns = std::max<int>((nbt + FUSED_MAXB - 1) / FUSED_MAXB, std::min<int>(nbt, (int)nsplit));
  const int bps = (nbt + ns - 1) / ns;
  ns = (nbt + bps - 1) / bps;
  const int Mt = M == 1 ? 1 : M == 2 ? 2 : M <= 4 ? 4 : 8;
  auto part = torch::empty({ns, Mt, m}, x.options().dtype(torch::kFloat32));
  auto y = torch::empty({M, m}, x.options());
  auto st = at::cuda::getCurrentCUDAStream();
  dim3 grid(m / 128, ns);
  const uint32_t* q = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  const __half* sc = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  unsigned* cnt = reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>());
  const uint8_t* sup = su.data_ptr<uint8_t>();
  const uint8_t* svp = sv.data_ptr<uint8_t>();
  float* pp = part.data_ptr<float>();
  if (x.scalar_type() == torch::kBFloat16)
    fused_dispatch<__nv_bfloat16>(K, M, grid, st, q, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), sup, svp, sc, (float)cb,
                                  bias.has_value() ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr<at::BFloat16>()) : nullptr,
                                  pp, cnt, reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), m, n, bps);
  else if (x.scalar_type() == torch::kHalf)
    fused_dispatch<__half>(K, M, grid, st, q, reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), sup, svp, sc, (float)cb,
                           bias.has_value() ? reinterpret_cast<const __half*>(bias->data_ptr<at::Half>()) : nullptr,
                           pp, cnt, reinterpret_cast<__half*>(y.data_ptr<at::Half>()), m, n, bps);
  else TORCH_CHECK(false, "gemv_fused: x must be bf16/f16");
  return y;
}

template <int KK>
static void launch_dqw(dim3 grid, cudaStream_t st, const uint32_t* q, const __half* sc, float cb, const uint8_t* su, const uint8_t* sv,
                       __nv_bfloat16* W, int nt) {
  // k_dequant_w2 is bitwise identical but measured neutral (dev/test_v5_dqw{0,1}.json, -3%..+4%) -> opt-in only (HTR_DQW2=1)
  // v6: HTR_DQW selects the kernel (1 = k_dequant_w, 2 = k_dequant_w2, 3 = k_dequant_w3); legacy HTR_DQW2=1 -> 2. Default 1 until v6 qualifies.
  static const int mode = std::getenv("HTR_DQW") ? std::atoi(std::getenv("HTR_DQW"))
                          : ((std::getenv("HTR_DQW2") && std::getenv("HTR_DQW2")[0] == '1') ? 2 : 1);
  if (mode == 3) {
    k_dequant_w3<KK><<<grid, 256, 0, st>>>(q, sc, cb, su, sv, W, nt);
  } else if (mode == 2) {
    k_dequant_w2<KK><<<grid, 256, 0, st>>>(q, sc, cb, su, sv, W, nt);
  } else {
    k_dequant_w<KK><<<grid, 256, 0, st>>>(q, sc, cb, su, sv, W, nt);
  }
}

torch::Tensor dequant_w(torch::Tensor qw, torch::Tensor scales, double cb, torch::Tensor su, torch::Tensor sv) {
  TORCH_CHECK(qw.dim() == 3 && qw.is_contiguous() && qw.scalar_type() == torch::kInt32);
  const int mt = qw.size(0), nt = qw.size(1), K = qw.size(2) / 8;
  TORCH_CHECK(mt % 8 == 0 && nt % 8 == 0, "dequant_w needs m, n % 128 == 0");
  TORCH_CHECK(scales.numel() == mt * 16 && su.numel() == nt * 2 && sv.numel() == mt * 2);
  auto W = torch::empty({mt * 16, nt * 16}, qw.options().dtype(torch::kBFloat16));
  auto st = at::cuda::getCurrentCUDAStream();
  dim3 grid(nt / 8, mt / 8);
  const uint32_t* q = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  const __half* sc = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  __nv_bfloat16* Wp = reinterpret_cast<__nv_bfloat16*>(W.data_ptr<at::BFloat16>());
  DISPATCH_K(K, launch_dqw<KK>(grid, st, q, sc, (float)cb, su.data_ptr<uint8_t>(), sv.data_ptr<uint8_t>(), Wp, nt))
  return W;
}


static thread_local bool EPI8_MODE = false;
template <int K, int M, typename TO>
static void launch_epi(int blocks, cudaStream_t st, const uint32_t* q, const _Float16* x, float* p, unsigned* cnt, const float* xs,
                       const float* tok, const __half* sc, float cb, const uint8_t* svp, const TO* b, TO* y, int mt, int nt,
                       int ns, int tps) {
  // hipify pitfall: keep each <<<>>> launch as its own statement inside braces ("else k<<<" became "elsehipLaunchKernelGGL")
  if (EPI8_MODE) {
    k_gemv_epi8<K, M, TO><<<blocks, 256, 0, st>>>(q, x, p, cnt, xs, tok, sc, cb, svp, b, y, mt, nt, ns, tps);
  } else {
    k_gemv_epi<K, M, TO><<<blocks, 256, 0, st>>>(q, x, p, cnt, xs, tok, sc, cb, svp, b, y, mt, nt, ns, tps);
  }
}

template <int K, typename TO>
static void epi_dispatch_m(int M, int blocks, cudaStream_t st, const uint32_t* q, const _Float16* x, float* p, unsigned* cnt,
                           const float* xs, const float* tok, const __half* sc, float cb, const uint8_t* svp, const TO* b, TO* y,
                           int mt, int nt, int ns, int tps) {
  switch (M) {
    case 1: launch_epi<K, 1, TO>(blocks, st, q, x, p, cnt, xs, tok, sc, cb, svp, b, y, mt, nt, ns, tps); break;
    case 2: launch_epi<K, 2, TO>(blocks, st, q, x, p, cnt, xs, tok, sc, cb, svp, b, y, mt, nt, ns, tps); break;
    case 3: launch_epi<K, 3, TO>(blocks, st, q, x, p, cnt, xs, tok, sc, cb, svp, b, y, mt, nt, ns, tps); break;
    case 4: launch_epi<K, 4, TO>(blocks, st, q, x, p, cnt, xs, tok, sc, cb, svp, b, y, mt, nt, ns, tps); break;
    case 5: launch_epi<K, 5, TO>(blocks, st, q, x, p, cnt, xs, tok, sc, cb, svp, b, y, mt, nt, ns, tps); break;
    case 6: launch_epi<K, 6, TO>(blocks, st, q, x, p, cnt, xs, tok, sc, cb, svp, b, y, mt, nt, ns, tps); break;
    case 7: launch_epi<K, 7, TO>(blocks, st, q, x, p, cnt, xs, tok, sc, cb, svp, b, y, mt, nt, ns, tps); break;
    default: launch_epi<K, 8, TO>(blocks, st, q, x, p, cnt, xs, tok, sc, cb, svp, b, y, mt, nt, ns, tps);
  }
}

template <typename TO>
static void epi_dispatch(int K, int M, int blocks, cudaStream_t st, const uint32_t* q, const _Float16* x, float* p, unsigned* cnt,
                         const float* xs, const float* tok, const __half* sc, float cb, const uint8_t* svp, const TO* b, TO* y,
                         int mt, int nt, int ns, int tps) {
  DISPATCH_K(K, epi_dispatch_m<KK, TO>(M, blocks, st, q, x, p, cnt, xs, tok, sc, cb, svp, b, y, mt, nt, ns, tps))
}

torch::Tensor gemv_epi(torch::Tensor qw, torch::Tensor xt, torch::Tensor xs, torch::Tensor tok, torch::Tensor scales, double cb,
                       torch::Tensor sv, c10::optional<torch::Tensor> bias, torch::Tensor counters, int64_t nsplit,
                       int64_t out_dtype, int64_t wg_reduce) {
  TORCH_CHECK(qw.dim() == 3 && qw.scalar_type() == torch::kInt32 && xt.scalar_type() == torch::kHalf);
  const int mt = qw.size(0), nt = qw.size(1), K = qw.size(2) / 8, M = xt.size(0);
  TORCH_CHECK(xt.size(1) == nt * 16 && M >= 1 && M <= 8 && xt.is_contiguous() && qw.is_contiguous() && mt % 8 == 0);
  TORCH_CHECK(counters.numel() >= mt / 8 && counters.scalar_type() == torch::kInt32);
  const int tps = ((nt + nsplit - 1) / nsplit + 3) / 4 * 4;
  const int ns = (nt + tps - 1) / tps;
  TORCH_CHECK(!wg_reduce || ns % 8 == 0, "gemv_epi wg_reduce needs nsplit % 8 == 0, got ", ns);
  EPI8_MODE = wg_reduce != 0;
  auto part = torch::empty({wg_reduce ? ns / 8 : ns, M, mt * 16}, xt.options().dtype(torch::kFloat32));
  const int waves = mt * ns, blocks = (waves + 7) / 8;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const uint32_t* q = reinterpret_cast<const uint32_t*>(qw.data_ptr<int32_t>());
  const _Float16* x = reinterpret_cast<const _Float16*>(xt.data_ptr<at::Half>());
  float* p = part.data_ptr<float>();
  unsigned* cnt = reinterpret_cast<unsigned*>(counters.data_ptr<int32_t>());
  const __half* sc = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  const uint8_t* svp = sv.data_ptr<uint8_t>();
  const float* xsp = xs.data_ptr<float>();
  const float* tkp = tok.data_ptr<float>();
  if (out_dtype == 0) {
    auto y = torch::empty({M, mt * 16}, xt.options().dtype(torch::kBFloat16));
    const __nv_bfloat16* b = bias.has_value() ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr<at::BFloat16>()) : nullptr;
    epi_dispatch<__nv_bfloat16>(K, M, blocks, st, q, x, p, cnt, xsp, tkp, sc, (float)cb, svp, b,
                                reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), mt, nt, ns, tps);
    return y;
  } else if (out_dtype == 1) {
    auto y = torch::empty({M, mt * 16}, xt.options().dtype(torch::kHalf));
    const __half* b = bias.has_value() ? reinterpret_cast<const __half*>(bias->data_ptr<at::Half>()) : nullptr;
    epi_dispatch<__half>(K, M, blocks, st, q, x, p, cnt, xsp, tkp, sc, (float)cb, svp, b,
                         reinterpret_cast<__half*>(y.data_ptr<at::Half>()), mt, nt, ns, tps);
    return y;
  }
  auto y = torch::empty({M, mt * 16}, xt.options().dtype(torch::kFloat32));
  const float* b = bias.has_value() ? bias->data_ptr<float>() : nullptr;
  epi_dispatch<float>(K, M, blocks, st, q, x, p, cnt, xsp, tkp, sc, (float)cb, svp, b, y.data_ptr<float>(), mt, nt, ns, tps);
  return y;
}
