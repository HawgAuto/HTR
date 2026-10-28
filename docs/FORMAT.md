# HTR-W1 packed weight wire (version 1)

Scope: one dense linear y = W x, W [m out, n in], bias untouched. Requirements: n % 128 == 0, m % 128 == 0
(both-side RHT). Tensors failing this are ESCAPES (kept in source dtype) in v1.

## Math (encoder uses both-side rotation and unit codebook gain)
  W = Sv · Ho · Q · Hi · Su            (Hi, Ho = blockdiag(H128 / sqrt(128)), Su = diag(su), Sv = diag(sv), signs ±1)
  y = sv ⊙ Ho( Q · Hi(su ⊙ x) )
  Q[i, j] = cb[s(i, j)] * float32(scales[i])          (float32 multiply)
  cb[s]   = float32( v(s) / sqrt(mean_{s'} v(s')^2) )  (mean over all 65536 states, float64)
  v(s)    = bytesum((s * 0xDF6ADF + 0x34983) mod 2^32) - 510      integer in [-510, 510]  ("rdna24" generator)
Kernels may compute y~_i = (float32(scales[i]) * CBSCALE) * sum_j v(s(i,j)) * x~_j with CBSCALE = 1/sqrt(mean v^2);
this differs from the reference only by float rounding.

## Trellis / tiles
- Tile = 16 output rows x 16 input cols of Q. Tile (ti, tj) covers rows 16ti.., cols 16tj..
- Inside a tile, step t = 16*r + c (r = row in tile, c = col in tile). L = 16-bit state, K bits per step:
  s_t = ((s_{t-1} << K) & 0xFFFF) | b_t ; tail-biting: s_{-1} = s_255.
- Bitstream = b_0 b_1 ... b_255 (each K bits, most-significant bit first), 256*K bits, CIRCULAR.
  s_t = the 16 stream bits ending at the last bit of b_t, wrapping to the end of the stream for t*K+K < 16.
- Word packing: stream bit p -> word p >> 5, bit (31 - (p & 31)) (MSB-first). 8*K uint32 words per tile.
- GEMV note: row r of a tile owns stream bits [16rK, 16rK+16K) plus the 16-K history bits that end row r-1
  (row 15 for r = 0) -> lane r can get its history from lane (r-1) mod 16 via DPP row_ror:1.

## Tensors per quantized linear `<prefix>`
  <prefix>.htr_qweight  int32  [m/16, n/16, 8*K]   (uint32 bit patterns stored as int32)
  <prefix>.htr_scales   float16 [m]
  <prefix>.htr_su       uint8  [n/8]   bit j%8 (LSB-first) of byte j/8 set -> su[j] = -1
  <prefix>.htr_sv       uint8  [m/8]   same convention for sv
K = htr_qweight.shape[-1] / 8. Allowed K: 2..8 (K=8 has 8 history bits).
Matrices fused by a runtime along the output dim (q/k/v, gate/up) MUST share su and K (encoder guarantees this
for a declared input group); their qweight/scales/sv concatenate along dim 0.

## Checkpoint metadata (config.json quantization_config)
  {"quant_method": "htr", "wire": "HTR-W1", "generator": "rdna24", "L": 16, "tile": 16, "rht_block": 128,
   "modules": {<prefix>: {"K": k}}, "escapes": [<prefix>...]}

The recommended research family is K2–K5. The broader low-level encoding range is not a claim of qualification or quality superiority at every width. See METHODOLOGY.md for true BPW and the tail-biting approximation.
