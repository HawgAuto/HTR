# HTR methodology: Halo Trellis

HTR is a Strix Halo–oriented member of the rotated, Hessian-aware trellis-quantization family. It combines ideas learned by studying exlr8, ExLlamaV3/EXL3, QTIP, QuIP/QuIP#, and GPTQ/LDLQ. Its contribution is the particular codec, packing, tensor policy, and AMD execution integration described here—not the invention of rotation, trellis coding, Viterbi search, or second-order quantization. Read [LINEAGE.md](LINEAGE.md) alongside this document.

## 1. What problem are we solving?

At low batch size, an integrated GPU often spends much of a decode step reading weights from shared memory. Reducing the stored bytes can reduce that traffic and let a larger checkpoint fit. But it helps only if decoding the compressed representation, rotating activations, and scheduling the work do not consume the savings.

Our objective is therefore not simply the smallest file or the highest nominal bit depth. It is a useful combination of:

- fidelity to the source model, measured on held-out data;
- actual packed bytes, including scales, signs, padding, and untouched tensors;
- bytes read by the active dense and routed matrices per token;
- end-to-end prefill and generation behavior on the intended hardware.

HTR is lossy weight quantization. “Precision” in our comparisons means measured reconstruction/model fidelity at a storage budget, not recovering BF16 exactly or doing native three-/four-bit floating-point arithmetic.

## 2. Rotate before quantizing

For a matrix `W` with shape `[m output, n input]`, the W1 encoder uses block-diagonal, normalized 128-point Hadamard transforms on both axes, with reproducible random sign vectors:

    W_rot = H_out · diag(sv) · W · diag(su) · H_in
    W_hat = diag(sv) · H_out · Q_hat · H_in · diag(su)

Here the Hadamard blocks are symmetric and orthogonal. Without quantization, the forward/inverse transforms preserve the matrix up to floating-point rounding. They spread concentrated/outlier energy across coordinates so that a common codebook is more useful. This is statistical conditioning, not a guarantee that every tensor becomes Gaussian.

The input Hessian is transformed into the same coordinates. Fused projections that consume the same activation share input signs and a compatible width. That is important for a runtime which concatenates Q/K/V or gate/up matrices: incompatible input transforms cannot be silently fused.

W1 requires both dimensions to be divisible by 128. The generic checkpoint encoder leaves unsupported dimensions and excluded tensor families in their source representation. An escape is a deliberate retained-precision tensor, not a failed quantization silently mislabeled as HTR.

## 3. Calibration and block-LDLQ error feedback

Calibration observes the real model's inputs to each target linear and estimates

    H = E[x xᵀ]
    output-error proxy = tr((W - W_hat) H (W - W_hat)ᵀ).

This proxy values errors along frequently used input directions more heavily. It is not an exact task loss or a substitute for full-model evaluation.

The implementation regularizes the rotated Hessian, handles dead input directions, and uses a Cholesky-derived block-LDL factorization with 16-column blocks. It walks from the last input block toward the first, adding feedback from already-encoded blocks to the next block's target. The feedback lets errors in correlated directions partly compensate one another.

Within a tile, the released dense encoder's Viterbi objective uses the per-row scale-squared weighting; it does not solve an arbitrary full covariance objective inside every 16×16 tile. This approximation and the blockwise feedback should not be described as a globally optimal quantizer.

Calibration is part of the artifact identity: source checkpoint, tensor selection, corpus, windows, seeds, damping, group construction, and widths all matter. Encoding the same model with different calibration is a different experiment. The low-bit module metadata is not a promise that generic calibration reproduces a published model byte for byte.

## 4. Trellis coding: a sequence, not independent scalar levels

A tile contains 16×16 weights, read in output-row-major order. Its packed stream contains exactly `256 × K` bits. A 16-bit state evolves as

    s[t] = ((s[t-1] << K) & 0xffff) | new_bits[t].

Each reconstructed weight depends on a sliding 16-bit window of the circular stream. Adjacent weights share state, so they cannot be selected independently. Viterbi search chooses a low-error valid sequence rather than independently picking one of `2^K` scalar levels for each weight.

A large state space does not imply storing 16 bits per weight. Only K new bits per step are stored. Conversely, many reachable states do not mean every weight can independently choose any codebook entry.

The two-pass tail-biting implementation first estimates a boundary state, then reruns with a compatible boundary constraint. This closes the circular stream without storing an initial state for each tile. It is a practical boundary-selection procedure, not an exhaustive proof of the globally best tail-biting boundary.

## 5. RDNA24 procedural codebook

W1 uses the following fixed integer generator:

    x = (s × 0xDF6ADF + 0x34983) mod 2^32
    v(s) = sum_of_four_unsigned_bytes(x) - 510
    cb(s) = float32(v(s) / RMS_over_all_65536_states(v)).

The encoder normalizes each rotated output row and stores its scale as FP16. The decoder reconstructs the selected codebook value and applies the row scale.

This generator was selected to fit RDNA-oriented integer multiply/add and byte-sum operations. It avoids a large per-weight codebook load in the fused decoder. Unlike exlr8's FP8-E4M3-rounded path, W1 does not require FP8 tensor-core execution. Unlike an INT4 scalar code, a K4 trellis stream does not decode to only 16 independent integer values.

The generator is an integer-lattice construction before normalization, but this does not make the serving operation “native INT4” or “native FP4.” Small-row HTR kernels reconstruct values and use floating-point dot-pair arithmetic and accumulation. The compressed representation, reconstruction precision, activation precision, and arithmetic instruction are distinct.

Current EXL3 also has a byte-sum codebook and RDNA-specific kernels. Our historical comparison to an older MCG decoder is not a benchmark of current EXL3. No current-EXL3 superiority claim is made.

## 6. Exact W1 storage and bits per weight

Each quantized `[m,n]` matrix contains:

| Tensor | Stored type | Shape | Purpose |
|---|---|---|---|
| `htr_qweight` | INT32 bit patterns | `[m/16,n/16,8K]` | MSB-first circular trellis streams |
| `htr_scales` | FP16 | `[m]` | One rotated-row scale |
| `htr_su` | UINT8 | `[n/8]` | Packed input signs; LSB-first; bit 1 means −1 |
| `htr_sv` | UINT8 | `[m/8]` | Packed output signs; same convention |

The payload cost, before safetensors headers or runtime repacking, is

    bytes = m*n*K/8 + 2*m + n/8 + m/8
    effective bpw = K + 16/n + 1/m + 1/n.

Signs and scales are not free. Their overhead is small for large matrices, but whole-checkpoint bitrate also includes embeddings, norms, small escaped projections, vision components if retained, metadata, and any draft or lookup tables. A report's `target_bpw` must never be presented as whole-model bitrate.

K2–K5 is the practical HTR family selected by this work. The low-level W1 routines can represent additional integer widths, but K6/K8 experiments did not justify making them recommended products; higher-precision scalar/GPTQ escapes can be preferable. Wider trellis coding is not automatically better than a good scalar quantizer at the same storage budget.

See [FORMAT.md](FORMAT.md) for the serialization contract.

## 7. Tensor policy matters as much as the codec

The release candidates use different policies for different roles:

- MiMo-9B: selected dense linears at K3 or K4, with the output head at K4; excluded tensors remain at source precision.
- Qwen3.8-Flash-Next: selected dense trunk at K5, output head at K4, routed experts at K4, and hyperconnections on a separate GPTQ8-HC wire. It is a composite model, not “everything at four bits.”

These are measured model-specific decisions, not universal width prescriptions. In particular, a small local reconstruction error can conceal downstream sensitivity in hyperconnections or the output head. Keep model KL/PPL and representative task behavior in the loop.

For routed experts, static size and active bytes are different. Giving the hottest experts more bits can shrink the file relative to one baseline while increasing bytes read per token relative to another. Mixed-width allocation therefore needs both a storage budget and a routing-weighted traffic ledger. The current Qwen release candidate uses uniform K4 routed experts; a general multi-width optimizer is not claimed as a completed feature of this source release.

## 8. Execution on Strix Halo

### Decode / few rows

The source includes HIP-oriented packed GEMV kernels, block-Hadamard activation transforms, shape-specific split-K/fused paths, and fused epilogues. Compressed streams remain in memory; decoded values are reconstructed inside the consuming kernel. Avoiding a persistent whole-model BF16 expansion is essential to retaining the bandwidth advantage.

### Prefill / many rows

More rows change the best strategy. The included dense runtime generally reconstructs a matrix into its original BF16 basis and calls GEMM for larger-row prefill; short multirow calls can use GEMV chunks. The separately developed routed-MoE runtime uses dedicated packed expert layouts and prefill kernels. It is not included merely because the encoder can emit expert packets.

Both paths dequantize. “No whole-weight expansion on the decode hot path” is accurate; “no dequantization” is not. A decoder that saves bytes during generation can still regress prefill. The measured MiMo table shows that tradeoff rather than hiding it.

### Runtime integration is part of the result

HTR-W1 is not GGUF, AWQ, GPTQ, or EXL3-compatible storage. Stock runtimes do not gain support by changing a model label. The supplied vLLM quantization plugin and native sources are an experimental integration snapshot; Qwen3.8's final MoE/HC/PLE/speculation runtime must be pinned and qualified separately.

A package import, successful CPU round trip, or compiled HIP extension is not a complete model-serving test. This release preparation performs CPU packaging checks and retains historical model evidence without interrupting ongoing native qualification.

## 9. What the evidence does—and does not—show

[Evidence tables](EVIDENCE.md) distinguish three separate questions:

1. Codec screen: same target tensors and BF16 teacher, comparing HTR with local scalar/GGML controls. This measures a real fidelity-per-bit signal.
2. Packed artifact check: the actual MiMo K3/K4 + K4-head artifacts versus their BF16 teacher and fake-quantized reconstruction.
3. Serving: historical single-sequence, fixed-output-length wall-generation rates and separate prompt-processing rates on the Radeon 8060S.

Do not combine a codec screen's lower KL with a different artifact's speed to advertise a single matched product result. KL here is top-128-plus-rest coarsened teacher KL, a lower bound on full-vocabulary KL; it is not accuracy. PPL can move in a different direction, and neither metric establishes tool reliability, coding quality, multimodal fidelity, or every language/domain.

The demonstrated benefit is lower measured teacher divergence at fewer target bits than specific tested controls, plus substantial historical generation improvement over BF16 on the MiMo deployment. It is not lossless compression, an all-workload speedup, a claim against every GPTQ implementation, or an established win over current EXL3/exlr8.
