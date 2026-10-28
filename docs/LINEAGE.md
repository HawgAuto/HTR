# Lineage, acknowledgments, and scope of our contribution

HTR grew out of studying **[kindlingai/exlr8](https://github.com/kindlingai/exlr8)** and bringing its format-and-engine co-design philosophy to AMD Strix Halo. exlr8 explicitly credits **[Turboderp's ExLlamaV3 / EXL3](https://github.com/turboderp-org/exllamav3)** for its trellis codec, and EXL3 identifies its relationship to **QTIP**. We gratefully acknowledge those authors and the research behind them. HTR builds on that foundation with its own RDNA-oriented codebook, packed format, and execution path.

[Jump to the technical comparison: HTR versus EXL3 →](#htr-and-exl3-what-actually-differs)

A compact view of the intellectual lineage is:

    second-order quantization / GPTQ / LDLQ
                  +
    incoherence processing / QuIP / QuIP#
                  +
    trellis-coded quantization / QTIP
                  ↓
             ExLlamaV3 / EXL3
                  ↓
          exlr8 (direct inspiration)
                  ↓
     HTR: Halo Trellis / Strix Halo specialization

These arrows describe ideas and engineering study. They are not a Git ancestry graph, a complete history of the research field, or a claim that every project copied source from its predecessor. HTR is not an official release of any of those projects, and no endorsement is implied.

## What we learned, and what HTR changes

| Source / lineage | Ideas studied or inherited | HTR treatment |
|---|---|---|
| exlr8 / kindlingai | Format and engine co-design; low bytes-per-token objective; shared rotations; multiple encodes and explicit allocation manifests; explaining the codec openly | Direct starting inspiration. HTR targets a single Strix Halo system, applies the policy to dense trunk as well as routed experts, and does not transplant DGX Spark FP8 or multi-node sharding assumptions. |
| ExLlamaV3 / EXL3 / Turboderp and contributors | Practical low-bit trellis quantization, 16-bit-state bitshift coding, 16×16 tiles, Viterbi/tail-biting, transformed coordinates, calibration and serving integration | Same broad quantization family. HTR-W1 chooses its own RDNA24 generator, sign/scale metadata, packed wire and runtime integration. It is not EXL3 checkpoint-compatible. |
| QTIP — Albert Tseng, Qingyao Sun, David Hou, Christopher De Sa | Trellis-coded quantization combined with incoherence processing; procedural/lookup-free codebooks and efficient bitshift trellises | Fundamental methodological credit. HTR does not claim to originate high-dimensional trellis coding or the decoupling of state count from bits per weight. |
| QuIP and QuIP# — research by Jerry Chee, Albert Tseng and collaborators | Incoherence processing, randomized Hadamard transforms, and principled low-bit quantization | Credit for the transform/conditioning and quantization lineage. W1 is not QuIP#'s E8 lattice format or its complete fine-tuning procedure. |
| GPTQ — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh; related LDLQ work | Calibration-derived second-order reconstruction objectives and blockwise error feedback | HTR uses a related block-LDLQ formulation around its trellis search. “Hessian-aware” is not a claim to have invented second-order quantization. |
| AMD ROCm / HIP / hipBLASLt / LLVM; RDNA community | Compiler, device primitives, math libraries, hardware knowledge and profiling | HTR's kernels and runtime build on this ecosystem. We distinguish packed storage from actual reconstructed operands and executed arithmetic. |
| vLLM, PyTorch, Hugging Face Transformers, Safetensors, Triton | Model execution, tensor operations, checkpoint serialization, plugin infrastructure and runtime kernels | Dependencies/integration infrastructure, not HTR inventions. Their licenses remain separate. |
| llama.cpp / GGML contributors | Mature scalar and importance-aware control formats used in codec screens | Controls are credited by their original quant names. Their measurements do not imply a direct runtime head-to-head. |
| Qwen and Xiaomi MiMo | Source model architecture, pretrained weights, and MiMo's SFT/distillation | HTR is a quantized derivative, not the creator of these models. Model licenses and provenance remain attached to each model card. |

## HTR and EXL3: what actually differs

**HTR combines a different numerical codebook, an incompatible checkpoint format, and an RDNA-oriented reconstruction strategy.** Those differences are visible in the stored tensors, reconstructed weights, and generated instructions—not just in project names.

Both projects belong to the same broad family: incoherence transforms, Hessian-informed error feedback, and trellis-coded quantization. HTR's contribution is the particular codebook/format/kernel co-design for Strix Halo, rather than the invention of a new quantization family.

**Comparison scope:** the published HTR-W1 source at [`84e90df`](https://github.com/HawgAuto/HTR/tree/84e90dfdb15e42b01c686ad713aaac268db4f96a), versus EXL3's `mul1` codebook and optimized RDNA tiles-GEMV path at [`cdc785d`](https://github.com/turboderp-org/exllamav3/tree/cdc785db89e263e65e5849cb53e1ada156411414). EXL3 also supports other codebooks; the instruction comparison below is specifically against this `mul1` path.

| Design choice | HTR-W1 / RDNA24 | EXL3 `mul1` at the pinned source |
|---|---|---|
| Integer generator, modulo 2³² | `s * 0xDF6ADF + 0x34983` | `s * 0x83DCD12D` |
| Codebook normalization | Byte sum minus 510, normalized by the codebook's RMS over all 65,536 states | Byte sum mapped through a fixed FP16 affine FMA |
| Within-tile traversal | Row-major, `t = 16*r + c` | Tensor-core-oriented tile permutation |
| Stored weight stream | `htr_qweight`: INT32 packed words | `trellis`: INT16 packed storage with EXL3's ordering/packing contract |
| Scale/sign metadata | FP16 rotated-row `htr_scales`, bit-packed `htr_su` / `htr_sv` | FP16 signed scale vectors `suh` / `svh`, plus codebook marker metadata |
| Generator arithmetic per state | One `v_mad_u32_u24` | One `v_pk_mul_lo_u16` plus one `v_mad_u32_u16` |
| Normalization before each dot-product pair | No per-pair normalization FMA; correction and scaling are deferred | One packed FP16 FMA, `v_pk_fma_f16` |

### 1. A different codebook—not interchangeable state IDs

For a 16-bit trellis state `s`, the generators are:

```text
HTR RDNA24: x = (s * 0xDF6ADF + 0x34983) mod 2^32
EXL3 mul1:  x = (s * 0x83DCD12D)           mod 2^32
```

Both use unsigned byte sums, but they produce different state-to-value mappings and apply different normalization. HTR defines `v(s) = bytesum(x) - 510`, then normalizes `v` by its RMS across the complete state space and applies an FP16 row scale. EXL3 `mul1` constructs a shifted FP16 value and applies a fused FP16 affine transformation using the half-precision constants with bit patterns `0x1EEE` and `0xC931`. See the [HTR codebook][htr-codebook] and [EXL3 decoder][exl3-codebook].

**The same state ID is not the same weight across these formats.** Changing the generator changes the reconstruction values the encoder must optimize against. It is a substantive numerical choice within the procedural-codebook trellis family.

### 2. A different checkpoint format and weight ordering

HTR-W1 stores each quantized linear as:

```text
htr_qweight   INT32   packed stream words (unsigned bit patterns)
htr_scales    FP16    one scale per rotated output row
htr_su        UINT8   bit-packed input signs
htr_sv        UINT8   bit-packed output signs
```

Within each 16×16 tile, HTR visits weights in row-major order and packs a circular, MSB-first trellis stream. Its sign vectors use one bit per sign. The [wire specification][htr-format] and [reference packer/decoder][htr-packing] define those semantics explicitly.

EXL3 instead uses a [tensor-core-oriented tile permutation][exl3-order], its own [`trellis` packing contract][exl3-packing], and FP16 [`suh` / `svh` scale vectors][exl3-scales]. Those vectors can carry scale magnitudes as well as signs; they are not equivalent to HTR's packed sign arrays.

**Renaming tensor keys cannot convert an HTR checkpoint into EXL3.** A conversion must account for the codebook, traversal, stream layout, and scale/transform semantics. The reconstruction rule itself differs, so this is not merely a byte-repacking task.

### 3. Why HTR's smaller multiplier matters on RDNA

HTR's state is 16 bits and `0xDF6ADF` fits in 24 bits. The multiply and addition therefore fit the native unsigned 24-bit multiply-add:

```text
v_mad_u32_u24
```

The 32-bit result preserves the required modulo-2³² arithmetic. AMD's [RDNA 3.5 ISA reference][amd-isa] describes this instruction as using the single-precision multiplier and says it is expected to be as efficient as basic single-precision operations.

EXL3's `0x83DCD12D` multiplier needs 32 bits. Its optimized RDNA implementation already avoids the slower generic 32-bit multiply: [`exl3_hmul16`][exl3-hash] reconstructs the exact product from two 16-bit operations:

```text
v_pk_mul_lo_u16
v_mad_u32_u16
```

For **hash generation alone**, that is one arithmetic instruction per state for HTR versus two for this EXL3 `mul1` path. Simply truncating EXL3's multiplier to fit a 24-bit instruction would change its codebook.

**This is a hardware-conscious codebook choice, not an arbitrary change of a magic number.** It gives HTR a cheaper generator stage while retaining a procedural trellis codebook.

### 4. Moving normalization out of the inner loop

HTR's second distinction comes from the codebook and kernel together. Let `b` be the unsigned byte sum. The FP16 bit pattern `0x6400 + b` represents the exact integer-valued number `1024 + b`. HTR's [GEMV core][htr-gemv] packs those shifted values directly into FP16 pairs and feeds them to the dot product, accumulating in FP32.

Because HTR's centered codebook is `b - 510`, the offset between the shifted operand and the centered value is **1534**. The [output stage][htr-output] applies the algebraic correction:

```text
shifted_dot_i = sum_j (1024 + b_ij) * a_j
centered_dot_i = shifted_dot_i - 1534 * sum_j a_j
rotated_output_i = centered_dot_i * row_scale_i * codebook_scale * token_scale
```

Here `a` is the transformed FP16 activation vector supplied to GEMV, and `token_scale` restores any activation-range scaling. The output transform/sign handling follows. The fused GEMV variants retain the [same correction in their output stage][htr-fused-output].

The inspected EXL3 [`mul1` pair decoder][exl3-pair] instead normalizes each reconstructed FP16 pair with `__hfma2` before the dot product, yielding `v_pk_fma_f16` on this target.

| Selected stage | HTR | EXL3 `mul1` RDNA tiles path |
|---|---|---|
| Hash generation, per state | **1 arithmetic instruction** | 2 arithmetic instructions |
| Pre-dot normalization, per pair | **0 packed FP16 FMAs** | 1 packed FP16 FMA |

A compile-only `gfx1151` spot-check with AMD Clang `23.0.0git` reproduced these instruction patterns using the published HTR constants and the pinned EXL3 reconstruction primitives. It checked the generator and pair-normalization operations, not complete kernel execution or performance.

**The saving is the placement of the work:** HTR moves affine correction and scale application from every reconstructed pair to the output stage. It still reconstructs weights in registers; this is fused dequantization, not native low-bit matrix arithmetic. EXL3's per-pair FP16 rounding is also part of its numerical behavior, so moving its FMA is not automatically a bit-exact rewrite.

### What these differences establish

HTR has concrete numerical, format, and instruction-selection differences from EXL3, with less arithmetic in the named hash and normalization stages. That is the specific Strix Halo co-design contribution we claim.

The stage counts are not a complete decoder-cost or end-to-end speed ratio. EXL3 also optimizes state extraction, packed SAD pair assembly, and addressing; HTR must account for its activation-sum/output correction, packing, transforms, and reductions. Occupancy, memory traffic, and workload shape still matter. Our published [model measurements](EVIDENCE.md) establish the HTR results under their stated controls, not a matched HTR-versus-EXL3 quality or throughput win.

## Source reuse and licensing disclosure

The imported files in this repository are HTR campaign modules, not a copy of the exlr8 fileset writer/loader or the ExLlamaV3 source tree. A source snapshot manifest records the imported file hashes, and the release notes identify the packaging adaptations. This statement is about the shipped file set, **not a formal clean-room certification or legal determination about algorithmic lineage**. We deliberately retain the intellectual credits even where no upstream file is vendored.

The HTR code is offered under the repository's MIT license. We retain Turboderp's MIT notice in `licenses/ExLlamaV3-MIT.txt` and the identical notice found in exlr8's loader in `licenses/exlr8-loader-MIT.txt` as explicit lineage notices. These files do not purport to license all of exlr8.

License inspection at the references below found no repository-wide exlr8 license declaration, but did find `loader/LICENSE` containing Turboderp's MIT notice. It would be incorrect to infer that every file in exlr8 is MIT from that subdirectory alone. No exlr8 calibration corpus, writer, or loader source files are bundled here. Any future source import must carry its own verified file-level license and attribution; acknowledgments do not replace permission.

No base-model weights, calibration corpora, private prompts, model-output logs, build caches, or binary runtime environment are included in the GitHub source release. Model-Hub artifacts have their own license declarations. In particular, the Qwen3.8 derivative uses Qwen Community License 1.0, not the code repository's MIT license.

Development involved AI-assisted research and coding, with human direction and real hardware evaluation. A generated explanation, source import, or successful unit test is not represented as a new model-quality or GPU-performance measurement.

## Pinned reference points

These pins record the source inspected for release documentation, not a claim that the entire historical project was derived from those later revisions:

- exlr8: [`80aa7e15aad38e94877e561aff5c47c5335c506d`](https://github.com/kindlingai/exlr8/tree/80aa7e15aad38e94877e561aff5c47c5335c506d), including its README/FORMAT and `loader/LICENSE`.
- ExLlamaV3: [`cdc785db89e263e65e5849cb53e1ada156411414`](https://github.com/turboderp-org/exllamav3/tree/cdc785db89e263e65e5849cb53e1ada156411414).
- Earlier EXL3 source comparison: [`a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a`](https://github.com/turboderp-org/exllamav3/tree/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a). This already had RDNA kernels; we make no matched HTR-versus-EXL3 performance claim from that reconnaissance.
- QTIP: [Quantization with Trellises and Incoherence Processing](https://arxiv.org/abs/2406.11235).
- QuIP#: [Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396).
- GPTQ: [Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323).

Please credit these upstream authors when describing or extending HTR. If you find a missing source acknowledgment or file-level notice, open an issue with the exact source and affected file so it can be corrected.

[htr-codebook]: https://github.com/HawgAuto/HTR/blob/84e90dfdb15e42b01c686ad713aaac268db4f96a/htr/wire.py#L5-L30
[htr-format]: https://github.com/HawgAuto/HTR/blob/84e90dfdb15e42b01c686ad713aaac268db4f96a/docs/FORMAT.md#L6-L29
[htr-packing]: https://github.com/HawgAuto/HTR/blob/84e90dfdb15e42b01c686ad713aaac268db4f96a/htr/wire.py#L33-L74
[htr-gemv]: https://github.com/HawgAuto/HTR/blob/84e90dfdb15e42b01c686ad713aaac268db4f96a/htr/csrc/htr_kernels.cu#L64-L105
[htr-output]: https://github.com/HawgAuto/HTR/blob/84e90dfdb15e42b01c686ad713aaac268db4f96a/htr/csrc/htr_kernels.cu#L299-L329
[htr-fused-output]: https://github.com/HawgAuto/HTR/blob/84e90dfdb15e42b01c686ad713aaac268db4f96a/htr/csrc/htr_kernels.cu#L333-L419
[exl3-codebook]: https://github.com/turboderp-org/exllamav3/blob/cdc785db89e263e65e5849cb53e1ada156411414/exllamav3/exllamav3_ext/quant/codebook.cuh#L36-L52
[exl3-order]: https://github.com/turboderp-org/exllamav3/blob/cdc785db89e263e65e5849cb53e1ada156411414/exllamav3/modules/quant/exl3_lib/quantize.py#L43-L71
[exl3-packing]: https://github.com/turboderp-org/exllamav3/blob/cdc785db89e263e65e5849cb53e1ada156411414/exllamav3/modules/quant/exl3_lib/quantize.py#L1117-L1132
[exl3-scales]: https://github.com/turboderp-org/exllamav3/blob/cdc785db89e263e65e5849cb53e1ada156411414/exllamav3/modules/quant/exl3_lib/quantize.py#L1832-L1867
[exl3-hash]: https://github.com/turboderp-org/exllamav3/blob/cdc785db89e263e65e5849cb53e1ada156411414/exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_tiles_rdna.cuh#L96-L134
[exl3-pair]: https://github.com/turboderp-org/exllamav3/blob/cdc785db89e263e65e5849cb53e1ada156411414/exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_tiles_rdna.cuh#L136-L148
[amd-isa]: https://docs.amd.com/v/u/en-US/rdna35_instruction_set_architecture
