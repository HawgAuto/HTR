# Lineage, acknowledgments, and scope of our contribution

HTR did not begin in a vacuum. The project began by studying **[kindlingai/exlr8](https://github.com/kindlingai/exlr8)** and asking how that design philosophy could be adapted to AMD Strix Halo. exlr8 explicitly credits **[Turboderp's ExLlamaV3 / EXL3](https://github.com/turboderp-org/exllamav3)** for its trellis codec, and EXL3 identifies its relationship to **QTIP**. HTR should be understood in that lineage, not advertised as an unrelated invention.

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

## What is specifically HTR in this release?

- The RDNA24 generator choice (`0xDF6ADF`, `0x34983`) and its W1 codebook normalization.
- The HTR-W1 metadata/bitstream contract and corresponding encoder/decoder implementations.
- Its grouped shared-sign encoding and the particular Strix-oriented packed execution integration.
- The model-specific tensor policies, calibration/evaluation work, and measured tradeoffs published with the releases.
- The work to reduce memory traffic without calling low-bit storage native low-bit arithmetic.

Some ingredients overlap contemporary EXL3 implementations. **This is a specialized implementation and co-design effort, not a claim that every ingredient is novel.** Current EXL3 already includes RDNA/HIP support and optimized byte-sum decoding; an older MCG instruction-count comparison cannot establish that HTR is faster than current EXL3.

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
