![Halo Trellis — HTR](media/hero.png)

# HTR — Halo Trellis

Hessian-aware, rotated trellis weight quantization designed around AMD Strix Halo / Radeon 8060S (gfx1151).

HTR grew from studying **[exlr8](https://github.com/kindlingai/exlr8)** and, through it, **[ExLlamaV3 / EXL3](https://github.com/turboderp-org/exllamav3)** and **[QTIP](https://arxiv.org/abs/2406.11235)**. It is a hardware-specialized member of that family—not an invention of trellis quantization. Please read the prominent [lineage and acknowledgments](docs/LINEAGE.md), [methodology](docs/METHODOLOGY.md), and [license notices](THIRD_PARTY_NOTICES.md).

## What is here

- HTR-W1 packing, reference decoding, and a self-contained PyTorch reference encoder.
- CUDA/HIP Viterbi encoders; Hessian-aware grouped quantization and checkpoint export.
- Output-head encoding and specialized Qwen3.8 dense/expert encoder source.
- Experimental HIP packed-linear kernels and vLLM quantization-plugin integration.
- Format documentation, honest size/fidelity evidence, and CPU release tests.

**Status: source/encoder preview, v0.1.0rc1.** The packed CPU encode → safetensors → reopen → decode path is tested. Historical full-model evidence is supplied separately. The complete production Qwen3.8 MoE/HC/PLE/speculation runtime and final model-release instructions are still being qualified; this source preview is not that turnkey runtime. Stock vLLM, Transformers and llama.cpp do not automatically read HTR weights.

## Why this is useful

Low-bit storage reduces memory traffic, but its quality and decoder cost matter. HTR combines 128-point randomized Hadamard transforms, block-LDLQ error feedback and 16-bit-state trellis search with an RDNA-oriented procedural codebook. K2–K5 is the practical family; sensitive tensors can retain other formats/precision.

In the saved MiMo-9B codec screen, HTR4 used **4.00330 target bpw** versus the local GPTQ4-g128 control's **4.25**, with teacher KL **0.028893 versus 0.065814**. That is 56.10% lower measured coarsened teacher KL with 5.80% fewer target bits—not a claim of 56% better task accuracy or universal superiority to GPTQ/EXL3.

The complete MiMo candidate checkpoints are **6.929 GB** (K4 trunk + K4 head) and **6.065 GB** (K3 trunk + K4 head). Historical single-sequence wall-generation medians were **42.49** and **50.79 tok/s**, versus **13.54** for BF16, on the tested Radeon 8060S setup. Prefill was slower, and these numbers are not a new measurement of this packaged preview. [Full tables, raw repetitions, conditions and limitations](docs/EVIDENCE.md).

## Start with the CPU reference

Use a Python environment with PyTorch, Safetensors and Transformers already installed for your platform. Do not replace a working ROCm/CUDA PyTorch installation blindly.

```bash
git clone https://github.com/HawgAuto/HTR.git
cd HTR
python -m pip install --no-build-isolation --no-deps .
python -m unittest discover -s tests -v
python -m htr.quantize --help
```

The unit suite intentionally uses CPU only. See [USAGE.md](docs/USAGE.md) for matrix encoding, model conversion requirements, and the distinction between an encoder and a serving runtime.

## Documentation

- [Methodology: rotation, calibration, LDLQ, Viterbi, RDNA24, byte accounting](docs/METHODOLOGY.md)
- [Lineage: exlr8 → EXL3/QTIP and related work, with source/license disclosure](docs/LINEAGE.md)
- [HTR-W1 format](docs/FORMAT.md)
- [Evidence and non-claims](docs/EVIDENCE.md)
- [Usage and hardware boundaries](docs/USAGE.md)
- [Release status and remaining model gates](docs/RELEASE_STATUS.md)

## Model release preparation

The separate model repositories are being staged privately until their artifacts and required runtimes are verified:

- `hstolte/Qwen3.8-Flash-Next-HTR`
- `hstolte/MiMo-V2.6-Distill-Qwen-9B-HTR`

These names are release destinations, not a claim of current public download availability. No weights or calibration corpora are included here.

## License and credit

HTR code: [MIT](LICENSE). Source models and external projects retain their own licenses. The Qwen3.8 derivative is subject to Qwen Community License 1.0; the MiMo source card declares MIT. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No upstream author endorsement is implied.
