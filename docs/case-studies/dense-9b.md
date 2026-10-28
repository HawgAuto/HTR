# Dense 9B case study: fidelity, packing, and Strix Halo serving

This study uses **XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B** to test HTR at three levels: the codec, the packed checkpoint, and GPU execution. It supplies the model-specific evidence behind the [HTR findings](../EVIDENCE.md); it is not a model-release checklist.

These are retained real-model experiments. The raw JSON is unchanged, including original row labels and all serving repetitions. The codec screen and packed-checkpoint evaluation use different tensor policies and remain separate comparisons.

Sources: [codec screen](../../evidence/codec-screen.json) · [packed quality](../../evidence/packed-quality.json) · [serving repetitions and runtime](../../evidence/historical-serving.json).

## Codec screen (E08)

Same BF16 source/target tensor set, 128×2048 calibration windows and 64×2048 held-out WikiText windows (131,008 scored tokens). Head and embeddings were excluded from the low-bit target subset. Values are fake-quantized weights evaluated through the HF model; they are not the K4-head release-artifact rows. Lower KL means closer teacher distribution, not higher task accuracy. KL is top-128-plus-rest coarsened KL.

| Original quant label | Target bpw | PPL | Teacher KL | Top-1 agreement |
|---|---:|---:|---:|---:|
| bf16 | 16.000000 | 9.02360 | reference | reference |
| gptq:4 | 4.250000 | 9.12474 | 0.065814 | 0.89984 |
| htr:3 | 3.003299 | 9.32047 | 0.082963 | 0.88609 |
| gptq:3 | 3.250000 | 9.67354 | 0.237551 | 0.79979 |
| ldlq_scalar_rht:3 | 3.003299 | 9.41058 | 0.138338 | 0.84702 |
| htr:4 | 4.003299 | 9.06851 | 0.028893 | 0.93634 |
| htr:2 | 2.003299 | 10.54739 | 0.276500 | 0.77959 |
| Q4_K | 4.500000 | 8.99879 | 0.041131 | 0.92177 |
| IQ4_XS | 4.250000 | 8.88564 | 0.056440 | 0.90846 |
| Q3_K | 3.437500 | 10.23806 | 0.212514 | 0.80948 |
| IQ3_XXS | 3.062500 | 9.29002 | 0.185692 | 0.82051 |
| Q2_K | 2.625000 | 9.96638 | 0.373139 | 0.74590 |
| IQ2_XS | 2.312500 | 10.89549 | 0.450601 | 0.71592 |

HTR4 in this screen had 56.10% lower teacher KL and 5.80% fewer bits on the target tensors than the local GPTQ4-g128 control. Several scalar/GGML controls have lower PPL despite higher teacher KL; both columns are shown. K3 and K4 provide different size/fidelity tradeoffs.

## Packed checkpoint fidelity (E10)

HF evaluation, 64×2048 WikiText windows, same-process BF16 teacher, 131,008 scored tokens. This includes the K4 head and is separate from the codec screen.

| Original label | PPL | Teacher KL | Top-1 agreement |
|---|---:|---:|---:|
| bf16 | 9.019484 | reference | reference |
| bf16-repeat | 9.019484 | 0.000000 | 1.000000 |
| htr4-lmh4-packed | 9.047606 | 0.030529 | 0.929737 |
| htr4-lmh4-fake | 9.048045 | 0.030586 | 0.929218 |
| bf16 | 9.019484 | reference | reference |
| bf16-repeat | 9.019484 | 0.000000 | 1.000000 |
| htr3-lmh4-packed | 9.223103 | 0.083370 | 0.882854 |
| htr3-lmh4-fake | 9.226923 | 0.083447 | 0.882725 |

The fake and packed rows are close, not bit-identical full-model outputs. The BF16 repeat is exact in this evaluation.

## Radeon 8060S serving (E10)

Graphs on, no speculation, single sequence, prefix cache off; model sampling temperature 0.6/top-p 0.95/top-k 20. Generation: 16-token prompt and 256 forced output tokens. Prompt test: 2048 input plus one output token. Three repetitions; medians below and all samples in [`historical-serving.json`](../../evidence/historical-serving.json). Both rates include the full in-process `generate` call, not an HTTP server-only device timer. Final-table HTR kernels were the then-promoted v4/v5 path. This records that runtime; successor runtimes require their own measurements.

| Original label | Wall generation tok/s | Prompt wall tok/s | Serving PPL (16×2048) |
|---|---:|---:|---:|
| mimo9b-bf16-final | 13.545 | 1640.73 | 9.152306 |
| mimo9b-htr4-final | 33.893 | 1450.30 | 9.206132 |
| mimo9b-htr3-final | 38.869 | 1469.19 | 9.364289 |
| mimo9b-htr4-lmh4-final | 42.486 | 1467.09 | 9.209396 |
| mimo9b-htr3-lmh4-final | 50.791 | 1320.11 | 9.367120 |
| mimo9b-htr3-lmh3-final | 51.834 | 1422.48 | 9.399853 |

Lower-bit variants improved generation while reducing prompt throughput versus BF16. The additional K3-head row is a diagnostic. The three repetitions establish the observed rates and spread for this workload, not cross-model portability or task-level quality.
