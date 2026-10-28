# Release status: source preview versus model release

## Source preview v0.1.0rc1

This repository packages the existing HTR encoder/codec and an experimental packed-linear integration. Its private-history source tree was not pushed. The imported snapshot is recorded by file hash; packaging changes remove machine-specific defaults, add the missing in-package PyTorch reference Viterbi, make MoE reference inputs explicit, and add packaging metadata/tests. No numerical algorithm change is intended.

Fresh CPU verification covers independent bitstream decoding for K2–K5, sign convention, Hadamard inversion, self-contained reference Viterbi with circular-state checks, and matrix encode → safetensors → reopen → exact decoded reconstruction. Historical full-model checks are retained separately in `evidence/`; they are not represented as reruns of this package.

CUDA/HIP compilation, the packaged plugin's native execution and the complete model paths have not been rerun in this preparation session. No production service or competing GPU job was changed. The source preview is not the full optimized Qwen3.8 engine, not a turnkey image, and not a release announcement for unfinished weights.

## MiMo-9B model package

Two accepted candidates are being staged with complete shards, tokenizer/config files, the model card, hero, source revision, checksums and evidence. Archive symlinks are resolved within the verified archive; the Hub payload contains actual files rather than paths to a private base checkpoint.

Remaining public-release gates: exact remote weight hashes/coverage, finalized runtime/source/image binding, a release-layout native smoke, and public-last verification. The intended model repository remains private during that process.

## Qwen3.8-Flash-Next model package

The active qualification session owns the final model/runtime selection. Current intended policy is K5 selected dense trunk, K4 output head, K4 routed experts, separate GPTQ8-HC hyperconnections and retained auxiliary precision. Preparation does not re-open that width decision.

The staged card/hero and upload contract must be finalized only after receiving:

1. Frozen complete checkpoint inventory, source revisions and hashes; exact HC, routed-expert, PLE and optional draft dependencies.
2. The accepted runtime source commit, full overlay closure, built/tested image digest and launch entry point.
3. Actual final checkpoint/PLE/draft sizes; startup weights/KV/runtime allocations labeled separately from total peak/minimum RAM.
4. Final matched prefill, generation/wall and task/fidelity evidence with sampling/context/batch/speculation conditions.
5. Successful checks through the published-layout loader and actual launch path.

The staged Qwen card does not invent final speed, quality, download size, image tags or commands. No mutable live campaign file is uploaded as a final model simply to fill the repository.
