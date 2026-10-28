# HTR component status

## v0.1.0rc1: codec and encoder source preview

The release packages HTR's W1 codec, reference encoder, accelerated encoder sources, checkpoint tools, and experimental packed-linear integration. The [source manifest](../provenance/source-snapshot.json) records imported and packaged file hashes; the [prerelease](https://github.com/HawgAuto/HTR/releases/tag/v0.1.0-rc.1) provides an installable source wheel and checksum.

| Component | Included | Verified boundary |
|---|---|---|
| W1 packing, signs, transforms, reference decoder | Python source and wire specification | CPU tests with an independent K2–K5 bitstream oracle |
| Reference Viterbi and matrix encoder | Self-contained PyTorch implementation | Circular-state checks and real encode → safetensors → reopen → exact decoded reconstruction |
| Python package and CLI entry points | Installable wheel, module commands, vLLM plugin registration | Isolated installation and CLI help checks; anonymous clone/download verification |
| Accelerated encoders and packed-linear kernels | CUDA/HIP encoder source, HIP inference source | Source included; native execution of this distributable snapshot has a separate qualification boundary |
| Model conversion and architecture adapters | Dense/head tools and specialized dense/expert adapters | Interfaces and intermediate artifacts documented in [usage](USAGE.md) and [adapters](ADAPTERS.md) |
| Experimental vLLM integration | Plugin and consuming-source snapshot | Complete model/runtime qualification remains separate from the CPU package tests |

The seven CPU tests exercise the shipped reference and packaging. [Saved model experiments](EVIDENCE.md) provide fidelity and performance evidence for their recorded artifacts/runtimes, rather than a fresh benchmark of the source wheel.

## Native runtime checklist

An end-to-end deployment needs:

1. A complete checkpoint with explicit widths, retained-precision tensors, layout identity, and index coverage.
2. Matching loader, native kernels, model adapters, and a pinned compatible runtime build.
3. Native checks of the published checkpoint layout through the actual launch path.
4. Memory accounting that separates weights, auxiliary tensors, KV cache, and runtime workspace.
5. Fidelity/task checks and separate prefill, generation, and full-request measurements under stated conditions.

Model weights, calibration datasets, and deployment images are independently versioned artifacts. Their release-specific inventory and launch instructions belong with those artifacts; this repository documents the reusable HTR implementation.

## Source packaging and documentation

Packaging added the in-package reference Viterbi, explicit input paths, setuptools metadata, and CPU tests while preserving the imported encoder numerics. The documentation refresh reorganizes the findings and moves model-specific tables into a case study. Encoder source, raw evidence, original tests, source provenance, acknowledgments, and license notices remain unchanged by that editorial pass.
