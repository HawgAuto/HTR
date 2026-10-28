# Using HTR

HTR provides a CPU reference path, accelerated encoder source, checkpoint conversion tools, and experimental packed-linear runtime integration. Start with a matrix round trip, then supply real calibration and a supported model adapter for checkpoint work.

## Requirements

Use Python 3.10+, a platform-appropriate PyTorch build, Safetensors, and model-compatible Transformers. The source-preview CPU packaging checks used PyTorch 2.10.0 and Transformers 5.10.2.

Native CUDA/HIP encoders also need the matching compiler, headers, and Ninja. The included inference kernels target AMD gfx1151; CUDA encoder support and CUDA inference support are different capabilities. Keep ROCm/CUDA library environments separate. [Component status](RELEASE_STATUS.md) records the verification boundary.

## Small CPU encode and round trip

```python
import torch
from htr import encode, wire

torch.set_num_threads(2)
W = torch.randn(128, 128) * 0.02
H = torch.eye(128)  # demonstration; use captured activations for model calibration
encode.VITERBI = None  # use the packaged PyTorch reference
packed = encode.encode_group([dict(W=W, H=H)], K=4)[0]
W_hat = wire.dequant(packed["qweight"], packed["scales"], packed["su"], packed["sv"])
assert torch.equal(W_hat, packed["What"])
```

This encodes a real matrix, emits W1, and decodes it. The [release test](../tests/test_release.py) also saves the tensors to safetensors, reopens them, and checks exact reconstruction of the encoded result. The CPU reference is for inspection and small checks; full-size matrices use accelerated encoding. GPU variants are selected by the checkpoint CLI or by assigning `encode.VITERBI = htr.vit_v2.viterbi_auto` after importing that module.

## Dense checkpoint encoding

Run conversion in an isolated GPU environment sized for the source model, Hessians, and encoder workspace. The generic command shape is:

```bash
python -m htr.quantize \
  --model /models/source-model \
  --out /models/source-model-htr4 \
  --bits 4 \
  --calib /data/calibration.txt \
  --hessians /data/hessians.pt \
  --ncal 128 --seqlen 2048
```

- **Calibration:** supply representative text and an existing Hessian file or `layerNNN.pt` directory as appropriate. The calibration text is also used to identify shared-input groups.
- **CPU offload:** `--offload` keeps the source model on CPU and moves one encode group to the GPU. It requires existing Hessians and substantial host RAM; it does not perform low-memory first-time calibration.
- **Width policy:** `--policy` accepts regex width/escape overrides. K2–K5 is the practical family. The default exclusions cover embeddings, the output head, vision/audio/MTP families, and unsupported dimensions; excluded tensors retain their source representation.
- **Group cache:** `--cache` supports reuse by same-host workers. Match source, tensor policy, and seeds; use a separate cache for a different encoding identity.

The CPU reference is exercised by the source-preview tests. Full-model conversion and serving require their own model/backend validation.

## Output-head encoding

`python -m htr.lmhead --help` exposes `capture`, `worker`, and `assemble`. This provides a separately calibrated head path so its precision can be chosen independently from the trunk.

Assembly can produce local symlinks to unchanged base shards. For a standalone checkpoint, materialize those references and verify that every tensor in the index resolves to an included file.

## Architecture adapters

The shared codec and matrix encoder are reusable; a model's tensor names, fusion groups, expert topology, and auxiliary components determine its checkpoint adapter. The bundled [architecture-specific encoders](ADAPTERS.md) document their supported inputs and intermediate packet formats separately.

## Experimental vLLM integration

Installation registers the `vllm.general_plugins` entry point `htr`. The package contains HIP packed-linear sources; native binaries and build caches are generated in the target environment.

A serving deployment binds the checkpoint, plugin, native kernels, model adapters, and compatible runtime build. This source preview supplies the reference and integration source rather than a prequalified launch image. Use the [runtime checklist](RELEASE_STATUS.md#native-runtime-checklist) when bringing up an end-to-end deployment.

## Input trust and workspace safety

Load only trusted `.pt` Hessian/cache/packet files: some inherited workflows use pickle-bearing `torch.load` inputs. Use a new output directory, preserve source checkpoints, and avoid writing through hardlinks. JIT compilation executes a local compiler; set `HTR_BUILD_DIR` to a writable cache directory outside the installed package when needed.

The CPU tests initialize no GPU and download no model. Before a full conversion, check available host/GPU memory and ownership of the target GPU.
