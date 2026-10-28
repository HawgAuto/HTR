# Using the source preview

## Requirements and scope

Python 3.10+, an appropriate PyTorch build, Safetensors, and model-compatible Transformers are required. The packaging check used PyTorch 2.10.0 and Transformers 5.10.2 with all GPUs hidden. These are CPU-test facts, not a GPU serving compatibility matrix.

Native CUDA/HIP encoders also need the matching compiler, headers and Ninja. Native inference kernels are specialized for AMD gfx1151. CUDA encoder source is not CUDA HTR inference support. Do not mix ROCm and CUDA libraries, and do not assume an arbitrary stock vLLM version matches the experimental plugin.

## Small CPU encode and round trip

```python
import torch
from htr import encode, wire

torch.set_num_threads(2)
W = torch.randn(128, 128) * 0.02
H = torch.eye(128)  # demonstration only; use captured real activations for model quality
encode.VITERBI = None  # packaged PyTorch reference
packed = encode.encode_group([dict(W=W, H=H)], K=4)[0]
W_hat = wire.dequant(packed["qweight"], packed["scales"], packed["su"], packed["sv"])
assert torch.equal(W_hat, packed["What"])
```

This encodes real values, emits W1 and decodes them; it does not load a model or prove a model-level quality result. The release test additionally saves to safetensors, reopens the file and compares the result. The reference path is deliberately slow for large matrices. GPU variants are selected explicitly by the checkpoint CLI or by assigning `encode.VITERBI = htr.vit_v2.viterbi_auto`.

## Dense HF checkpoint encoder

Run only in an isolated, appropriately sized GPU environment. The example is the supported CLI shape, not a freshly reproduced full-model conversion:

```bash
python -m htr.quantize   --model /models/source-model   --out /models/source-model-htr4   --bits 4   --calib /data/calibration.txt   --hessians /data/hessians.pt   --ncal 128 --seqlen 2048
```

An existing Hessian file or `layerNNN.pt` directory can be supplied. `--offload` keeps the source model on CPU and moves one encode group to the GPU, but it requires existing Hessians; it is not a low-memory first-time calibration implementation. The calibration text is still needed for input-group identification. CPU offload still consumes substantial host RAM. A shared per-group `--cache` can support same-host workers; claims and cached payloads must match source, policy and seeds. Do not reuse arbitrary stale caches.

The CLI excludes embeddings, head, vision/audio/MTP families and unsupported dimensions by default. All excluded tensors are retained; the output is not a uniform whole-model K4 artifact. `--policy` accepts regex width/escape overrides. Supported recommended widths are K2–K5; wider legacy paths are not qualified release choices.

## Output head

`python -m htr.lmhead --help` exposes `capture`, `worker` and `assemble`. This is the separately calibrated head procedure used by the MiMo candidates. Its assembly can produce local symlinks to unchanged base shards. Before uploading, materialize those references and verify the full index: an overlay directory is not automatically a self-contained Hub checkpoint.

## Qwen3.8-specific encoders

- `python -m htr.encode_q38 --help`: grouped dense-trunk overlay encoding from a compatible source checkpoint and captured Hessians.
- `python -m htr.moe_encode --help`: specialized 48-layer/512-expert Qwen3.8 source, calibration-window dumps and packed expert packets. `--reference` explicitly names the mixed INT4/INT8 control used for diagnostic proxy rows; it is not silently taken from an operator directory.

The MoE encoder preserves the original campaign numerics and checks packed reconstruction against the encoder's BF16 reconstruction before writing packets. It produces `HTR-W1-P2-PACKED-1` intermediate packets, not a standalone Hugging Face/vLLM release. The final routed layout, HC wire, model assembly and runtime are separate steps still owned by the model-release work. Do not advertise this specialized command as a generic all-model MoE converter.

## Experimental vLLM integration

Installation registers the `vllm.general_plugins` entry point `htr`. This is preferable to importing vLLM from `sitecustomize` in every Python child. The package also contains HIP packed-linear sources. Native binaries and caches are not bundled.

The source preview does not provide a qualified production launch command or container image. The active Qwen runtime contains additional routed-expert, hyperconnection, PLE and speculation work outside this package. Do not point stock vLLM at the model and infer support from an import succeeding. The final model card must bind weights, source commit, tested image digest, launch flags and validation evidence together.

## Trust and safety

Checkpoint and calibration paths are local inputs. Load only trusted `.pt` Hessian/cache/packet files: inherited workflows use `torch.load` and some permit pickle-bearing objects. Do not run shared caches from an untrusted party. Use a new output directory; preserve source checkpoints and avoid writing through hardlinks. JIT compilation executes a local compiler and writes a build cache; set `HTR_BUILD_DIR` to a writable cache location outside the installed package when needed.

No GPU reset, service stop, model download or remote upload is performed by the CPU tests. A full conversion is expensive and requires a separate capacity and GPU-ownership check.
