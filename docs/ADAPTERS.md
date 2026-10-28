# Architecture-specific encoder adapters

HTR's [wire format](FORMAT.md), matrix encoder, and reference decoder are the common layer. These additional tools adapt that layer to particular checkpoint structures. Start with [general usage](USAGE.md) for the shared interfaces.

## Qwen3.8 dense trunk

`python -m htr.encode_q38 --help` exposes grouped dense-trunk overlay encoding from a compatible source checkpoint and captured Hessians. It preserves the input-group relationships needed by fused projections.

An overlay is an intermediate artifact. Combine it with the intended retained tensors and validate the complete checkpoint index before distribution.

## Qwen3.8 routed experts

```bash
python -m htr.moe_encode --help
```

This encoder targets the 48-layer/512-expert Qwen3.8 layout, calibration-window dumps, and packed expert packets. `--reference` explicitly identifies the mixed INT4/INT8 control used for diagnostic proxy rows.

The encoder checks packed reconstruction against its BF16 reconstruction before writing `HTR-W1-P2-PACKED-1` intermediate packets. A full deployment also needs final routed layouts, hyperconnection/auxiliary handling, model assembly, and matching runtime kernels. The specialized adapter is not a generic converter for arbitrary MoE architectures.

## Extending HTR to another architecture

Identify the target matrices and retained-precision tensors, construct compatible shared-input groups, capture activation statistics, and verify emitted packets with the reference decoder. Then validate whole-model fidelity and the consuming runtime independently. Architecture support is demonstrated by that complete path, not by renaming a checkpoint's quantization method.
