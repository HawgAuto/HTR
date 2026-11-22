# Experimental routed-expert scale refit

`python -m htr.moe_encode --scale-refit` enables the existing fixed-path FP16 row-scale refitter for routed experts at K2–K5. It is **off by default**. This research candidate is not a model-quality or production-default recommendation.

The existing gate/up and down encode groups retain their seeds, sign sharing, calibration Hessians and `--damp`. After the unchanged code search, `encode_group` applies `refit_packed` with row batches of 128. Codes and signs do not change; scales and reconstructed weights may change. The existing packed-decode-versus-BF16-reconstruction check runs on every resulting matrix before the chunk is saved. Per-role prepared-objective diagnostics appear under each proxy row's `scale_refit`; they do not measure held-out model quality.

Example opt-in, with a NEW output directory:

```sh
python -m htr.moe_encode \
  --src /path/to/bf16-checkpoint \
  --reference /path/to/mixed-reference \
  --dump /path/to/calibration-dump \
  --out /path/to/new-refit-output \
  --bits 4 --layers 2 --chunk 16 --damp 0.01 --scale-refit
```

Omit `--scale-refit` for the original behavior, including at K6/K8. The flag rejects K6/K8 before device access.

## Resumption and identity

- Enabled output plans contain `scale_refit: true`; chunks contain `encoder.scale_refit: true` and `encoder.refit_row_batch: 128`.
- A legacy plan without the flag means the original, unrefit recipe. Its default resume remains supported; an enabled run cannot resume into it.
- Refitted and unrefitted jobs require separate output trees. Switching the flag for an existing tree is refused. Unmarked orphan chunks cannot be relabeled as refitted.
- Plans are published complete, atomically and without replacement, then recipe identity is checked again before encoding. This protects new-version workers racing to initialize the same tree. Do not mix older workers that lack this identity contract with a refit job.
- This entry point encodes new chunks; unlike the dense `htr.quantize` assembler, it does not refit an existing chunk cache in place.

## Evidence boundary

The new CPU regression suite exercises the actual main/parser/Hessian/encoder/refitter/proxy/verification/persistence path on synthetic expert data, with a placement-only CPU adapter, doubled external data/hardware boundaries, and a legal-path stand-in for native trellis search. K2–K5 results match independent post-encode `refit_packed` calls; default packed fields, proxy rows and plan bytes match the pinned predecessor. Existing core arithmetic and tests are unchanged.

Native routed-expert CLI qualification, real-checkpoint/model-quality evaluation and any default decision remain separate gates. Router-Hessian weighting is not changed. `encode_q38` is a separate dense-trunk/shared-expert entry point and is not modified by this candidate.
