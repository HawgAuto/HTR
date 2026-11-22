"""Opt-in fixed-path HTR scale fitting; no new wire or inference parameters.

Only the calibration quadratic is optimized. This is not a guarantee about
held-out/model KL. Code values and scales reconstruct through an FP32 product,
so candidates are checked using that actual product, not only its ideal real
arithmetic approximation. Zero is a legal binary16 scale for zero/dead rows.
"""
import torch


@torch.no_grad()
def refit_row_scales(target, code_values, hessian, scales, *, row_batch=128):
    """Refit existing nonnegative finite FP16 row scales without changing codes.

    target, code_values: [rows, columns] floating tensors on one device.
    hessian: symmetric [columns, columns] calibration covariance (normally PSD).
    scales: [rows] nonnegative finite float16; the incumbent is always a candidate.

    Uses a bounded row working set plus one FP64 Hessian. Nonpositive/degenerate
    row curvature retains the incumbent. For positive curvature, test the
    clamped real optimum's nearest FP16 value and its adjacent FP16 neighbors.
    Accept only an actual FP32-reconstruction quadratic improvement. This is a
    bounded candidate search, not a claim of exhaustive optimality of rounded
    FP32 arithmetic on arbitrary inputs. Exact ties retain the incumbent.

    Returns scales and per-row before/after calibration errors (FP64). Inputs
    are never mutated. No Viterbi search, model execution or device discovery.
    """
    tensors = (target, code_values, hessian, scales)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError('all arguments must be torch tensors')
    if isinstance(row_batch, bool) or not isinstance(row_batch, int) or row_batch <= 0:
        raise ValueError('row_batch must be a positive integer')
    if target.ndim != 2 or min(target.shape) <= 0:
        raise ValueError('target must be a nonempty matrix')
    m, n = target.shape
    if code_values.shape != target.shape or hessian.shape != (n, n) or scales.shape != (m,):
        raise ValueError('target, codes, Hessian and scales have incompatible shapes')
    if not all(t.is_floating_point() for t in tensors):
        raise TypeError('target, codes, Hessian and scales must be floating tensors')
    if scales.dtype != torch.float16:
        raise TypeError('incumbent scales must have their serialized float16 dtype')
    if len({t.device for t in tensors}) != 1:
        raise ValueError('all tensors must be on the same device')
    if not all(bool(torch.isfinite(t).all()) for t in tensors) or bool((scales < 0).any()):
        raise ValueError('inputs must be finite and scales nonnegative')
    hd = hessian.double()
    if not torch.allclose(hd, hd.T, rtol=1e-6, atol=1e-10):
        raise ValueError('Hessian must be symmetric')
    # The antisymmetric part is irrelevant to e H e^T; canonicalize roundoff.
    hd = hd * .5 + hd.T * .5
    result = scales.clone()
    before = torch.empty(m, dtype=torch.float64, device=scales.device)
    after = torch.empty_like(before)
    max_scale = torch.finfo(torch.float16).max
    for start in range(0, m, row_batch):
        stop = min(start + row_batch, m)
        w = target[start:stop].double()
        c = code_values[start:stop].float()
        if not bool(torch.isfinite(c).all()):
            raise ValueError('codes must be representable as finite float32')
        old = scales[start:stop]
        cd = c.double()
        ch = cd @ hd
        curvature = (ch * cd).sum(dim=1)
        cross = (ch * w).sum(dim=1)
        valid = torch.isfinite(curvature) & torch.isfinite(cross) & (curvature > 0)
        optimum = torch.where(valid, cross / torch.where(valid, curvature, 1.), old.double())
        nearest = optimum.clamp(0., max_scale).half()
        lower = torch.nextafter(nearest, torch.full_like(nearest, -float('inf')))
        upper = torch.nextafter(nearest, torch.full_like(nearest, float('inf')))
        e = w - (c * old.float()[:, None]).double()
        best_error = ((e @ hd) * e).sum(dim=1)
        if not bool(torch.isfinite(best_error).all()):
            raise ValueError('nonfinite incumbent quadratic; rescale the inputs')
        before[start:stop] = best_error
        best_scale = old.clone()
        for candidate in (nearest, lower, upper):
            admissible = valid & torch.isfinite(candidate) & (candidate >= 0)
            candidate = torch.where(admissible, candidate, old)
            e = w - (c * candidate.float()[:, None]).double()
            score = ((e @ hd) * e).sum(dim=1)
            improve = admissible & torch.isfinite(score) & (score < best_error)
            best_error = torch.where(improve, score, best_error)
            best_scale = torch.where(improve, candidate, best_scale)
        result[start:stop] = best_scale
        after[start:stop] = best_error
    return {'scales': result, 'before': before, 'after': after}


@torch.no_grad()
def refit_packed(target, hessian, packed, *, damp=0.01, row_batch=128):
    """Refit an existing K2--K5 W1 matrix, preserving all codes/signs and bytes.

    target/H are in the original basis. Uses the encoder's input/output
    transform and zero-column/damping preparation; statistics describe that
    prepared rotated-domain calibration objective. Returns an encode_group-
    compatible dictionary plus a scalar-only ``scale_refit`` diagnostic.
    Full matrix outputs are retained, but integer unpacking and fitting are
    row-bounded. No file writes or Viterbi search occur here.
    """
    import math
    from . import encode, wire

    if not isinstance(target, torch.Tensor) or not isinstance(hessian, torch.Tensor):
        raise TypeError('target and Hessian must be tensors')
    if target.ndim != 2 or min(target.shape) <= 0:
        raise ValueError('target must be a nonempty matrix')
    m, n = target.shape
    if m % 128 or n % 128 or hessian.shape != (n, n):
        raise ValueError('W1 requires aligned matrix dimensions and matching Hessian')
    if not target.is_floating_point() or not hessian.is_floating_point():
        raise TypeError('target and Hessian must be floating tensors')
    if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(hessian).all()):
        raise ValueError('target and Hessian must be finite')
    # FP32 Gram accumulation can differ across the two stored triangles. An
    # entrywise relative check over-penalizes near-zero cross-covariances.
    # Admit only normwise FP32 roundoff; retain the original raw H/rotation and
    # the existing prepared-H symmetrization below. No damping/eigenvalue fix.
    h_norm = torch.linalg.vector_norm(hessian, dtype=torch.float64)
    skew_norm = torch.linalg.vector_norm(hessian - hessian.T, dtype=torch.float64)
    if bool(skew_norm > (8 * torch.finfo(torch.float32).eps) * h_norm):
        raise ValueError('input Hessian must be symmetric within FP32 normwise roundoff')
    input_skew = float(skew_norm / h_norm) if bool(h_norm > 0) else 0.
    if isinstance(damp, bool) or not isinstance(damp, (int, float)) or not math.isfinite(damp) or damp < 0:
        raise ValueError('damp must be finite and nonnegative')
    if isinstance(row_batch, bool) or not isinstance(row_batch, int) or row_batch <= 0:
        raise ValueError('row_batch must be a positive integer')
    required = ('qweight', 'scales', 'su', 'sv')
    if not isinstance(packed, dict) or any(k not in packed or not isinstance(packed[k], torch.Tensor) for k in required):
        raise TypeError('packed must contain qweight, scales, su and sv tensors')
    qw, scale, su8, sv8 = (packed[k] for k in required)
    if qw.dtype != torch.int32 or qw.ndim != 3 or tuple(qw.shape[:2]) != (m//16, n//16):
        raise ValueError('invalid W1 qweight geometry or dtype')
    if qw.shape[2] not in (16, 24, 32, 40):
        raise ValueError('refit supports the K2--K5 W1 widths only')
    if scale.dtype != torch.float16 or scale.shape != (m,) or not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
        raise ValueError('invalid W1 FP16 scales')
    if su8.dtype != torch.uint8 or sv8.dtype != torch.uint8 or su8.shape != (n//8,) or sv8.shape != (m//8,):
        raise ValueError('invalid W1 packed signs')
    device = target.device
    qw_dev, scale_dev = qw.to(device), scale.to(device)
    su = wire.unpack_signs(su8.to(device), n)
    sv = wire.unpack_signs(sv8.to(device), m)
    wt = encode.blk_had_cols(target.float() * sv[:, None] * su[None, :])
    wt = encode.blk_had_cols(wt.T.contiguous()).T.contiguous()
    ht = encode.blk_had_cols(hessian.float().to(device) * su[:, None] * su[None, :])
    ht = encode.blk_had_cols(ht.T.contiguous()).T.contiguous()
    ht, wt = encode.prep_H(ht, wt, damp)
    ht = ht * .5 + ht.T * .5
    codes = torch.empty((m, n), dtype=torch.float32, device=device)
    tile_batch = max(1, row_batch // 16)
    for lo in range(0, m//16, tile_batch):
        hi = min(lo + tile_batch, m//16)
        codes[16*lo:16*hi] = wire.dequant_Q(qw_dev[lo:hi], torch.ones(16*(hi-lo), dtype=torch.float16, device=device))
    fit = refit_row_scales(wt, codes, ht, scale_dev, row_batch=row_batch)
    q = codes * fit['scales'].float()[:, None]
    w = wire.blk_had_cols_T(q.T.contiguous()).T
    w = wire.blk_had_cols_T(w.contiguous())
    what = (w * sv[:, None] * su[None, :]).contiguous()
    return dict(qweight=qw.cpu(), scales=fit['scales'].cpu(), su=su8.cpu(), sv=sv8.cpu(),
                Q=q, What=what, scale_refit=dict(
                    objective='prepared_rotated_input_hessian_fp32_product', damp=float(damp),
                    input_relative_skew_fro=input_skew,
                    before=float(fit['before'].sum()), after=float(fit['after'].sum()),
                    changed_rows=int((fit['scales'] != scale_dev).sum()), rows=m,
                    max_row_delta=float((fit['after'] - fit['before']).max())))
