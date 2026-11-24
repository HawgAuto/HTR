"""htr-quantize: HF checkpoint -> HTR-W1 packed checkpoint (vLLM quant_method "htr").

  python -m htr.quantize --model <hf dir> --out <dir> --bits 4 [--policy policy.json] [--calib text.raw]

Pipeline: load BF16 model -> capture input Hessians H = E[x x^T] for every target linear (random windows of the
calibration text) -> group linears that consume the same input tensor (shared input signs; fused by runtimes)
-> encode each group (RHT + block-LDLQ + tail-biting trellis Viterbi) -> write safetensors: quantized linears as
<prefix>.htr_{qweight,scales,su,sv}, every other tensor copied byte-for-byte from the source shards.
"""
import argparse, hashlib, json, os, random, re, shutil, sys, time, glob
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from . import encode, wire

SKIP_PARTS = ("visual", "vision", "audio", "mtp", "lm_head", "embed")
DEVICE = "cuda"
REFIT_DAMP = 0.01            # encode_group's damping: the refit prepares the same rotated calibration objective
REFIT_WIDTHS = (2, 3, 4, 5)  # htr.scale_refit.refit_packed supports the K2..K5 W1 widths


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def canon(name):
    if name.endswith(".weight"):
        name = name[:-7]
    i = name.rfind("layers.")
    return name[i:] if i >= 0 else name


def ckpt_index(model_dir):
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    idx = {}
    for f in files:
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                idx[k] = f
    return files, idx


def resolve_policy(policy, cname, default_k):
    for pat, val in (policy.get("overrides") or {}).items():
        if re.search(pat, cname):
            return None if str(val).lower() in ("bf16", "none", "skip", "escape") else int(val)
    return default_k


def quant_config(report, seam, scale_refit=False):
    """quantization_config for a packed checkpoint. The defaults write exactly the earlier config; an opt-in seam or
    scale refit is recorded as provenance under "encoder" (the wire and every decoder are unchanged, so loaders
    ignore it)."""
    qc = {"quant_method": "htr", "wire": "HTR-W1", "generator": "rdna24", "L": 16, "tile": 16,
          "rht_block": 128, "modules": {k: {"K": v["K"]} for k, v in report.items()}}
    enc = {}
    if seam != "htr":
        enc["seam"] = seam
    if scale_refit:
        enc["scale_refit"] = True
    if enc:
        qc["encoder"] = enc
    return qc


def rel_proxy(W, What, H):
    """Relative calibration proxy e H e^T / w H w^T in FP64 (the per-module report value)."""
    E_ = (W.double() - What.double())
    Hd = H.double()
    return float(((E_ @ Hd) * E_).sum() / ((W.double() @ Hd) * W.double()).sum())


def refit_group(jobs, res, damp=REFIT_DAMP):
    """Opt-in post-encode FP16 row-scale refit of one encoded group (htr.scale_refit.refit_packed per matrix).

    jobs: encode_group-style [{W, H}] in the original basis; res: that group's encoded (or cached) qweight/scales/su/sv.
    Equal to encode_group(..., scale_refit=True), which runs the same pass on its own outputs. Codes and signs are
    unchanged; returns the refit results (with each matrix's scale_refit diagnostic) and their rel_proxy values."""
    from .scale_refit import refit_packed
    outs = [refit_packed(jb["W"], jb["H"], r, damp=damp) for jb, r in zip(jobs, res)]
    return outs, [rel_proxy(jb["W"], o["What"], jb["H"]) for jb, o in zip(jobs, outs)]


def cache_matches(cached, names, K, seam):
    """A --cache group file is reusable only for the same names, K and seam (files without a seam predate it: 'htr')."""
    return cached["names"] == names and cached["K"] == K and cached.get("seam", "htr") == seam


def check_cache_seed(cached, names, shapes, group_seed):
    """Verify a worker's exact seed when recorded, and its actual RHT sign bytes even for old caches."""
    if "group_seed" in cached and (type(cached["group_seed"]) is not int or cached["group_seed"] != group_seed):
        raise ValueError("cache group seed mismatch; use the matching --seed and cache")
    results = cached.get("res")
    if not isinstance(results, (list, tuple)) or len(results) != len(names) or len(shapes) != len(names):
        raise ValueError("cache seed check: incomplete group")
    for ji, (name, (m, n), result) in enumerate(zip(names, shapes, results)):
        # Legacy caches lack the original integer. Their packed signs can establish consistency,
        # not uniqueness (different seeds may produce the same signs on a given RNG backend).
        su, sv = encode.make_signs(n, m, group_seed, ji, shared_su=True)
        for field, expected in (("su", su), ("sv", sv)):
            observed = result.get(field) if isinstance(result, dict) else None
            if (not isinstance(observed, torch.Tensor) or observed.device.type != "cpu"
                    or observed.dtype != torch.uint8 or not torch.equal(observed, wire.pack_signs(expected))):
                raise ValueError(f"cache seed/sign mismatch for {name}.{field}; use the matching --seed and cache")


def _tensor_digest(t):
    """Hash logical matrix values in row order, including dtype, shape and strides, a few MiB at a time."""
    t = t.detach()
    h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|{tuple(t.stride())}|".encode())
    rows = max(1, (4 << 20) // max(1, t.shape[1] * t.element_size()))
    for chunk in t.split(rows):
        h.update(chunk.contiguous().cpu().view(torch.uint8).numpy())
    return h.hexdigest()


def group_input_identity(names, mods, hessians):
    # Use the loaded BF16 model values, not paths. A layer-directory cache already has a host copy.
    get_h = hessians.host_tensor if isinstance(hessians, LayerDirHessians) else hessians.__getitem__
    members, shared = {}, {}
    for name in names:
        h = get_h(name)
        if h not in shared:        # repeated Tensor object within this input group only
            shared[h] = _tensor_digest(h)
        members[name] = {"weight": _tensor_digest(mods[name].weight), "hessian": shared[h]}
    return {"format": "typed-layout-sha256-v1", "members": members}


def check_input_identity(record, expected):
    if not isinstance(record, dict) or set(record) != set(expected) or record["format"] != expected["format"]:
        raise ValueError("invalid cache input identity format")
    members = record["members"]
    if not isinstance(members, dict) or set(members) != set(expected["members"]):
        raise ValueError("invalid cache input identity members")
    for name, fields in expected["members"].items():
        got = members[name]
        if not isinstance(got, dict) or set(got) != set(fields):
            raise ValueError(f"invalid cache input identity for {name}")
        for field, digest in fields.items():
            if not isinstance(got[field], str) or got[field] != digest:
                raise ValueError(f"cache {field} differs for {name}; use a fresh cache")


class LayerDirHessians:
    """E08-format Hessian cache: <dir>/layerNNN.pt = {module name: H}. Loads one layer file at a time (MiMo-9B: 25 GB total)."""
    def __init__(self, d, dev):
        self.d, self.dev, self.cur, self.cache = d, dev, None, {}
        self.files = {int(f[5:8]): os.path.join(d, f) for f in os.listdir(d) if re.fullmatch(r"layer\d{3}\.pt", f)}
    def _layer(self, name):
        m = re.search(r"layers\.(\d+)\.", name)
        return int(m.group(1)) if m else None
    def _load(self, name):
        li = self._layer(name)
        if li not in self.files:
            return False
        if li != self.cur:
            cache = torch.load(self.files[li], map_location="cpu", weights_only=False)
            keys = {}
            for key in cache:
                c = canon(key)
                if c in keys:
                    raise ValueError(f"ambiguous Hessian module {c!r}: {keys[c]!r}, {key!r}")
                keys[c] = key
            self.cache, self.keys, self.cur = cache, keys, li
        return True
    def __contains__(self, name):
        return self._load(name) and canon(name) in self.keys
    def host_tensor(self, name):
        if name not in self:
            raise KeyError(name)
        return self.cache[self.keys[canon(name)]]
    def __getitem__(self, name):
        return self.host_tensor(name).to(self.dev)


@torch.no_grad()
def capture(model, tok, mods, calib, ncal, seqlen, seed, dev, want_h=True):
    text = open(calib).read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    rng = random.Random(seed)
    starts = [rng.randrange(0, ids.numel() - seqlen - 1) for _ in range(ncal)]
    H, cnt, seen = {}, {}, {}
    hooks = []
    for name, mod in mods.items():
        n = mod.weight.shape[1]
        if want_h:
            H[name] = torch.zeros(n, n, device=dev, dtype=torch.float32)
        cnt[name] = 0

        def hk(mod, inp, name=name):
            x = inp[0].reshape(-1, inp[0].shape[-1])
            seen.setdefault(name, (id(inp[0]), inp[0].data_ptr(), tuple(inp[0].shape)))
            if not want_h:
                return
            if x.dtype in (torch.bfloat16, torch.float16) and torch.version.hip:
                H[name] += torch.mm(x.T, x, out_dtype=torch.float32)
            else:
                H[name].addmm_(x.float().T, x.float())
            cnt[name] += x.shape[0]
        hooks.append(mod.register_forward_pre_hook(hk))
    for i, s in enumerate(starts):
        if i == 1:
            first = dict(seen)
        model(ids[s:s + seqlen][None].to(dev))
        if i % 16 == 0:
            log(f"calib {i}/{ncal}")
    for h in hooks:
        h.remove()
    if ncal == 1:
        first = dict(seen)
    for k in H:
        H[k] /= max(1, cnt[k])
    return H, first, dict(ncal=ncal, seqlen=seqlen, seed=seed, calib=calib)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--policy", default=None, help='json {"overrides": {"regex on layers.N.path": K | "bf16"}}')
    ap.add_argument("--calib", required=True, help="Calibration text file; not included in this distribution")
    ap.add_argument("--ncal", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib-seed", type=int, default=20261014)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hessians", default=None, help="cache file (load if exists, else save)")
    ap.add_argument("--save-fake", default=None, help="also write a BF16 fake-quant checkpoint (dequantized HTR) here")
    ap.add_argument("--viterbi", default="auto", choices=["auto", "v1", "ref"])
    ap.add_argument("--seam", default="qtip", choices=list(encode.SEAMS),
                    help="tail-biting seam rule: 'qtip' (default) or 'htr' (earlier-release compatibility); "
                         "changes codes only (wire and decoders unchanged); 'v1' supports 'htr' only")
    ap.add_argument("--scale-refit", action="store_true",
                    help="opt-in post-encode refit of the FP16 row scales (htr.scale_refit, encoder damping): codes, "
                         "signs, wire and decoders unchanged. Applied after encoding or after loading a --cache group, "
                         "whose files keep the unrefit codes, so a cached encode can be refit without re-encoding. "
                         "K2..K5 only")
    ap.add_argument("--cache", default=None, help="shared per-group result dir (multi-GPU workers: atomic claim + save; final run assembles)")
    ap.add_argument("--require-cache-inputs", action="store_true", default=argparse.SUPPRESS,
                    help="refuse legacy group packets with no weight/Hessian digests; new packets always verify")
    ap.add_argument("--order", default="up", choices=["up", "down"], help="group order for this worker")
    ap.add_argument("--worker-only", action="store_true", help="encode unclaimed groups into --cache and exit (no checkpoint write)")
    ap.add_argument("--offload", action="store_true", help="keep the model on CPU; move one group's weights/Hessians to the GPU at a time")
    args = ap.parse_args()
    if args.viterbi == "v1" and args.seam != "htr":
        ap.error("--viterbi v1 requires --seam htr; use auto or ref for the QTIP seam")
    dev = DEVICE
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from . import vit_v1, vit_v2
    encode.VITERBI = {"auto": vit_v2.viterbi_auto, "v1": vit_v1.viterbi, "ref": None}[args.viterbi]
    policy = json.load(open(args.policy)) if args.policy else {}
    t0 = time.time()
    files, cidx = ckpt_index(args.model)
    cmap = {}
    for k in cidx:
        if k.endswith(".weight") and not any(p in k for p in SKIP_PARTS[:4]):
            cmap.setdefault(canon(k), []).append(k)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    mdev = "cpu" if args.offload else dev
    model = model.to(mdev).eval()
    mods, kmap, escapes = {}, {}, []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or "layers." not in name or any(p in name for p in SKIP_PARTS):
            continue
        c = canon(name)
        ck = cmap.get(c)
        if not ck or len(ck) != 1:
            escapes.append((name, "no unique checkpoint tensor")); continue
        m, n = mod.weight.shape
        K = resolve_policy(policy, c, args.bits)
        if K is None or m % 128 or n % 128:
            escapes.append((name, "policy" if K is None else f"shape {m}x{n}")); continue
        mods[name] = mod
        kmap[name] = (K, ck[0])
    if args.scale_refit:
        bad = sorted({K for K, _ in kmap.values()} - set(REFIT_WIDTHS))
        if bad:
            ap.error(f"--scale-refit supports K{REFIT_WIDTHS[0]}..K{REFIT_WIDTHS[-1]} only; the policy selects K {bad}")
    log(f"targets {len(mods)} params {sum(m.weight.numel() for m in mods.values())/1e6:.1f}M escapes {len(escapes)}")
    if args.hessians and os.path.isdir(args.hessians):
        H = LayerDirHessians(args.hessians, dev)
        meta = dict(source=args.hessians, format="e08-layerdir")
        _, first, _ = capture(model, tok, mods, args.calib, 1, 64, args.calib_seed, mdev, want_h=False)   # input identity only
    elif args.hessians and os.path.exists(args.hessians):
        H = torch.load(args.hessians)
        meta = H.pop("__meta__", {})
        H = {k: v.to(dev) for k, v in H.items() if k in mods}
        _, first, _ = capture(model, tok, mods, args.calib, 1, 64, args.calib_seed, mdev, want_h=False)   # input identity only
    else:
        assert not args.offload, "Hessian capture needs the model on the GPU (drop --offload or pass --hessians)"
        H, first, meta = capture(model, tok, mods, args.calib, args.ncal, args.seqlen, args.calib_seed, dev)
        if args.hessians:
            torch.save({**{k: v.cpu() for k, v in H.items()}, "__meta__": meta}, args.hessians)
    missing = [k for k in mods if k not in H]
    assert not missing, missing[:5]
    # input groups: same parent module and the same input tensor object
    groups = {}
    for name in mods:
        parent = name.rsplit(".", 1)[0]
        groups.setdefault((parent, first[name][0], first[name][1]), []).append(name)
    log(f"input groups {len(groups)}; capture/load {time.time()-t0:.0f}s")
    out_t, report, fake = {}, {}, {}
    glist = sorted(groups.items(), key=lambda kv: kv[1][0])
    gidx = list(range(len(glist)))
    if args.order == "down":
        gidx = gidx[::-1]
    if args.cache:
        os.makedirs(args.cache, exist_ok=True)
        json.dump(dict(groups=[names for _, names in glist], bits=args.bits, seed=args.seed, seam=args.seam),
                  open(os.path.join(args.cache, f"plan.{os.getpid()}.json"), "w"))
    host = os.uname().nodename + ":" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    warned_legacy_inputs = False
    for gi in gidx:
        gk, names = glist[gi]
        Ks = {kmap[nm][0] for nm in names}
        K = max(Ks)
        t = time.time()
        cpath = os.path.join(args.cache, f"g{gi:04d}.pt") if args.cache else None
        group_seed = args.seed + 104729 * gi
        cached = None
        if cpath and os.path.exists(cpath):
            cached = torch.load(cpath, map_location="cpu", weights_only=False)
            if not cache_matches(cached, names, K, args.seam):
                raise ValueError(f"cache group {gi} has different names, K or seam; "
                                 "use a matching cache (legacy caches require --seam htr)")
        elif cpath:
            try:
                fd = os.open(cpath + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{host} pid {os.getpid()} {time.strftime('%F %T')}\n".encode()); os.close(fd)
            except FileExistsError:
                if args.worker_only:
                    continue
                while not os.path.exists(cpath):          # assembler: wait for the claiming worker
                    time.sleep(30)
                cached = torch.load(cpath, map_location="cpu", weights_only=False)
                if not cache_matches(cached, names, K, args.seam):
                    raise ValueError(f"cache group {gi} has different names, K or seam; "
                                     "use a matching cache (legacy caches require --seam htr)")
        if cached is not None:
            check_cache_seed(cached, names, [mods[nm].weight.shape for nm in names], group_seed)
            if "input_identity" in cached:
                check_input_identity(cached["input_identity"], group_input_identity(names, mods, H))
            elif getattr(args, "require_cache_inputs", False):
                raise ValueError("legacy cache has no weight/Hessian identity; use a fresh cache")
            elif not warned_legacy_inputs:
                log("warning: legacy cache has unverified weights/Hessians; use --require-cache-inputs to refuse")
                warned_legacy_inputs = True
            res, rels = cached["res"], cached["rel"]
        else:
            jobs = [dict(W=mods[nm].weight.data.to(dev).float(), H=H[nm]) for nm in names]
            identity = group_input_identity(names, mods, H) if cpath else None
            res = encode.encode_group(jobs, K, seed=group_seed, shared_su=True, seam=args.seam)
            rels = [rel_proxy(jb["W"], r["What"], jb["H"]) for jb, r in zip(jobs, res)]
            del jobs
            if cpath:
                keep = [{k: r[k].contiguous().cpu() for k in ("qweight", "scales", "su", "sv")} | {"What": r["What"].bfloat16().cpu()} for r in res]
                torch.save(dict(names=names, K=K, seam=args.seam, group_seed=group_seed, input_identity=identity, res=keep, rel=rels, host=host, seconds=time.time() - t), cpath + ".tmp")
                os.replace(cpath + ".tmp", cpath)
                if args.worker_only:
                    log(f"group {gi+1}/{len(glist)} K{K} {names} {time.time()-t:.0f}s -> cache")
                    del res
                    torch.cuda.empty_cache()
                    continue
        if args.worker_only:
            continue
        if args.scale_refit:            # post-encode: the group cache above keeps the unrefit codes
            jobs = [dict(W=mods[nm].weight.data.to(dev).float(), H=H[nm]) for nm in names]
            res, rels = refit_group(jobs, res)
            del jobs
        for nm, r, rel in zip(names, res, rels):
            ck = kmap[nm][1]
            pre = ck[:-7]
            od = "cpu" if args.offload else r["qweight"].device
            out_t[pre + ".htr_qweight"] = r["qweight"].contiguous().to(od)
            out_t[pre + ".htr_scales"] = r["scales"].contiguous().to(od)
            out_t[pre + ".htr_su"] = r["su"].contiguous().to(od)
            out_t[pre + ".htr_sv"] = r["sv"].contiguous().to(od)
            report[pre] = dict(K=K, shape=list(mods[nm].weight.shape), rel_proxy=rel, group=gi)
            if args.scale_refit:
                report[pre]["scale_refit"] = r["scale_refit"]
            if args.save_fake:
                fake[ck] = r["What"].bfloat16().cpu()
        log(f"group {gi+1}/{len(glist)} K{K} {names} {time.time()-t:.0f}s" + (" refit" if args.scale_refit else "")
            + (" (cache)" if cached is not None else ""))
        del res
        torch.cuda.empty_cache()
    del model
    if args.worker_only:
        log(f"worker done {time.time()-t0:.0f}s"); return
    replaced = {kmap[nm][1] for nm in mods}
    _write(args.model, args.out, files, replaced, out_t, report, escapes, meta, args, quantized=True)
    if args.save_fake:
        _write(args.model, args.save_fake, files, replaced, fake, report, escapes, meta, args, quantized=False)
    log(f"done {time.time()-t0:.0f}s")


def _write(src, out, files, replaced, new_t, report, escapes, meta, args, quantized):
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if os.path.isfile(p) and not f.endswith(".safetensors") and f != "model.safetensors.index.json":
            shutil.copy2(p, os.path.join(out, f))
    wmap, total = {}, 0
    pending = dict(new_t)
    for si, f in enumerate(files):
        ts = {}
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                if k in replaced:
                    if quantized:
                        pre = k[:-7]
                        for suf in (".htr_qweight", ".htr_scales", ".htr_su", ".htr_sv"):
                            ts[pre + suf] = pending.pop(pre + suf)
                    else:
                        ts[k] = pending.pop(k)
                else:
                    ts[k] = fh.get_tensor(k)
        name = f"model-{si+1:05d}-of-{len(files):05d}.safetensors"
        save_file(ts, os.path.join(out, name), metadata={"format": "pt"})
        for k, v in ts.items():
            wmap[k] = name
            total += v.numel() * v.element_size()
    assert not pending, list(pending)[:5]
    json.dump({"metadata": {"total_size": total}, "weight_map": wmap}, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
    cfg = json.load(open(os.path.join(src, "config.json")))
    if quantized:
        cfg["quantization_config"] = quant_config(report, args.seam, getattr(args, "scale_refit", False))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    nq = sum(v["shape"][0] * v["shape"][1] for v in report.values())
    bits = sum(v["shape"][0] * v["shape"][1] * v["K"] + 16 * v["shape"][0] + (v["shape"][0] + v["shape"][1]) for v in report.values())
    json.dump(dict(quantized=quantized, modules=report, escapes=escapes, calib=meta, args=vars(args),
                   target_params=nq, target_bpw=bits / max(1, nq), total_bytes=total),
              open(os.path.join(out, "htr_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
