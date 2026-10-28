"""htr.encode_q38: packed HTR-W1 encode of the Qwen3.8 (qwen4_exp) Bf16Core DENSE TRUNK, straight from the release safetensors
(no HF model class exists for qwen4_exp). E10 step 3 (2026-10-16).

Targets = the E09 set (312 tensors): per decoder layer linear_attn.{in_proj_qkv,in_proj_z,out_proj},
self_attn.{q_proj,k_proj,v_proj,o_proj,indexer.index_qk_proj}, mlp.shared_expert.{gate_proj,up_proj,down_proj}.
Hessians = E09 capture (vLLM module inputs, ncal128 L2048): merged vLLM modules share one input -> one ENCODE GROUP with a
shared input-sign vector su (needed by the serving prologue): in_proj_qkvz = [in_proj_qkv, in_proj_z]; qkv_proj = [q, k, v];
gate_up_proj = [gate, up]. Same encoder/seed convention as htr.quantize (encode_group(..., seed + 104729*gi, shared_su=True)).

  worker   : encode unclaimed groups into --cache (O_EXCL claims; multi-GPU safe; resumable)
  assemble : write --out (packed: <name>.htr_{qweight,scales,su,sv}) and/or --fake-out (bf16 What in place of <name>.weight).
             Unaffected files are HARD-linked (HaloLoom's loader rejects symlinked shards); affected shards are rewritten with the
             same file names. The Quark quantization_config is kept (experts stay int4/int8); the HTR module set goes to
             config["htr_overlay"] and htr_report.json.
"""
import argparse, json, os, re, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from . import encode

TGT = re.compile(r"^model\.language_model\.layers\.(\d+)\.(linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)|"
                 r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj|indexer\.index_qk_proj)|"
                 r"mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj))\.weight$")
HMAP = {"linear_attn.in_proj_qkv": "linear_attn.in_proj_qkvz", "linear_attn.in_proj_z": "linear_attn.in_proj_qkvz",
        "self_attn.q_proj": "self_attn.qkv_proj", "self_attn.k_proj": "self_attn.qkv_proj",
        "self_attn.v_proj": "self_attn.qkv_proj", "mlp.shared_expert.gate_proj": "mlp.shared_expert.gate_up_proj",
        "mlp.shared_expert.up_proj": "mlp.shared_expert.gate_up_proj"}
ORDER = ["in_proj_qkv", "in_proj_z", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]   # member order inside a merged group


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def plan(model):
    wm = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    tg = {k: wm[k] for k in wm if TGT.match(k)}
    assert len(tg) == 312, len(tg)
    groups = {}
    for k in tg:
        m = TGT.match(k)
        li, sub = int(m.group(1)), m.group(2)
        groups.setdefault((li, HMAP.get(sub, sub)), []).append(k)
    out = []
    for (li, g) in sorted(groups):
        names = sorted(groups[(li, g)], key=lambda k: ORDER.index(k.split(".")[-2]) if k.split(".")[-2] in ORDER else 0)
        out.append(dict(layer=li, group=g, names=names))
    return wm, out


class Src:
    def __init__(self, model, wm):
        self.model, self.wm, self.h = model, wm, {}

    def get(self, k):
        f = self.wm[k]
        if f not in self.h:
            self.h[f] = safe_open(os.path.join(self.model, f), "pt", device="cpu")
        return self.h[f].get_tensor(k)


@torch.no_grad()
def cmd_worker(a):
    from . import vit_v1, vit_v2
    encode.VITERBI = {"auto": vit_v2.viterbi_auto, "v1": vit_v1.viterbi, "ref": None}[a.viterbi]
    wm, groups = plan(a.model)
    src = Src(a.model, wm)
    os.makedirs(a.cache, exist_ok=True)
    json.dump(dict(groups=groups, bits=a.bits, seed=a.seed, hdir=a.hdir, hmix=a.hmix), open(os.path.join(a.cache, "plan.json"), "w"), indent=0)
    host = os.uname().nodename + ":" + torch.cuda.get_device_name(0)
    order = range(len(groups)) if a.order == "up" else range(len(groups) - 1, -1, -1)
    if a.groups:   # bounded canary subset (same cache file names/seeds as the full run, so the canary groups are reused)
        sel = parse_groups(a.groups, len(groups))
        order = [gi for gi in order if gi in sel]
    t0, done = time.time(), 0
    for gi in order:
        g = groups[gi]
        cpath = os.path.join(a.cache, f"g{gi:04d}.pt")
        if os.path.exists(cpath):
            continue
        try:
            fd = os.open(cpath + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{host} pid {os.getpid()} {time.strftime('%F %T')}\n".encode()); os.close(fd)
        except FileExistsError:
            continue
        t = time.time()
        H = load_h(a, f"language_model.model.layers.{g['layer']}.{g['group']}.pt")
        jobs = []
        for k in g["names"]:
            w = src.get(k)
            assert w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[1] == H.shape[0], (k, w.shape, H.shape)
            jobs.append(dict(W=w.to("cuda").float(), H=H))
        res = encode.encode_group(jobs, a.bits, seed=a.seed + 104729 * gi, shared_su=True)
        rels = []
        Hd = H.double()
        for jb, r in zip(jobs, res):
            E_ = jb["W"].double() - r["What"].double()
            rels.append(float(((E_ @ Hd) * E_).sum() / ((jb["W"].double() @ Hd) * jb["W"].double()).sum()))
        keep = [{k: r[k].contiguous().cpu() for k in ("qweight", "scales", "su", "sv")} | {"What": r["What"].bfloat16().cpu()} for r in res]
        torch.save(dict(names=g["names"], K=a.bits, seed=a.seed + 104729 * gi, res=keep, rel=rels, host=host,
                        seconds=time.time() - t), cpath + ".tmp")
        os.replace(cpath + ".tmp", cpath)
        done += 1
        log(f"group {gi+1}/{len(groups)} L{g['layer']} {g['group']} K{a.bits} rel {['%.5f' % r for r in rels]} {time.time()-t:.0f}s "
            f"(this worker: {done} groups, {time.time()-t0:.0f}s)")
        del jobs, res, H, Hd
        torch.cuda.empty_cache()
    log(f"worker done {time.time()-t0:.0f}s")


def load_h(a, fname):
    """--hmix 'dirA=wA,dirB=wB' -> (sum w_i H_i) / sum w_i (each H_i = E[x x^T] over its own tokens); else --hdir."""
    if not a.hmix:
        return torch.load(os.path.join(a.hdir, fname)).to("cuda").float()
    H, wsum = None, 0.0
    for part in a.hmix.split(","):
        d, _, w = part.rpartition("=")
        w = float(w); x = torch.load(os.path.join(d, fname)).to("cuda").float() * w
        H = x if H is None else H.add_(x); wsum += w
    return H.div_(wsum)


def parse_groups(spec, n):
    sel = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        sel.update(range(int(lo), int(hi or lo) + 1))
    assert sel and max(sel) < n, (spec, n)
    return sel


def _assemble_one(a, wm, groups, packed, out):
    src = os.path.abspath(a.model)
    new, report = {}, {}
    gis = list(range(len(groups)))
    if a.partial:   # bounded canary: only the cached groups become HTR; every other target stays BF16 (byte-identical tensors)
        gis = [gi for gi in gis if os.path.exists(os.path.join(a.cache, f"g{gi:04d}.pt"))]
        assert gis, "no cached groups"
        log(f"PARTIAL assemble: {len(gis)}/{len(groups)} groups {gis}")
    for gi in gis:
        g = groups[gi]
        c = torch.load(os.path.join(a.cache, f"g{gi:04d}.pt"), map_location="cpu", weights_only=False)
        assert c["names"] == g["names"] and (c["K"] == a.bits or a.mixed), (gi, c["K"], c["names"], g["names"])
        su0 = c["res"][0]["su"]
        assert all(torch.equal(r["su"], su0) for r in c["res"]), f"group {gi} members disagree on su"
        for k, r, rel in zip(c["names"], c["res"], c["rel"]):
            pre = k[:-len(".weight")]
            if packed:
                for suf in ("qweight", "scales", "su", "sv"):
                    new[f"{pre}.htr_{suf}"] = r[suf].contiguous()
            else:
                new[k] = r["What"].contiguous()
            report[pre] = dict(K=int(c["K"]), shape=[int(r["What"].shape[0]), int(r["What"].shape[1])], rel_proxy=rel, group=gi,
                               seconds=c["seconds"] / len(c["names"]), host=c["host"])
    targets = {g2 for gi in gis for g2 in groups[gi]["names"]}
    affected = sorted({wm[k] for k in targets})
    assert not os.path.exists(out), out
    tmp = out + ".partial"
    os.makedirs(tmp)
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        for dn in dirs:
            os.makedirs(os.path.join(tmp, rel, dn), exist_ok=True)
        for fn in files:
            if rel == "." and (fn in affected or fn in ("model.safetensors.index.json", "config.json")):
                continue
            os.link(os.path.join(root, fn), os.path.join(tmp, rel, fn))
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wmap = dict(idx["weight_map"])
    for f in affected:
        with safe_open(os.path.join(src, f), "pt", device="cpu") as h:
            md = h.metadata()
            ts = {}
            for k in h.keys():
                if k in targets:
                    if packed:
                        pre = k[:-len(".weight")]
                        del wmap[k]
                        for suf in ("qweight", "scales", "su", "sv"):
                            ts[f"{pre}.htr_{suf}"] = new.pop(f"{pre}.htr_{suf}")
                            wmap[f"{pre}.htr_{suf}"] = f
                    else:
                        q = new.pop(k)
                        t = h.get_tensor(k)
                        assert q.shape == t.shape and q.dtype == t.dtype, k
                        ts[k] = q
                else:
                    ts[k] = h.get_tensor(k)
        save_file(ts, os.path.join(tmp, f), metadata=md)
        log("wrote", f)
    assert not new, list(new)[:5]
    total = 0
    for f in sorted(set(wmap.values())):
        with safe_open(os.path.join(tmp, f), "pt") as fh:
            for k in fh.keys():
                s = fh.get_slice(k); n = 1
                for d in s.get_shape():
                    n *= d
                total += n * {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U8": 1, "I8": 1, "I64": 8, "F8_E4M3": 1}.get(s.get_dtype(), 1)
    idx["weight_map"] = wmap
    idx.setdefault("metadata", {})["total_size"] = total
    json.dump(idx, open(os.path.join(tmp, "model.safetensors.index.json"), "w"), indent=1)
    cfg = json.load(open(os.path.join(src, "config.json")))
    if packed:   # v1: inside quantization_config (read by htr.vllm_plugin's Quark overlay selector; quant_method stays "quark")
        assert cfg["quantization_config"]["quant_method"] == "quark" and "htr_overlay" not in cfg["quantization_config"]
        cfg["quantization_config"]["htr_overlay"] = {"wire": "HTR-W1", "generator": "rdna24", "L": 16, "tile": 16, "rht_block": 128,
                                                     "vllm_prefix": "language_model.model.layers.",
                                                     "modules": {k: {"K": v["K"]} for k, v in report.items()}}
    json.dump(cfg, open(os.path.join(tmp, "config.json"), "w"), indent=2)
    nq = sum(v["shape"][0] * v["shape"][1] for v in report.values())
    bits = sum(v["shape"][0] * v["shape"][1] * v["K"] + 16 * v["shape"][0] + (v["shape"][0] + v["shape"][1]) for v in report.values())
    json.dump(dict(tool="htr.encode_q38", quantized=packed, partial=bool(a.partial), groups_encoded=gis, groups_total=len(groups), source=src, hdir=a.hdir, hmix=getattr(a, "hmix", None), bits=a.bits, seed=a.seed, modules=report,
                   target_params=nq, target_bpw=bits / nq, affected_shards=affected, total_bytes=total,
                   mean_rel_proxy=sum(v["rel_proxy"] for v in report.values()) / len(report),
                   encode_seconds=sum(v["seconds"] for v in report.values())),
              open(os.path.join(tmp, "htr_report.json"), "w"), indent=1)
    os.rename(tmp, out)
    log(f"{'packed' if packed else 'fake'} -> {out} total {total/1e9:.3f} GB target {nq/1e6:.1f}M params {bits/nq:.4f} bpw")


def cmd_assemble(a):
    wm, groups = plan(a.model)
    if a.out:
        _assemble_one(a, wm, groups, True, os.path.abspath(a.out))
    if a.fake_out:
        _assemble_one(a, wm, groups, False, os.path.abspath(a.fake_out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["worker", "assemble"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--hdir", default=None, help="Captured Hessians directory; required unless --hmix supplies it")
    ap.add_argument("--hmix", default=None, help="worker: weighted Hessian mixture 'dirA=wA,dirB=wB' (overrides --hdir)")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--order", default="up", choices=["up", "down"])
    ap.add_argument("--viterbi", default="auto", choices=["auto", "v1", "ref"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--fake-out", default=None)
    ap.add_argument("--groups", default=None, help="worker: encode only these group indices, e.g. 0-3,12-16")
    ap.add_argument("--partial", action="store_true", help="assemble: only cached groups become HTR (canary)")
    ap.add_argument("--mixed", action="store_true", help="assemble: per-group K taken from each cache file (e.g. K5 early layers + K4 rest)")
    a = ap.parse_args()
    {"worker": cmd_worker, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    main()
