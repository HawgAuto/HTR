"""htr.lmhead: add an HTR-W1 lm_head to an existing HTR checkpoint (E10 kernel-polish lever 1, 2026-10-16).

The decode profile (artifacts/e10/prof) shows the bf16 lm_head is 27-31% of a MiMo-9B HTR decode step (2.03 GB @ 233 GB/s) while
the HTR GEMV already runs at ~214 GB/s. Encoding the lm_head with the same codec + kernel removes most of that read.

  capture : H = E[x x^T] of the lm_head input (final-norm hidden states), same calib policy as E08 (wiki.train, 128x2048,
            seed 20261014) -> --hessian file.
  worker  : encode row chunks of lm_head.weight (each chunk = rows % 128 == 0) into --cache with O_EXCL claims (multi-GPU safe).
            Rows are independent in block-LDLQ (feedback runs along input columns), so chunking is exact except that every
            chunk draws the same output-sign vector sv (seed + 7919, length = chunk rows); su (input signs) is shared by
            construction (generator(seed)), as one prologue serves the whole head.
  assemble: new checkpoint dir = --base (packed HTR) with lm_head.weight replaced by lm_head.htr_*; unchanged shards are
            symlinked; config modules["lm_head"] = {"K": K}. Optional --fake-base/--fake-out writes the bf16 fake-quant twin.
"""
import argparse, json, os, random, shutil, time, glob
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from . import encode

KEY = "lm_head.weight"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def shard_of(model_dir, key):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    return os.path.join(model_dir, idx[key])


def load_head(model_dir):
    with safe_open(shard_of(model_dir, KEY), "pt") as fh:
        return fh.get_tensor(KEY)


@torch.no_grad()
def cmd_capture(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    W = model.lm_head.weight
    assert torch.equal(W.detach().cpu(), load_head(a.model)), "model.lm_head is not the checkpoint lm_head.weight"
    n = W.shape[1]
    H = torch.zeros(n, n, device="cuda", dtype=torch.float32)
    cnt = [0]

    def hk(mod, inp):
        x = inp[0].reshape(-1, n)
        H.addmm_(x.float().T, x.float())
        cnt[0] += x.shape[0]
    h = model.lm_head.register_forward_pre_hook(hk)
    ids = tok(open(a.calib).read(), return_tensors="pt").input_ids[0]
    rng = random.Random(a.calib_seed)
    starts = [rng.randrange(0, ids.numel() - a.seqlen - 1) for _ in range(a.ncal)]
    t0 = time.time()
    for i, s in enumerate(starts):
        model(ids[s:s + a.seqlen][None].to("cuda"), logits_to_keep=0)
        if i % 16 == 0:
            log(f"calib {i}/{a.ncal}")
    h.remove()
    H /= cnt[0]
    meta = dict(ncal=a.ncal, seqlen=a.seqlen, seed=a.calib_seed, calib=a.calib, tokens=cnt[0], seconds=time.time() - t0,
                device=torch.cuda.get_device_name(0))
    torch.save({"lm_head": H.cpu(), "__meta__": meta}, a.hessian)
    log("saved", a.hessian, meta)


def plan(m, chunk):
    assert m % 128 == 0 and chunk % 128 == 0
    return [(r, min(r + chunk, m)) for r in range(0, m, chunk)]


@torch.no_grad()
def cmd_worker(a):
    from . import vit_v1, vit_v2
    encode.VITERBI = vit_v2.viterbi_auto
    W = load_head(a.model)
    m, n = W.shape
    H = torch.load(a.hessian)["lm_head"].to("cuda")
    os.makedirs(a.cache, exist_ok=True)
    chunks = plan(m, a.chunk)
    host = os.uname().nodename + ":" + torch.cuda.get_device_name(0)
    order = range(len(chunks)) if a.order == "up" else range(len(chunks) - 1, -1, -1)
    t0 = time.time()
    for ci in order:
        r0, r1 = chunks[ci]
        cpath = os.path.join(a.cache, f"K{a.bits}_c{ci:03d}.pt")
        if os.path.exists(cpath):
            continue
        try:
            fd = os.open(cpath + ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{host} pid {os.getpid()} {time.strftime('%F %T')}\n".encode()); os.close(fd)
        except FileExistsError:
            continue
        t = time.time()
        Wc = W[r0:r1].to("cuda").float()
        r = encode.encode_group([dict(W=Wc, H=H)], a.bits, seed=a.seed, shared_su=True)[0]
        E_ = Wc.double() - r["What"].double(); Hd = H.double()
        rel = float(((E_ @ Hd) * E_).sum() / ((Wc.double() @ Hd) * Wc.double()).sum())
        keep = {k: r[k].contiguous().cpu() for k in ("qweight", "scales", "su", "sv")} | {"What": r["What"].bfloat16().cpu()}
        torch.save(dict(rows=(r0, r1), K=a.bits, seed=a.seed, res=keep, rel=rel, host=host, seconds=time.time() - t),
                   cpath + ".tmp")
        os.replace(cpath + ".tmp", cpath)
        log(f"chunk {ci+1}/{len(chunks)} rows {r0}:{r1} K{a.bits} rel {rel:.5f} {time.time()-t:.0f}s")
        del Wc, r, E_
        torch.cuda.empty_cache()
    log(f"worker done {time.time()-t0:.0f}s")


def _rewrite(base, out, key_old, new_tensors, cfg_update=None, report_update=None):
    """out = base with key_old removed from its shard and new_tensors added there; other shards symlinked."""
    os.makedirs(out, exist_ok=True)
    idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    tgt = wm[key_old]
    for f in os.listdir(base):
        p = os.path.join(base, f)
        if not os.path.isfile(p) or f in ("model.safetensors.index.json", "config.json", "htr_report.json"):
            continue
        dst = os.path.join(out, f)
        if os.path.lexists(dst):
            os.remove(dst)
        if f.endswith(".safetensors") and f != tgt:
            os.symlink(os.path.realpath(p), dst)
        elif not f.endswith(".safetensors"):
            shutil.copy2(p, dst)
    ts = {}
    with safe_open(os.path.join(base, tgt), "pt") as fh:
        for k in fh.keys():
            if k != key_old:
                ts[k] = fh.get_tensor(k)
    ts.update(new_tensors)
    save_file(ts, os.path.join(out, tgt), metadata={"format": "pt"})
    del wm[key_old]
    for k, v in new_tensors.items():
        wm[k] = tgt
    total = 0
    for f in sorted(set(wm.values())):
        with safe_open(os.path.join(out, f), "pt") as fh:
            for k in fh.keys():
                t = fh.get_slice(k)
                sz = 1
                for d in t.get_shape():
                    sz *= d
                total += sz * {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U8": 1, "I8": 1, "I64": 8}[t.get_dtype()]
    idx["metadata"]["total_size"] = total
    json.dump(idx, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
    cfg = json.load(open(os.path.join(base, "config.json")))
    if cfg_update:
        cfg_update(cfg)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    rp = os.path.join(base, "htr_report.json")
    if os.path.exists(rp):
        rep = json.load(open(rp))
        if report_update:
            report_update(rep, total)
        json.dump(rep, open(os.path.join(out, "htr_report.json"), "w"), indent=1)
    return total


def cmd_assemble(a):
    W = load_head(a.model)
    m, n = W.shape
    chunks = plan(m, a.chunk)
    parts = []
    for ci, (r0, r1) in enumerate(chunks):
        c = torch.load(os.path.join(a.cache, f"K{a.bits}_c{ci:03d}.pt"), weights_only=False)
        assert c["rows"] == (r0, r1) and c["K"] == a.bits and c["seed"] == a.seed, (ci, c["rows"], c["K"])
        parts.append(c)
    su = parts[0]["res"]["su"]
    assert all(torch.equal(p["res"]["su"], su) for p in parts), "chunks disagree on the input sign vector"
    new = {"lm_head.htr_qweight": torch.cat([p["res"]["qweight"] for p in parts]).contiguous(),
           "lm_head.htr_scales": torch.cat([p["res"]["scales"] for p in parts]).contiguous(),
           "lm_head.htr_su": su.contiguous(),
           "lm_head.htr_sv": torch.cat([p["res"]["sv"] for p in parts]).contiguous()}
    assert new["lm_head.htr_qweight"].shape[0] == m // 16 and new["lm_head.htr_scales"].shape[0] == m
    rels = [p["rel"] for p in parts]
    rel = sum(r * (p["rows"][1] - p["rows"][0]) for r, p in zip(rels, parts)) / m   # row-weighted (proxy)
    secs = sum(p["seconds"] for p in parts)

    def cfg_q(cfg):
        cfg["quantization_config"]["modules"]["lm_head"] = {"K": a.bits}

    def rep_q(rep, total):
        rep["modules"]["lm_head"] = dict(K=a.bits, shape=[m, n], rel_proxy=rel, chunks=len(parts), chunk_rows=a.chunk,
                                         chunk_rel=rels, encode_seconds=secs, hessian=a.hessian, seed=a.seed)
        rep["total_bytes"] = total
        nq = sum(v["shape"][0] * v["shape"][1] for v in rep["modules"].values())
        bits = sum(v["shape"][0] * v["shape"][1] * v["K"] + 16 * v["shape"][0] + (v["shape"][0] + v["shape"][1])
                   for v in rep["modules"].values())
        rep["target_params"], rep["target_bpw"] = nq, bits / nq
        rep["lm_head_added"] = dict(tool="htr.lmhead", base=a.base, time=time.strftime("%F %T"))
    tot = _rewrite(a.base, a.out, KEY, new, cfg_q, rep_q)
    log(f"packed -> {a.out} total {tot/1e9:.3f} GB lm_head rel_proxy {rel:.5f} encode {secs:.0f}s")
    if a.fake_base:
        What = torch.cat([p["res"]["What"] for p in parts]).contiguous()
        assert What.shape == W.shape
        tot = _rewrite(a.fake_base, a.fake_out, KEY, {KEY: What})
        log(f"fake -> {a.fake_out} total {tot/1e9:.3f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["capture", "worker", "assemble"])
    ap.add_argument("--model", required=True, help="bf16 source checkpoint (lm_head.weight + calibration forward)")
    ap.add_argument("--hessian", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234 + 104729 * 9973)
    ap.add_argument("--chunk", type=int, default=24832)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--order", default="up", choices=["up", "down"])
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fake-base", default=None)
    ap.add_argument("--fake-out", default=None)
    ap.add_argument("--calib", default="calibration.txt", help="Calibration text file for capture")
    ap.add_argument("--ncal", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib-seed", type=int, default=20261014)
    a = ap.parse_args()
    {"capture": cmd_capture, "worker": cmd_worker, "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    main()
