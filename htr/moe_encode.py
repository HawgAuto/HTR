#!/usr/bin/env python3
"""E10 P2: full-model routed-expert HTR encode (48 layers), packed-only chunks with in-process exact decode checks.

Algorithm, seeds, chunking, Hessians and proxy rows are P1 encode_experts.py (byte copy in source/p1/, sealed in
artifacts/p1/widths-K345-20261020/ENCODER.sha256). Changes, all outside the numerics:
  * inputs : BF16 experts per layer from the full source checkpoint (--src, default the verified Seagate copy of
             Qwen3.8-Flash-Next@de4b8e4d); calibration dump from --dump (<dump>/L<L>/w*.pt, P1 layout).
  * output : packed HTR-W1 wire + proxy rows, NO BF16 reconstruction. P1 kept What for its BF16 eval hook (~157 of
             207 MB per K5 chunk); for 48 layers that is ~241 GB per width.
  * verify : before What is dropped, EVERY matrix is decoded with wire.dequant on the encoding device and must be
             bitwise identical to What in BF16 (P1: 3 sampled matrices per chunk, after the fact). The sha256 of each
             BF16 What is kept so later decoders/kernels can be checked against the encoder's exact reconstruction.
  * space  : refuses to start, or to write a chunk, without room for the remaining chunks plus a margin.
  * memory : BF16 experts are read per 16-expert chunk with safetensors get_slice (P1 held the whole layer, ~5 GB);
             calibration windows are copied into one preallocated tensor (P1: window list + torch.cat, a transient
             second copy of ~3.4 GB per layer). Same values and row order; P1 K5 ran its 56 GiB memcg at the limit.
"""
import argparse, glob, hashlib, json, os, shutil, sys, time
import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import encode, vit_v2, wire
from . import moe_io as M

FORMAT = "HTR-W1-P2-PACKED-1"
ROLES = ("gate", "up", "down")
SHAPES = {"gate": (640, 2560), "up": (640, 2560), "down": (2560, 640)}
FIELDS = ("qweight", "scales", "su", "sv")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rel(W, Q, H):
    E = (W - Q).double(); Wd = W.double(); Hd = H.double()
    return float(((E @ Hd) * E).sum() / ((Wd @ Hd) * Wd).sum())


def packed_bytes_per_expert(K):
    """Exact HTR-W1 payload: qweight int32 [m/16, n/16, 8K] + FP16 row scales [m] + packed signs su [n/8], sv [m/8]."""
    return sum((m // 16) * (n // 16) * 8 * K * 4 + 2 * m + n // 8 + m // 8 for m, n in SHAPES.values())


def chunk_bytes_estimate(K, chunk):
    """Serialized chunk upper estimate: payload + 2% + 1 MiB for torch.save framing and proxy rows."""
    return int(chunk * packed_bytes_per_expert(K) * 1.02) + (1 << 20)


def require_space(path, need, margin, usage=shutil.disk_usage):
    free = usage(path).free
    if free < need + margin:
        raise RuntimeError(f"insufficient space at {path}: free {free/2**30:.1f} GiB < need {need/2**30:.1f} GiB "
                           f"+ margin {margin/2**30:.1f} GiB")
    return free


def bf16_bits(t):
    return t.contiguous().view(torch.int16)


def verify_and_strip(r, dev):
    """Decode the packed wire on `dev`, require bitwise BF16 equality with the encoder's What, return
    (packed tensors on CPU, sha256 of What's BF16 bytes). Raises on any mismatch; nothing is written then."""
    packed = {f: r[f].contiguous().cpu() for f in FIELDS}
    what = r["What"].to(dev).bfloat16()
    dec = wire.dequant(*(packed[f].to(dev) for f in FIELDS)).bfloat16()
    if dec.shape != what.shape:
        raise ValueError(f"decode shape {tuple(dec.shape)} != What {tuple(what.shape)}")
    unequal = int((bf16_bits(dec) != bf16_bits(what)).sum())
    if unequal:
        raise ValueError(f"packed decode != What in {unequal} BF16 elements (bitwise)")
    return packed, hashlib.sha256(bf16_bits(what).cpu().numpy().tobytes()).hexdigest()


def strip_what(z):
    """True when no BF16/FP32 matrix-shaped reconstruction remains anywhere in a chunk record."""
    if isinstance(z, dict):
        return "What" not in z and all(strip_what(v) for v in z.values())
    if isinstance(z, (list, tuple)):
        return all(strip_what(v) for v in z)
    if torch.is_tensor(z):
        return not (z.is_floating_point() and z.dtype != torch.float16 and z.ndim == 2)
    return True


def load_dump(d, dev):
    """P1 rows in P1 order (sorted windows, concatenated), filled into one preallocated tensor on `dev`."""
    fs = sorted(glob.glob(os.path.join(d, "w*.pt")))
    if not fs:
        raise FileNotFoundError(f"no calibration windows in {d}")
    X, ids, off = None, [], 0
    for f in fs:
        z = torch.load(f, weights_only=False); x = z["x"]
        if X is None:
            X = torch.empty((len(fs) * x.shape[0], x.shape[1]), dtype=x.dtype, device=dev)
        if x.ndim != 2 or x.shape[1] != X.shape[1] or x.dtype != X.dtype or off + x.shape[0] > X.shape[0]:
            raise ValueError(f"window {f} {tuple(x.shape)} {x.dtype} does not fit the first window's layout")
        if z["ids"].shape[0] != x.shape[0]:
            raise ValueError(f"window {f}: {z['ids'].shape[0]} id rows for {x.shape[0]} x rows")
        X[off:off + x.shape[0]].copy_(x); off += x.shape[0]; ids.append(z["ids"])
    return X[:off], torch.cat(ids).to(dev).long(), len(fs)


class SourceExperts:
    """BF16 experts of one layer, read per chunk from the fused source tensors (get_slice reads only those experts).
    chunk(e0, e1) == moe_io.bf16_experts(L, src)[0][e0:e1], [1][e0:e1] without holding the layer in memory."""
    WANT = {"gate_up_proj": [512, 1280, 2560], "down_proj": [512, 2560, 640]}

    def __init__(self, L, src):
        wm = M._index(src); p = f"model.language_model.layers.{L}.mlp.experts."
        self.names = {n: p + n for n in self.WANT}
        self.fh = {n: safe_open(os.path.join(src, wm[k]), "pt") for n, k in self.names.items()}
        for n, k in self.names.items():
            s = self.fh[n].get_slice(k)
            if list(s.get_shape()) != self.WANT[n] or s.get_dtype() != "BF16":
                raise ValueError(f"L{L} {k}: {s.get_shape()} {s.get_dtype()}")

    def chunk(self, e0, e1):
        return tuple(self.fh[n].get_slice(self.names[n])[e0:e1] for n in ("gate_up_proj", "down_proj"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--src", required=True, help="Full BF16 checkpoint with fused per-layer expert tensors")
    ap.add_argument("--reference", required=True, help="Mixed INT4/INT8 reference for diagnostic proxy rows")
    ap.add_argument("--bits", type=int, required=True); ap.add_argument("--layers", default=",".join(map(str, range(48))))
    ap.add_argument("--chunk", type=int, default=16); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--rev", action="store_true", help="walk layers in reverse (second worker)")
    ap.add_argument("--x-cpu", action="store_true", help="keep the layer dump X on CPU, gather per expert")
    ap.add_argument("--max-chunks", type=int, default=0, help="stop cleanly after this many new chunks; 0 means all")
    ap.add_argument("--margin-gb", type=float, default=20.0, help="free space kept on the output filesystem")
    a = ap.parse_args()
    assert a.max_chunks >= 0 and 2 <= a.bits <= 8 and 512 % a.chunk == 0
    done_chunks = 0
    dev = "cuda"; os.makedirs(a.out, exist_ok=True)
    gpu = torch.cuda.get_device_name(0); me = f"{os.uname().nodename}:{os.getpid()}:{gpu}"
    margin = int(a.margin_gb * 2**30); per_chunk = chunk_bytes_estimate(a.bits, a.chunk)
    pf = os.path.join(a.out, "PLAN.json")
    keys = ("bits", "chunk", "seed", "damp", "dump", "src")
    if not os.path.exists(pf):
        json.dump(dict({k: getattr(a, k) for k in keys}, layers=a.layers, format=FORMAT), open(pf, "w"))
    plan = json.load(open(pf))
    assert plan.get("format") == FORMAT and all(plan[k] == getattr(a, k) for k in keys), "PLAN mismatch"
    Ls = [int(x) for x in a.layers.split(",")]
    todo = sum(not os.path.exists(os.path.join(a.out, f"L{L}", f"c{e0:03d}.pt")) for L in Ls for e0 in range(0, 512, a.chunk))
    free = require_space(a.out, todo * per_chunk, margin)
    log(f"P2 {FORMAT} K{a.bits} layers {len(Ls)} chunks to do {todo} est {todo*per_chunk/2**30:.1f} GiB "
        f"free {free/2**30:.1f} GiB [{gpu}]")
    for L in (Ls[::-1] if a.rev else Ls):
        od = os.path.join(a.out, f"L{L}"); os.makedirs(od, exist_ok=True)
        for c in glob.glob(os.path.join(od, "c*.claim")):   # drop stale claims of dead local workers
            try:
                pid = int(open(c).read().split(":")[1])
                if not os.path.exists(f"/proc/{pid}") and not os.path.exists(c[:-6] + ".pt"):
                    os.remove(c); log("stale claim removed", c)
            except Exception:
                pass
        free_c = lambda e0: not os.path.exists(os.path.join(od, f"c{e0:03d}.pt")) and not os.path.exists(os.path.join(od, f"c{e0:03d}.claim"))
        if not any(free_c(e0) for e0 in range(0, 512, a.chunk)):
            log(f"L{L} K{a.bits} nothing to claim"); continue
        t = time.time()
        X, ids, nwin = load_dump(os.path.join(a.dump, f"L{L}"), "cpu" if a.x_cpu else dev)
        S = SourceExperts(L, a.src)
        R = M.Release(L, a.reference)
        log(f"L{L} K{a.bits} [{gpu}] dump {nwin} windows rows {X.shape[0]} load {time.time()-t:.0f}s")
        for e0 in range(0, 512, a.chunk):
            if not free_c(e0):
                continue
            require_space(a.out, per_chunk, margin)
            try:
                fd = os.open(os.path.join(od, f"c{e0:03d}.claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, me.encode()); os.close(fd)
            except FileExistsError:
                continue
            tc = time.time(); es = list(range(e0, e0 + a.chunk))
            gu, dn = S.chunk(e0, e0 + a.chunk)
            jobs_gu, jobs_dn, meta = [], [], []
            for e in es:
                rows = (ids == e).any(1).nonzero().squeeze(1)
                n = int(rows.numel())
                G = gu[e - e0, :640].to(dev); U = gu[e - e0, 640:].to(dev); D = dn[e - e0].to(dev)
                if n:
                    Xe = X[rows.to(X.device)].to(dev)
                    Hin = (Xe.float().T @ Xe.float()) / n
                    A = F.silu(Xe @ G.T) * (Xe @ U.T)
                    Hdn = (A.float().T @ A.float()) / n
                else:
                    Hin = torch.eye(2560, device=dev); Hdn = torch.eye(640, device=dev)
                jobs_gu += [dict(W=G.float(), H=Hin), dict(W=U.float(), H=Hin)]
                jobs_dn.append(dict(W=D.float(), H=Hdn))
                meta.append(dict(e=e, n=n, Hin=Hin, Hdn=Hdn, G=G, U=U, D=D))
            r_gu = encode.encode_group(jobs_gu, a.bits, seed=a.seed + 7919 * L + 104729 * e0, shared_su=True, damp=a.damp)
            r_dn = encode.encode_group(jobs_dn, a.bits, seed=a.seed + 7919 * L + 104729 * e0 + 1, shared_su=True, damp=a.damp)
            keep, rows_out, verify_s = [], [], 0.0
            for i, m in enumerate(meta):
                rg, ru, rd = r_gu[2 * i], r_gu[2 * i + 1], r_dn[i]
                g4, u4, d4 = (w.to(dev) for w in R.expert(m["e"]))
                su = (u4 * m["U"].float()).sum(1) / (m["U"].float() ** 2).sum(1)
                sd = (d4 * m["D"].float()).sum(0) / (m["D"].float() ** 2).sum(0)
                row = dict(L=L, e=m["e"], n=m["n"], rel_bits=R.bits[m["e"]], K=a.bits,
                           htr=[rel(m["G"].float(), rg["What"], m["Hin"]), rel(m["U"].float(), ru["What"], m["Hin"]),
                                rel(m["D"].float(), rd["What"], m["Hdn"])],
                           release=[rel(m["G"].float(), g4, m["Hin"]), rel(m["U"].float(), u4 / su[:, None], m["Hin"]),
                                    rel(m["D"].float(), d4 / sd[None, :], m["Hdn"])],
                           energy=[float(((m["G"].float() @ m["Hin"]) * m["G"].float()).sum()),
                                   float(((m["U"].float() @ m["Hin"]) * m["U"].float()).sum()),
                                   float(((m["D"].float() @ m["Hdn"]) * m["D"].float()).sum())])
                rows_out.append(row)
                tv = time.time(); packed, sha = {}, {}
                for role, r in zip(ROLES, (rg, ru, rd)):
                    packed[role], sha[role] = verify_and_strip(r, dev)
                verify_s += time.time() - tv
                keep.append(dict(e=m["e"], n=m["n"], packed=packed, what_bf16_sha256=sha))
            z = dict(format=FORMAT, L=L, K=a.bits, experts=keep, rows=rows_out, seconds=time.time() - tc,
                     verify_seconds=verify_s, decode_check="bitwise BF16 equality of wire.dequant vs What, every matrix",
                     decode_device=gpu, worker=me)
            assert strip_what(z), "reconstruction leaked into a P2 chunk"
            p = os.path.join(od, f"c{e0:03d}.pt")
            torch.save(z, p + ".tmp")
            os.replace(p + ".tmp", p); os.remove(p[:-3] + ".claim")
            with open(os.path.join(a.out, "proxy.jsonl"), "a") as f:
                for r in rows_out:
                    f.write(json.dumps(r) + "\n")
            hm = lambda k: sum(r[k][j] for r in rows_out for j in range(3)) / (3 * len(rows_out))
            log(f"L{L} c{e0:03d} K{a.bits} mean rel htr {hm('htr'):.5f} release {hm('release'):.5f} "
                f"n {[r['n'] for r in rows_out][:6]}.. {time.time()-tc:.0f}s verify {verify_s:.1f}s "
                f"{os.path.getsize(p)/2**20:.1f} MiB [{gpu}]")
            del jobs_gu, jobs_dn, meta, r_gu, r_dn, keep, z, gu, dn
            torch.cuda.empty_cache()
            done_chunks += 1
            if a.max_chunks and done_chunks >= a.max_chunks:
                log("bounded chunk preflight done", done_chunks)
                return
        del X, ids, S
        torch.cuda.empty_cache()
    log("encode done")


if __name__ == "__main__":
    encode.VITERBI = vit_v2.viterbi_auto
    main()
