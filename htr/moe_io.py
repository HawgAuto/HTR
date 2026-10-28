"""E10 P1 routed-expert I/O: BF16 slice source (fused 3D tensors) and release Quark mixed INT4/INT8 expert dequant.
Wire (quark_mixed_moe.weight_loader): INT4 [K, N/8] int32, nibble j of column c -> output row 8c+j, signed two's complement;
INT8 [K, N] int8 (file2file KN wire); scales FP32 [N, K/128]. gate = w1 (rows 0:640 of gate_up), up = w3, down = w2."""
import json, os, struct
import torch
from safetensors import safe_open

LAYERS = (3, 13, 30, 45)


def _index(d):
    f = [x for x in os.listdir(d) if x.endswith("index.json")][0]
    return json.load(open(os.path.join(d, f)))["weight_map"]


def bf16_experts(L, d):
    """-> gate_up [512,1280,2560], down [512,2560,640] BF16 (CPU)."""
    wm = _index(d); p = f"model.language_model.layers.{L}.mlp.experts."
    out = {}
    for n in ("gate_up_proj", "down_proj"):
        with safe_open(os.path.join(d, wm[p + n]), "pt") as fh:
            out[n] = fh.get_tensor(p + n)
    return out["gate_up_proj"], out["down_proj"]


def recipe_bits(L, d):
    r = json.load(open(os.path.join(d, "artifact-metadata/RECIPE.json")))
    b = [x for x in r["trunk"]["blocks"] if x["prefix"].endswith(f"layers.{L}.mlp.experts")][0]
    bits = {e: 4 for e in b["INT4_AWQ"] + b["INT4_RTN_unknown"]}
    bits.update({e: 8 for e in b["INT8_AWQ"]})
    assert sorted(bits) == list(range(512))
    return bits


def deq(w, s, bits):
    """wire weight + scale -> W [N, K] fp32."""
    if bits == 4:
        k = w.shape[0]; n = w.shape[1] * 8
        q = torch.stack([(w >> (4 * j)) & 15 for j in range(8)], -1).reshape(k, n)
        q = torch.where(q >= 8, q - 16, q)
    else:
        q = w.to(torch.int32); k, n = q.shape
    Wt = q.T.float()                                   # [N, K]
    return (Wt.view(n, k // 128, 128) * s.float().view(n, k // 128, 1)).view(n, k)


class Release:
    def __init__(self, L, d):
        self.L, self.d, self.wm, self.bits = L, d, _index(d), recipe_bits(L, d)
        self._fh = {}

    def _get(self, k):
        f = self.wm[k]
        if f not in self._fh:
            self._fh[f] = safe_open(os.path.join(self.d, f), "pt")
        return self._fh[f].get_tensor(k)

    def expert(self, e):
        """-> gate [640,2560], up [640,2560], down [2560,640] fp32 dequant."""
        p = f"model.language_model.layers.{self.L}.mlp.experts.{e}."
        return tuple(deq(self._get(p + f"{n}.weight"), self._get(p + f"{n}.weight_scale"), self.bits[e])
                     for n in ("gate_proj", "up_proj", "down_proj"))
