"""vLLM quantization plugin for HTR-W1 checkpoints (quant_method "htr"). TP=1 only.
Import this module (e.g. via sitecustomize or the vllm.general_plugins entry point) before the engine starts."""
import json, os, time
import torch
from torch.nn import Parameter
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.utils import set_weight_attrs

SKIP = ("visual", "vision", "audio", "mtp")


def canon(name):
    if name.endswith(".weight"):
        name = name[:-7]
    i = name.rfind("layers.")
    return name[i:] if i >= 0 else name


@register_quantization_config("htr")
class HTRConfig(QuantizationConfig):
    def __init__(self, modules, wire="HTR-W1"):
        super().__init__()
        assert wire == "HTR-W1", wire
        self.modules = {canon(k): int(v["K"]) for k, v in modules.items()}

    def get_name(self):
        return "htr"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls(config.get("modules", {}), config.get("wire", "HTR-W1"))

    def get_quant_method(self, layer, prefix):
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        if isinstance(layer, ParallelLMHead):   # v2 (2026-10-16): HTR lm_head; any other embedding stays unquantized
            K = self.modules.get("lm_head")
            return HTRLinearMethod(K) if (K is not None and prefix.split(".")[-1] == "lm_head") else None
        if not isinstance(layer, LinearBase):
            return None
        if any(s in prefix.split(".") or s in prefix for s in SKIP):
            return UnquantizedLinearMethod()
        c = canon(prefix)
        parent, _, leaf = c.rpartition(".")
        pm = getattr(self, "packed_modules_mapping", None) or {}
        comps = pm.get(leaf, [leaf])
        ks = [self.modules.get(f"{parent}.{x}" if parent else x) for x in comps]
        if all(k is None for k in ks):
            return UnquantizedLinearMethod()
        if any(k is None for k in ks) or len(set(ks)) != 1:
            raise ValueError(f"HTR: fused module {prefix} has mixed quantization {dict(zip(comps, ks))}")
        return HTRLinearMethod(ks[0])


def _shard_range(sizes, shard_id, unit):
    if shard_id is None:
        idx = list(range(len(sizes)))
    elif isinstance(shard_id, str):
        idx = [{"q": 0, "k": 1, "v": 2}[shard_id]]
    elif isinstance(shard_id, (tuple, list)):
        idx = list(shard_id)
    else:
        idx = [int(shard_id)]
    assert idx == list(range(idx[0], idx[-1] + 1)), idx
    start = sum(sizes[:idx[0]])
    length = sum(sizes[i] for i in idx)
    assert start % unit == 0 and length % unit == 0
    return start // unit, length // unit


class HTRLinearMethod(LinearMethodBase):
    def __init__(self, K):
        self.K = K

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        n, sizes = input_size_per_partition, list(output_partition_sizes)
        m = sum(sizes)
        assert input_size_per_partition == input_size, "HTR: tensor parallelism is not supported"
        assert n % 128 == 0 and all(s % 128 == 0 for s in sizes), (n, sizes)
        K = self.K
        qw = Parameter(torch.empty(m // 16, n // 16, 8 * K, dtype=torch.int32), requires_grad=False)
        sc = Parameter(torch.empty(m, dtype=torch.float16), requires_grad=False)
        su = Parameter(torch.empty(n // 8, dtype=torch.uint8), requires_grad=False)
        sv = Parameter(torch.empty(m // 8, dtype=torch.uint8), requires_grad=False)

        def rows_loader(unit):
            def load(param, loaded, shard_id=None):
                if shard_id is None:   # v3: vLLM WeightsMapper(orig_to_new_stacked) passes the shard as a tensor attribute
                    shard_id = getattr(loaded, "shard_id", None)
                s, ln = _shard_range(sizes, shard_id, unit)
                assert tuple(loaded.shape) == (ln,) + tuple(param.shape[1:]), (param.shape, loaded.shape, shard_id)
                param.data[s:s + ln].copy_(loaded)
            return load

        def su_loader(param, loaded, shard_id=None):
            assert loaded.shape == param.shape
            if getattr(param, "_htr_loaded", False):
                if not torch.equal(param.data.cpu(), loaded.cpu()):
                    raise ValueError("HTR: fused shards do not share the input sign vector")
            else:
                param.data.copy_(loaded)
                param._htr_loaded = True

        layer.register_parameter("htr_qweight", qw)
        layer.register_parameter("htr_scales", sc)
        layer.register_parameter("htr_su", su)
        layer.register_parameter("htr_sv", sv)
        set_weight_attrs(qw, {"weight_loader": rows_loader(16)})
        set_weight_attrs(sc, {"weight_loader": rows_loader(1)})
        set_weight_attrs(sv, {"weight_loader": rows_loader(8)})
        set_weight_attrs(su, {"weight_loader": su_loader})
        layer.htr_K = K

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None):
        from htr import ops
        if _COUNT_FILE is not None:   # eager-only diagnosis (launcher --eager): per-M-bucket dispatch census, rewritten every 512 calls
            M = x.numel() // x.shape[-1]
            _COUNTS["M1" if M == 1 else ("M2-8" if M <= 8 else "M>8")] += 1
            _COUNTS["K%d" % layer.htr_K] += 1
            if sum(_COUNTS[k] for k in ("M1", "M2-8", "M>8")) % 512 == 0:
                _dump_counts()
        return ops.linear(x, layer.htr_qweight, layer.htr_scales, layer.htr_su, layer.htr_sv, bias)


import collections as _collections
_COUNT_FILE = os.environ.get("HTR_COUNT_FILE") or None
_COUNTS = _collections.Counter()


def _dump_counts():
    tmp = f"{_COUNT_FILE}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(dict(pid=os.getpid(), t=time.time(), counts=dict(_COUNTS), selected=len(HTR_OVERLAY_SELECTED)), f)
    os.replace(tmp, f"{_COUNT_FILE}.{os.getpid()}.json")


def _htr_overlay_method(cfg, layer, prefix):
    """HTR method for a dense layer listed in a Quark config's quantization_config["htr_overlay"]; None = not an HTR target."""
    ov = cfg.quant_config.get("htr_overlay") if isinstance(getattr(cfg, "quant_config", None), dict) else None
    if not ov:
        return None
    if not isinstance(layer, LinearBase):
        return None
    if prefix.startswith("mtp") or any(s in prefix.split(".") for s in SKIP):
        return None
    vp = ov.get("vllm_prefix")   # e.g. "language_model.model.layers." : only the main trunk (a draft's "model.layers.N" never matches)
    if vp is not None and not prefix.startswith(vp):
        return None
    mods = cfg.__dict__.get("_htr_modules")
    if mods is None:
        assert ov.get("wire") == "HTR-W1", ov.get("wire")
        mods = {canon(k): int(v["K"]) for k, v in ov["modules"].items()}
        cfg.__dict__["_htr_modules"] = mods
    c = canon(prefix)
    parent, _, leaf = c.rpartition(".")
    pm = getattr(cfg, "packed_modules_mapping", None) or {}
    comps = pm.get(leaf, [leaf])
    ks = [mods.get(f"{parent}.{x}" if parent else x) for x in comps]
    if all(k is None for k in ks):
        return None
    if any(k is None for k in ks) or len(set(ks)) != 1:
        raise ValueError(f"HTR overlay: fused module {prefix} has mixed quantization {dict(zip(comps, ks))}")
    HTR_OVERLAY_SELECTED.append((prefix, ks[0]))
    return HTRLinearMethod(ks[0])


HTR_OVERLAY_SELECTED = []


def install_quark_overlay():
    """v3 (2026-10-16, Qwen3.8 HaloLoom): mixed Quark + HTR checkpoints keep quant_method "quark" (the runtime's QuarkMixedConfig
    requires type(base) is QuarkConfig, so no subclass/registry change). QuarkConfig.get_quant_method is wrapped at class level
    (same mechanism as the image's direct_w8a16): only dense LinearBase layers named in quantization_config["htr_overlay"] get
    HTRLinearMethod; everything else (experts, router, HC, PLE, MTP, unlisted dense) goes to the original Quark selector.
    Configs without "htr_overlay" are unaffected."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    if getattr(QuarkConfig.get_quant_method, "_htr_overlay", False):
        return
    orig = QuarkConfig.get_quant_method

    def get_quant_method(self, layer, prefix):
        m = _htr_overlay_method(self, layer, prefix)
        return m if m is not None else orig(self, layer, prefix)
    get_quant_method._htr_overlay = True
    get_quant_method._htr_orig = orig
    QuarkConfig.get_quant_method = get_quant_method


install_quark_overlay()


def _current_htr_overlay():
    """htr_overlay dict of the model being built (None outside model construction or for configs without one). The MTP draft is
    built under the draft's own model config (draft/config.json has no htr_overlay), so only the target can match."""
    try:
        from vllm.config import get_current_vllm_config_or_none
        vc = get_current_vllm_config_or_none()
        qc = getattr(vc.model_config.hf_config, "quantization_config", None) if vc is not None else None
    except Exception:
        return None
    return qc.get("htr_overlay") if isinstance(qc, dict) else None


HTR_LMHEAD_SELECTED = []


def install_lmhead_overlay():
    """v4 (2026-10-17, E10 P0): HTR lm_head for Qwen3.8 overlays. The model builds its lm_head as
    ParallelLMHead(quant_config=direct_w8a16.config_or_none()) -> None in the bf16 variant, so the Quark selector never sees it.
    ParallelLMHead.__init__ is wrapped at class level: when the model under construction has htr_overlay["lm_head"] = {"K": k} and
    the head is the target's (leaf "lm_head", no "mtp" in the prefix), it is built with an HTRConfig whose get_quant_method returns
    HTRLinearMethod(k) for exactly that layer. The MTP proposer then replaces the draft head by this target head (shared)."""
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    if getattr(ParallelLMHead.__init__, "_htr_lmhead", False):
        return
    orig = ParallelLMHead.__init__

    def __init__(self, *args, **kw):
        prefix = kw.get("prefix", args[7] if len(args) > 7 else "")
        ov = _current_htr_overlay()
        ent = ov.get("lm_head") if ov else None
        if ent is not None and prefix.split(".")[-1] == "lm_head" and "mtp" not in prefix.split(".") and not _DRAFT_BUILD[0]:
            qc = kw.get("quant_config", args[6] if len(args) > 6 else None)
            if qc is not None:
                raise ValueError(f"HTR lm_head: {prefix} already has quant_config {type(qc).__name__} (DIRECT_W8A16 + HTR head?)")
            assert len(args) <= 6, "HTR lm_head: unexpected positional quant_config/prefix"
            kw["quant_config"] = HTRConfig({"lm_head": {"K": int(ent["K"])}})
            HTR_LMHEAD_SELECTED.append((prefix, int(ent["K"])))
            HTR_OVERLAY_SELECTED.append((prefix, int(ent["K"])))
        orig(self, *args, **kw)
    __init__._htr_lmhead = True
    __init__._htr_orig = orig
    ParallelLMHead.__init__ = __init__
    # v5b: Qwen4ExpMTP builds its own head as ParallelLMHead(prefix="lm_head") OUTSIDE its draft-config scope, so the overlay
    # (read from the current = target config) matched it and the BF16 draft/lm_head.weight had nowhere to load. The draft head
    # must stay stock while weights load; load_eagle_model then shares the target (HTR) head into the draft (has_own_lm_head unset).
    try:
        from vllm.models.qwen4_exp.amd import mtp as _mtp
    except Exception:
        return
    MC = _mtp.Qwen4ExpMTP
    if getattr(MC.__init__, "_htr_draft", False):
        return
    morig = MC.__init__

    def minit(self, *args, **kw):
        _DRAFT_BUILD[0] += 1
        try:
            morig(self, *args, **kw)
        finally:
            _DRAFT_BUILD[0] -= 1
    minit._htr_draft = True
    minit._htr_orig = morig
    MC.__init__ = minit


_DRAFT_BUILD = [0]


install_lmhead_overlay()


class HTRHCLinearMethod(HTRLinearMethod):
    """v5 (2026-10-17, E10 P0): HTR for hyper-connection projections whose dims are not % 128 (q38/hc_q38.py pads to 384).
    Stored tensors cover the padded [m_pad, n_pad] matrix; apply zero-pads the input to n_pad and returns the first m_out rows
    (m_out = the module's own output width, e.g. 336 = 320 lora + 4 inject + 12 alignment pad). A zero-storage uint8 `weight`
    buffer of the module's logical shape keeps the image's HC call sites working (they read .weight to pass shape carriers
    into the nb1/hcv3 opaque ops); its dtype fails every fused-path guard, so those ops run their exact nb1 reference sequence,
    which calls the module forwards (= this method)."""

    def __init__(self, spec):
        super().__init__(int(spec["K"]))
        self.spec = spec

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        sp, K = self.spec, self.K
        n, m_out = input_size_per_partition, sum(output_partition_sizes)
        assert input_size_per_partition == input_size, "HTR: tensor parallelism is not supported"
        m_pad, n_pad = int(sp["m_pad"]), int(sp["n_pad"])
        assert n == sp["n"] and sp["m"] <= m_out <= m_pad and n <= n_pad and m_pad % 128 == 0 and n_pad % 128 == 0, (sp, n, m_out)
        shapes = dict(htr_qweight=((m_pad // 16, n_pad // 16, 8 * K), torch.int32), htr_scales=((m_pad,), torch.float16),
                      htr_su=((n_pad // 8,), torch.uint8), htr_sv=((m_pad // 8,), torch.uint8))
        for nm, (shp, dt) in shapes.items():
            p = Parameter(torch.empty(shp, dtype=dt), requires_grad=False)

            def load(param, loaded, shard_id=None, nm=nm):
                assert shard_id is None and getattr(loaded, "shard_id", None) is None, (nm, shard_id)
                assert tuple(loaded.shape) == tuple(param.shape) and loaded.dtype == param.dtype, (nm, param.shape, loaded.shape)
                param.data.copy_(loaded)
            layer.register_parameter(nm, p)
            set_weight_attrs(p, {"weight_loader": load})
        layer.htr_K = K
        layer._htr_n, layer._htr_n_pad, layer._htr_m_out, layer._htr_m_pad = n, n_pad, m_out, m_pad
        layer.register_buffer("weight", torch.zeros(1, dtype=torch.uint8).expand(m_out, n), persistent=False)
        layer._htr_weight_carrier = True

    def apply(self, layer, x, bias=None):
        from htr import ops
        import torch.nn.functional as F
        if _COUNT_FILE is not None:
            M = x.numel() // x.shape[-1]
            _COUNTS["M1" if M == 1 else ("M2-8" if M <= 8 else "M>8")] += 1
            _COUNTS["K%d" % layer.htr_K] += 1; _COUNTS["hc"] += 1
        if layer._htr_n_pad != layer._htr_n:
            x = F.pad(x, (0, layer._htr_n_pad - layer._htr_n))
        y = ops.linear(x, layer.htr_qweight, layer.htr_scales, layer.htr_su, layer.htr_sv, bias)
        if layer._htr_m_out != layer._htr_m_pad:
            y = y[..., :layer._htr_m_out].contiguous()
        return y


class HTRHCConfig(QuantizationConfig):
    """Explicit config handed to GatedResidual's HC linears (instead of direct_w8a16.config_or_none(), which is None in the
    bf16 variant): listed HC modules -> HTRHCLinearMethod, everything else -> base config or UnquantizedLinearMethod."""

    def __init__(self, mods, base=None):
        super().__init__()
        self.mods, self.base = mods, base

    def get_name(self):
        return "htr_hc"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    def get_quant_method(self, layer, prefix):
        sp = self.mods.get(prefix)
        if sp is not None and isinstance(layer, LinearBase):
            HTR_OVERLAY_SELECTED.append((prefix, int(sp["K"])))
            HTR_HC_SELECTED.append((prefix, int(sp["K"])))
            return HTRHCLinearMethod(sp)
        if self.base is not None:
            return self.base.get_quant_method(layer, prefix)
        return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else None


HTR_HC_SELECTED = []


def install_hc_overlay():
    """Wrap GatedResidual.__init__: while a target HC block listed in htr_overlay["hc"]["modules"] is being built, the module-level
    direct_w8a16.config_or_none (called by GatedResidual for its linears) returns an HTRHCConfig. MTP-draft HC (prefix mtp.*,
    draft config without htr_overlay) is untouched. Refuses to combine with DIRECT_W8A16 (base config not None)."""
    try:
        from vllm.models.qwen4_exp.amd import hyperconnection as hcm
    except Exception:
        return
    GR = hcm.GatedResidual
    if getattr(GR.__init__, "_htr_hc", False):
        return
    orig = GR.__init__

    def __init__(self, *args, **kw):
        prefix = kw.get("prefix", args[2] if len(args) > 2 else "")
        ov = _current_htr_overlay()
        hc = (ov or {}).get("hc")
        mods = hc["modules"] if hc else {}
        if mods and "mtp" not in prefix.split(".") and any(k.startswith(prefix + ".") for k in mods):
            w8 = hcm._w8
            oc = w8.config_or_none
            base = oc()
            if base is not None:
                raise ValueError("HTR HC overlay cannot be combined with DIRECT_W8A16")
            w8.config_or_none = lambda: HTRHCConfig(mods, None)
            try:
                orig(self, *args, **kw)
            finally:
                w8.config_or_none = oc
        else:
            orig(self, *args, **kw)
    __init__._htr_hc = True
    __init__._htr_orig = orig
    GR.__init__ = __init__


install_hc_overlay()


def register():
    """vllm.general_plugins entry point (vllm_site/htr_toolbox-0.1.0.dist-info). Importing this module already registered
    quant_method "htr"; vLLM calls this only in its own engine/worker processes, never in unrelated Python children.
    NEVER import this module from sitecustomize: E10 2026-10-15 16:12 fork-bombed (vllm -> xgrammar -> tvm_ffi JIT-builds its
    torch-dlpack addon in a python child, which re-ran sitecustomize -> vllm -> tvm_ffi -> build ... unbounded)."""
    return None
