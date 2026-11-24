"""Standalone unit tests for HTR-W1/rdna24 wire/generator identity guards in htr.vllm_plugin."""
import importlib.util
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace as NS


def _load_plugin():
    saved = {}
    def stub(name):
        if name not in saved:
            saved[name] = sys.modules.get(name)
        m = sys.modules.get(name)
        if m is None or not hasattr(m, "_htr_stub"):
            m = types.ModuleType(name)
            m.__path__ = []
            m._htr_stub = True
            sys.modules[name] = m
            if "." in name:
                parent, attr = name.rsplit(".", 1)
                setattr(stub(parent), attr, m)
        return m

    lin = stub("vllm.model_executor.layers.linear")
    lin.LinearMethodBase = object
    class Linear: pass
    lin.LinearBase = Linear
    class Legacy: pass
    lin.UnquantizedLinearMethod = Legacy
    stub("vllm.model_executor.layers.quantization").register_quantization_config = lambda name: (lambda c: c)
    stub("vllm.model_executor.layers.quantization.base_config").QuantizationConfig = object
    stub("vllm.model_executor.utils").set_weight_attrs = None
    class Quark:
        def __init__(self, ov=None):
            self.quant_config = {} if ov is None else {"htr_overlay": ov}
        def get_quant_method(self, *a):
            return "base-quark"
    stub("vllm.model_executor.layers.quantization.quark.quark").QuarkConfig = Quark
    class Head:
        def __init__(self, **kw):
            self.qc = kw.get("quant_config")
    stub("vllm.model_executor.layers.vocab_parallel_embedding").ParallelLMHead = Head
    hc = stub("vllm.models.qwen4_exp.amd.hyperconnection")
    hc._w8 = NS(config_or_none=lambda: None)
    class Block:
        def __init__(self, **kw):
            self.qc = hc._w8.config_or_none()
    hc.GatedResidual = Block
    state = {"ov": None}
    ctx = stub("vllm.config")
    ctx.get_current_vllm_config_or_none = lambda: NS(
        model_config=NS(hf_config=NS(quantization_config={"htr_overlay": state["ov"]}))
    )
    plugin_path = pathlib.Path(__file__).resolve().parents[1] / "htr" / "vllm_plugin.py"
    spec = importlib.util.spec_from_file_location("htr._test_vllm_plugin", plugin_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, saved, Linear, Quark, Head, Block, state


def _restore(saved):
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


class TestReaderIdentityGuard(unittest.TestCase):
    def setUp(self):
        self.hp, self.saved, self.Linear, self.Quark, self.Head, self.Block, self.state = _load_plugin()

    def tearDown(self):
        _restore(self.saved)

    def test_valid_direct_and_legacy_unmarked(self):
        cfg = {"wire": "HTR-W1", "generator": "rdna24",
               "modules": {"model.layers.0.mlp.gate_proj": {"K": 3}, "lm_head": {"K": 4}}}
        c = self.hp.HTRConfig.from_config(cfg)
        self.assertIs(type(c.get_quant_method(self.Linear(), "model.layers.0.mlp.gate_proj")), self.hp.HTRLinearMethod)
        self.assertEqual(c.get_quant_method(self.Head(), "lm_head").K, 4)
        # Legacy unmarked direct config keeps default HTR-W1/rdna24
        legacy = {"modules": {"model.layers.0.mlp.gate_proj": {"K": 3}}}
        c2 = self.hp.HTRConfig.from_config(legacy)
        self.assertEqual(c2.get_quant_method(self.Linear(), "model.layers.0.mlp.gate_proj").K, 3)

    def test_rejects_unknown_direct_wire_or_generator(self):
        base = {"wire": "HTR-W1", "generator": "rdna24", "modules": {"model.layers.0.mlp.gate_proj": {"K": 3}}}
        for k, v in (("generator", "c185697"), ("generator", None), ("wire", "HTR-W2"), ("wire", None)):
            with self.subTest(k=k, v=v), self.assertRaises(ValueError):
                self.hp.HTRConfig.from_config(dict(base, **{k: v}))
        bad_member = {"wire": "HTR-W1", "modules": {"model.layers.0.mlp.gate_proj": {"K": 3, "generator": "c32298"}}}
        with self.assertRaises(ValueError):
            self.hp.HTRConfig.from_config(bad_member)

    def test_quark_overlay_fused_and_rejections(self):
        ov = {"wire": "HTR-W1", "vllm_prefix": "language_model.model.layers.",
              "modules": {
                  "model.language_model.layers.2.mlp.shared_expert.gate_proj": {"K": 5},
                  "model.language_model.layers.2.mlp.shared_expert.up_proj": {"K": 5},
              },
              "lm_head": {"K": 4},
              "hc": {"modules": {"language_model.model.layers.0.hc.proj": {"K": 4}}}}
        q = self.Quark(ov)
        q.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
        m = q.get_quant_method(self.Linear(), "language_model.model.layers.2.mlp.shared_expert.gate_up_proj")
        self.assertIs(type(m), self.hp.HTRLinearMethod)
        self.assertEqual(m.K, 5)
        self.state["ov"] = ov
        self.assertEqual(self.Head(prefix="lm_head").qc.get_quant_method(self.Head(), "lm_head").K, 4)
        self.assertIs(type(self.Block(prefix="language_model.model.layers.0.hc").qc.get_quant_method(
            self.Linear(), "language_model.model.layers.0.hc.proj")), self.hp.HTRHCLinearMethod)
        # Missing wire on Quark overlay rejected
        bad_wire = dict(ov); bad_wire.pop("wire")
        with self.assertRaises(ValueError):
            self.Quark(bad_wire).get_quant_method(self.Linear(), "language_model.model.layers.2.mlp.shared_expert.gate_proj")
        # Non-rdna24 generator on overlay, member, lm_head, and hc rejected
        for mod_ov in (
            dict(ov, generator="c185697"),
            dict(ov, modules={"model.language_model.layers.2.mlp.shared_expert.gate_proj": {"K": 5, "generator": "c32298"}}),
        ):
            with self.assertRaises(ValueError):
                self.Quark(mod_ov).get_quant_method(self.Linear(), "language_model.model.layers.2.mlp.shared_expert.gate_proj")
        self.state["ov"] = dict(ov, lm_head={"K": 4, "generator": "c7"})
        with self.assertRaises(ValueError):
            self.Head(prefix="lm_head")
        self.state["ov"] = dict(ov, hc={"modules": {"language_model.model.layers.0.hc.proj": {"K": 4, "generator": "c7"}}})
        with self.assertRaises(ValueError):
            self.Block(prefix="language_model.model.layers.0.hc")
        # MTP passthrough remains untouched
        self.state["ov"] = dict(ov, generator="c185697")
        self.assertIsNone(self.Head(prefix="mtp.lm_head").qc)
        self.assertEqual(self.Quark(self.state["ov"]).get_quant_method(self.Linear(), "mtp.layers.0.foo"), "base-quark")


if __name__ == "__main__":
    unittest.main()
