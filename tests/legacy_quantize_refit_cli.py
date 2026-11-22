"""CPU tests for the opt-in post-encode scale refit in htr.quantize (--scale-refit). No GPU, no HF model load.

The refit runs after encode or after a --cache group load, per matrix, through htr.scale_refit.refit_packed with the
encoder's own damping. So a cached encode can be refit without re-encoding, and group caches keep the unrefit codes.
The default path must assemble exactly what d4ae293 assembles from the same caches (htr_report.json args gain
'scale_refit'). HF model/tokenizer construction and the input-identity capture are test doubles; the trellis search
is a test-only legal-path backend (valid circular paths, not a quality oracle). Everything else is the real main().
"""
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from htr import encode, quantize, wire  # noqa: E402

torch.set_num_threads(2)

PINNED = "d4ae293"          # last commit before --scale-refit
FIELDS = ("qweight", "scales", "su", "sv")
GATE, UP, DOWN = (f"model.layers.0.mlp.{p}" for p in ("gate_proj", "up_proj", "down_proj"))
SHAPES = {GATE: (256, 128), UP: (256, 128), DOWN: (128, 256)}
GROUPS = [[DOWN], [GATE, UP]]   # quantize orders input groups by their first module name
BD0B92C_QC = {"quant_method": "htr", "wire": "HTR-W1", "generator": "rdna24", "L": 16, "tile": 16, "rht_block": 128,
              "modules": {"model.layers.0.mlp.down_proj": {"K": 3}}}


def legal_backend(x, w, cb, k, seam="htr"):
    """Test-only trellis search: a valid circular path per tile that depends on the tile's input."""
    s = (x.double() * 4096).round().to(torch.int64).sum(-1, keepdim=True)
    j = torch.arange(8 * k, dtype=torch.int64, device=x.device)
    u = (s * 1000003 + j * 2654435761 + 12345) % 2 ** 32
    return wire.unpack_states(torch.where(u >= 2 ** 31, u - 2 ** 32, u).to(torch.int32), k)


def make_fixture(d):
    g = torch.Generator().manual_seed(20261122)
    src = d / "src"
    src.mkdir()
    W = {n: (torch.randn(*s, generator=g) * .02).bfloat16() for n, s in SHAPES.items()}
    save_file({**{n + ".weight": w for n, w in W.items()}, "model.norm.weight": torch.ones(128, dtype=torch.bfloat16)},
              str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps({"architectures": ["X"]}))
    H = {}
    for n in (GATE, DOWN):
        X = torch.randn(4 * SHAPES[n][1], SHAPES[n][1], generator=g)
        h = X.T @ X / X.shape[0]
        H[n] = (h + h.T) / 2
    H[UP] = H[GATE]                      # shared input, shared Hessian
    hd = d / "hessians"
    hd.mkdir()
    torch.save(H, hd / "layer000.pt")
    return src, hd, W, H


def model_double(W):
    def build(*args, **kwargs):
        model = torch.nn.Module()
        for name, w in W.items():
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                if p not in parent._modules:
                    parent.add_module(p, torch.nn.Module())
                parent = parent._modules[p]
            mod = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="meta", dtype=torch.bfloat16)
            mod.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
            parent.add_module(parts[-1], mod)
        return model
    return build


def identities(model, tok, mods, calib, ncal, seqlen, seed, dev, want_h=True):
    assert not want_h, "fixture supplies --hessians"
    gid = {DOWN: 0, GATE: 1, UP: 1}
    return {}, {n: (gid[n], gid[n], (1, 64, SHAPES[n][1])) for n in mods}, {}


def pinned_quantize():
    source = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PINNED}:htr/quantize.py"], text=True)
    m = types.ModuleType("htr._quantize_pinned")
    m.__package__ = "htr"
    m.__file__ = f"<{PINNED}:htr/quantize.py>"
    exec(compile(source, m.__file__, "exec"), m.__dict__)
    return m


def load_tree(path):
    out = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


def files_bytes(path):
    return {p.name: p.read_bytes() for p in sorted(path.iterdir()) if p.is_file()}


def cache_bytes(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.glob("g*.pt"))}


def jobs_for(W, H, names):
    return [dict(W=W[n].float(), H=H[n]) for n in names]


class Help(unittest.TestCase):
    def test_help_lists_scale_refit(self):
        r = subprocess.run([sys.executable, "-m", "htr.quantize", "--help"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--scale-refit", r.stdout)


class Config(unittest.TestCase):
    report = {"model.layers.0.mlp.down_proj": dict(K=3, shape=[128, 128], rel_proxy=0.1, group=0)}

    def test_default_config_unchanged(self):
        self.assertEqual(quantize.quant_config(self.report, "htr"), BD0B92C_QC)
        self.assertEqual(quantize.quant_config(self.report, "htr", scale_refit=False), BD0B92C_QC)
        self.assertEqual(quantize.quant_config(self.report, "qtip", scale_refit=False)["encoder"], {"seam": "qtip"})

    def test_refit_recorded_as_encoder_provenance(self):
        for seam, enc in (("htr", {"scale_refit": True}), ("qtip", {"seam": "qtip", "scale_refit": True})):
            qc = quantize.quant_config(self.report, seam, scale_refit=True)
            self.assertEqual(qc["encoder"], enc)
            self.assertEqual({k: v for k, v in qc.items() if k != "encoder"}, BD0B92C_QC)

    def test_refit_damping_is_the_encoder_default(self):
        import inspect
        self.assertEqual(quantize.REFIT_DAMP, inspect.signature(encode.encode_group).parameters["damp"].default)


class RefitGroup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        _, _, cls.W, cls.H = make_fixture(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def encode(self, names, seam, scale_refit):
        with patch.object(encode, "VITERBI", legal_backend):
            return encode.encode_group(jobs_for(self.W, self.H, names), 3, seed=1234 + 104729, shared_su=True,
                                       seam=seam, scale_refit=scale_refit)

    def test_post_encode_refit_equals_encode_group_opt_in(self):
        for seam in ("htr", "qtip"):
            with self.subTest(seam=seam):
                fused = self.encode([GATE, UP], seam, True)
                plain = self.encode([GATE, UP], seam, False)
                post, rels = quantize.refit_group(jobs_for(self.W, self.H, [GATE, UP]), plain)
                self.assertEqual(len(post), 2)
                for f, p, r, pl, n in zip(fused, post, rels, plain, (GATE, UP)):
                    for k in FIELDS + ("What", "Q"):
                        self.assertTrue(torch.equal(f[k], p[k]), (seam, n, k))
                    self.assertEqual(f["scale_refit"], p["scale_refit"])
                    for k in ("qweight", "su", "sv"):
                        self.assertTrue(torch.equal(p[k], pl[k]), (seam, n, k))
                    self.assertGreater(p["scale_refit"]["changed_rows"], 0)
                    self.assertEqual(r, quantize.rel_proxy(self.W[n].float(), p["What"], self.H[n]))

    def test_cached_fields_refit_identically(self):
        plain = self.encode([DOWN], "htr", False)
        keep = [{k: r[k].contiguous().cpu() for k in FIELDS} | {"What": r["What"].bfloat16().cpu()} for r in plain]
        a, ra = quantize.refit_group(jobs_for(self.W, self.H, [DOWN]), plain)
        b, rb = quantize.refit_group(jobs_for(self.W, self.H, [DOWN]), keep)
        self.assertEqual(ra, rb)
        for k in FIELDS + ("What", "Q"):
            self.assertTrue(torch.equal(a[0][k], b[0][k]), k)

    def test_rel_proxy_is_the_earlier_inline_formula(self):
        W, H = self.W[DOWN].float(), self.H[DOWN]
        What = W + 1e-3 * torch.randn(W.shape, generator=torch.Generator().manual_seed(5))
        E_ = (W.double() - What.double())
        Hd = H.double()
        self.assertEqual(quantize.rel_proxy(W, What, H),
                         float(((E_ @ Hd) * E_).sum() / ((W.double() @ Hd) * W.double()).sum()))


class MainCli(unittest.TestCase):
    """Real quantize.main() on CPU (DEVICE patched); doubles only for HF construction and input identities."""

    @classmethod
    def setUpClass(cls):
        assert not torch.cuda.is_initialized()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.src, cls.hd, cls.W, cls.H = make_fixture(cls.d)
        cls.calls = {}
        cls.invoke("worker-default", ["--cache", "c-default", "--worker-only"])
        cls.invoke("worker-refitflag", ["--cache", "c-refitflag", "--worker-only", "--scale-refit"])
        for label in ("asm-pinned", "asm-default", "asm-refit"):
            shutil.copytree(cls.d / "c-default", cls.d / f"c-{label}")
        cls.cache_before = cache_bytes(cls.d / "c-asm-refit")
        cls.invoke("asm-pinned", ["--cache", "c-asm-pinned", "--save-fake", "fake-asm-pinned"], pinned=True, encode=False)
        cls.invoke("asm-default", ["--cache", "c-asm-default", "--save-fake", "fake-asm-default"], encode=False)
        cls.invoke("asm-refit", ["--cache", "c-asm-refit", "--save-fake", "fake-asm-refit", "--scale-refit"], encode=False)
        cls.invoke("fresh-refit", ["--save-fake", "fake-fresh-refit", "--scale-refit"])
        cls.invoke("fresh-qtip-refit", ["--seam", "qtip", "--scale-refit"])
        cls.oracle = {}
        with patch.object(encode, "VITERBI", legal_backend):
            for seam in ("htr", "qtip"):
                for gi, names in enumerate(GROUPS):
                    res = encode.encode_group(jobs_for(cls.W, cls.H, names), 3, seed=1234 + 104729 * gi,
                                              shared_su=True, seam=seam, scale_refit=True)
                    for n, r in zip(names, res):
                        cls.oracle[seam, n] = r

    @classmethod
    def tearDownClass(cls):
        assert not torch.cuda.is_initialized()
        cls.tmp.cleanup()

    @classmethod
    def invoke(cls, label, extra, pinned=False, encode=True, policy=None):
        mod = pinned_quantize() if pinned else quantize
        argv = ["htr.quantize", "--model", str(cls.src), "--out", str(cls.d / f"out-{label}"), "--bits", "3",
                "--calib", "/not-read", "--hessians", str(cls.hd), "--offload", "--viterbi", "ref"]
        if policy is not None:
            (cls.d / f"policy-{label}.json").write_text(json.dumps(policy))
            argv += ["--policy", str(cls.d / f"policy-{label}.json")]
        argv += [str(cls.d / a) if a.startswith(("c-", "fake-")) else a for a in extra]
        from transformers import AutoModelForCausalLM, AutoTokenizer
        real = quantize.encode.encode_group
        counter = {"n": 0}

        def counted(*a, **k):
            counter["n"] += 1
            if not encode:
                raise RuntimeError("cache miss: forbidden encode")
            return real(*a, **k)
        out = io.StringIO()
        with contextlib.ExitStack() as st:
            st.enter_context(patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=model_double(cls.W)))
            st.enter_context(patch.object(AutoTokenizer, "from_pretrained", return_value=object()))
            st.enter_context(patch.object(mod, "capture", side_effect=identities))
            st.enter_context(patch.object(quantize.encode, "encode_group", side_effect=counted))
            st.enter_context(patch.object(quantize.encode, "_viterbi_ref", legal_backend))
            st.enter_context(patch.object(quantize.encode, "VITERBI", quantize.encode.VITERBI))
            if not pinned:
                st.enter_context(patch.object(quantize, "DEVICE", "cpu"))
            st.enter_context(patch.object(sys, "argv", argv))
            try:
                with contextlib.redirect_stdout(out):
                    mod.main()
                rc = 0
            except SystemExit as e:
                rc = e.code
            except RuntimeError as e:
                if encode or "forbidden encode" not in str(e):
                    raise
                rc = "forbidden-encode"
        cls.calls[label] = dict(encode_calls=counter["n"], rc=rc, log=out.getvalue())
        return rc

    def out(self, label):
        return self.d / f"out-{label}"

    def test_default_assembly_equals_pinned_d4ae293(self):
        for a, b in (("out-asm-default", "out-asm-pinned"), ("fake-asm-default", "fake-asm-pinned")):
            new, old = files_bytes(self.d / a), files_bytes(self.d / b)
            self.assertEqual(set(new), set(old))
            for name in new:
                if name != "htr_report.json":
                    self.assertEqual(new[name], old[name], (a, name))
            rn, ro = json.loads(new["htr_report.json"]), json.loads(old["htr_report.json"])
            self.assertIs(rn["args"].pop("scale_refit"), False)
            for r in (rn, ro):
                for k in ("out", "save_fake", "cache"):
                    r["args"].pop(k)
            self.assertEqual(rn, ro)
        self.assertNotIn("encoder", json.loads((self.out("asm-default") / "config.json").read_text())["quantization_config"])

    def test_assembly_runs_never_encode(self):
        for label in ("asm-pinned", "asm-default", "asm-refit"):
            self.assertEqual(self.calls[label]["encode_calls"], 0, label)
            self.assertEqual(self.calls[label]["rc"], 0, label)
            self.assertEqual(self.calls[label]["log"].count(" (cache)"), 2, label)
        self.assertEqual(self.calls["fresh-refit"]["encode_calls"], 2)

    def test_refit_from_cache_equals_fresh_refit(self):
        for a, b in (("out-asm-refit", "out-fresh-refit"), ("fake-asm-refit", "fake-fresh-refit")):
            new, old = files_bytes(self.d / a), files_bytes(self.d / b)
            self.assertEqual(set(new), set(old))
            for name in new:
                if name != "htr_report.json":
                    self.assertEqual(new[name], old[name], (a, name))
            rn, ro = json.loads(new["htr_report.json"]), json.loads(old["htr_report.json"])
            self.assertEqual(rn["modules"], ro["modules"])

    def test_refit_matches_library_opt_in(self):
        for label, seam, fake in (("asm-refit", "htr", "fake-asm-refit"), ("fresh-refit", "htr", "fake-fresh-refit"),
                                  ("fresh-qtip-refit", "qtip", None)):
            packed = load_tree(self.out(label))
            report = json.loads((self.out(label) / "htr_report.json").read_text())["modules"]
            fk = load_tree(self.d / fake) if fake else None
            for n in SHAPES:
                o = self.oracle[seam, n]
                for f in FIELDS:
                    self.assertTrue(torch.equal(packed[n + ".htr_" + f], o[f]), (label, n, f))
                self.assertEqual(report[n]["rel_proxy"], quantize.rel_proxy(self.W[n].float(), o["What"], self.H[n]))
                self.assertEqual(report[n]["scale_refit"], o["scale_refit"])
                if fk is not None:
                    self.assertTrue(torch.equal(fk[n + ".weight"], o["What"].bfloat16()), (label, n))

    def test_refit_changes_scales_only(self):
        refit, default = load_tree(self.out("asm-refit")), load_tree(self.out("asm-default"))
        self.assertEqual(set(refit), set(default))
        report = json.loads((self.out("asm-refit") / "htr_report.json").read_text())["modules"]
        moved = 0
        for k in refit:
            if k.endswith(".htr_scales"):
                n = k[:-len(".htr_scales")]
                changed = int((refit[k] != default[k]).sum())
                self.assertEqual(changed, report[n]["scale_refit"]["changed_rows"], n)
                self.assertLessEqual(report[n]["scale_refit"]["after"], report[n]["scale_refit"]["before"])
                moved += changed
            else:
                self.assertTrue(torch.equal(refit[k], default[k]), k)
        self.assertGreater(moved, 0)

    def test_refit_provenance(self):
        for label, enc in (("asm-refit", {"scale_refit": True}), ("fresh-qtip-refit", {"seam": "qtip", "scale_refit": True})):
            o = self.out(label)
            self.assertEqual(json.loads((o / "config.json").read_text())["quantization_config"]["encoder"], enc)
            self.assertIs(json.loads((o / "htr_report.json").read_text())["args"]["scale_refit"], True)
        fake_cfg = json.loads((self.d / "fake-asm-refit" / "config.json").read_text())
        self.assertNotIn("quantization_config", fake_cfg)

    def test_group_cache_keeps_unrefit_codes(self):
        self.assertEqual(cache_bytes(self.d / "c-asm-refit"), self.cache_before)
        for gi, names in enumerate(GROUPS):
            a = torch.load(self.d / "c-default" / f"g{gi:04d}.pt", map_location="cpu", weights_only=False)
            b = torch.load(self.d / "c-refitflag" / f"g{gi:04d}.pt", map_location="cpu", weights_only=False)
            self.assertEqual(set(a), set(b))
            self.assertNotIn("scale_refit", b)
            self.assertEqual((a["names"], a["K"], a["seam"], a["rel"]), (b["names"], b["K"], b["seam"], b["rel"]))
            for ra, rb in zip(a["res"], b["res"]):
                self.assertEqual(set(ra), set(rb))
                for k in ra:
                    self.assertTrue(torch.equal(ra[k], rb[k]), (gi, k))
        self.assertFalse(self.out("worker-refitflag").exists())

    def test_unsupported_width_rejected_before_encode(self):
        label = "reject-k6"
        rc = self.invoke(label, ["--scale-refit"], policy={"overrides": {"down_proj": 6}})
        self.assertEqual(rc, 2)
        self.assertEqual(self.calls[label]["encode_calls"], 0)
        self.assertFalse(self.out(label).exists())
        # Without --scale-refit the same policy is not refused by this check: it reaches the encoder.
        rc = self.invoke("k6-no-refit", ["--worker-only", "--cache", "c-k6"], policy={"overrides": {"down_proj": 6}},
                      encode=False)
        self.assertEqual(rc, "forbidden-encode")
        self.assertEqual(self.calls["k6-no-refit"]["encode_calls"], 1)


if __name__ == "__main__":
    unittest.main()
