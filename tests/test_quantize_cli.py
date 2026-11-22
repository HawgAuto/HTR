"""CPU tests for htr.quantize CLI plumbing: --seam reaches encode_group, the group cache records its seam, and the
default writes exactly the earlier quantization_config (no GPU, no model load)."""
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from htr import quantize  # noqa: E402

BD0B92C_QC = {"quant_method": "htr", "wire": "HTR-W1", "generator": "rdna24", "L": 16, "tile": 16, "rht_block": 128,
              "modules": {"model.layers.0.mlp.down_proj": {"K": 3}}}


class Cli(unittest.TestCase):
    def test_help_lists_seam(self):
        r = subprocess.run([sys.executable, "-m", "htr.quantize", "--help"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--seam {htr,qtip}", r.stdout)


class QuantConfig(unittest.TestCase):
    report = {"model.layers.0.mlp.down_proj": dict(K=3, shape=[128, 128], rel_proxy=0.1, group=0)}

    def test_default_is_the_earlier_config(self):
        self.assertEqual(quantize.quant_config(self.report, "htr"), BD0B92C_QC)

    def test_opt_in_seam_is_recorded(self):
        qc = quantize.quant_config(self.report, "qtip")
        self.assertEqual(qc["encoder"], {"seam": "qtip"})
        self.assertEqual({k: v for k, v in qc.items() if k != "encoder"}, BD0B92C_QC)

    def test_write_default_config_unchanged(self):
        from safetensors.torch import save_file
        d = Path(tempfile.mkdtemp())
        src, out = d / "src", d / "out"
        src.mkdir()
        save_file({"model.layers.0.mlp.down_proj.weight": torch.zeros(128, 128, dtype=torch.bfloat16),
                   "model.norm.weight": torch.ones(128, dtype=torch.bfloat16)}, str(src / "model.safetensors"))
        (src / "config.json").write_text(json.dumps({"architectures": ["X"]}))
        new_t = {f"model.layers.0.mlp.down_proj.htr_{f}": torch.zeros(2, dtype=torch.int16) for f in ("qweight", "scales", "su", "sv")}
        for seam, enc in (("htr", None), ("qtip", {"seam": "qtip"})):
            args = types.SimpleNamespace(seam=seam)
            o = out / seam
            quantize._write(str(src), str(o), [str(src / "model.safetensors")], {"model.layers.0.mlp.down_proj.weight"},
                            dict(new_t), self.report, [], {}, args, quantized=True)
            qc = json.loads((o / "config.json").read_text())["quantization_config"]
            self.assertEqual(qc.get("encoder"), enc)
            self.assertEqual(json.loads((o / "htr_report.json").read_text())["args"]["seam"], seam)


class GroupCache(unittest.TestCase):
    def test_seam_is_part_of_cache_identity(self):
        names = ["a", "b"]
        self.assertTrue(quantize.cache_matches(dict(names=names, K=3), names, 3, "htr"))       # pre-seam cache = htr
        self.assertTrue(quantize.cache_matches(dict(names=names, K=3, seam="qtip"), names, 3, "qtip"))
        for bad in (dict(names=names, K=3), dict(names=names, K=3, seam="htr"), dict(names=["a"], K=3, seam="qtip"),
                    dict(names=names, K=4, seam="qtip")):
            self.assertFalse(quantize.cache_matches(bad, names, 3, "qtip"))


if __name__ == "__main__":
    unittest.main()
