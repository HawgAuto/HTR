"""Unit tests for opt-in --seam {htr,qtip} provenance and compatibility in htr.encode_q38."""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from htr import encode_q38 as q38


class TestQ38Seam(unittest.TestCase):
    def _args(self, cache, **kw):
        base = dict(cache=cache, bits=4, seed=1234, hdir="/h", hmix=None, model="/m",
                    viterbi="auto", scale_refit=False, mixed=False, seam="htr")
        base.update(kw)
        return SimpleNamespace(**base)

    def test_default_htr_omits_seam_and_accepts_legacy_plan(self):
        groups = [dict(layer=0, group="g0", names=["a"])]
        with tempfile.TemporaryDirectory() as td:
            a = self._args(td)
            q38.check_plan(a, groups, publish=True)
            plan = json.load(open(os.path.join(td, "plan.json")))
            self.assertNotIn("seam", plan)
            with tempfile.TemporaryDirectory() as td2:
                self.assertEqual(q38.check_plan(self._args(td2), groups, publish=False), {})

    def test_qtip_records_seam_and_rejects_cross_seam_reuse(self):
        groups = [dict(layer=0, group="g0", names=["a"])]
        with tempfile.TemporaryDirectory() as td:
            q38.check_plan(self._args(td, seam="qtip"), groups, publish=True)
            plan = json.load(open(os.path.join(td, "plan.json")))
            self.assertEqual(plan["seam"], "qtip")
            with self.assertRaisesRegex(ValueError, "seam plan mismatch"):
                q38.check_plan(self._args(td, seam="htr"), groups, publish=False)
        with tempfile.TemporaryDirectory() as td:
            q38.check_plan(self._args(td, seam="htr"), groups, publish=True)
            with self.assertRaisesRegex(ValueError, "seam plan mismatch"):
                q38.check_plan(self._args(td, seam="qtip"), groups, publish=False)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "plan provenance"):
                q38.check_plan(self._args(td, seam="qtip"), groups, publish=False)

    def test_check_group_validates_seam_and_refit_together(self):
        g = dict(layer=0, group="g0", names=["a"])
        base = dict(names=["a"], K=4, seed=1234, res=[{}], rel=[0.01])
        q38.check_group(dict(base, encoder={"seam": "qtip"}), g, self._args("/c", seam="qtip"), 0)
        with self.assertRaisesRegex(ValueError, "seam group recipe mismatch"):
            q38.check_group(base, g, self._args("/c", seam="qtip"), 0)
        with self.assertRaisesRegex(ValueError, "seam group recipe mismatch"):
            q38.check_group(dict(base, encoder={"seam": "qtip"}), g, self._args("/c", seam="htr"), 0)
        combo = dict(base, encoder=dict(q38.REFIT_RECIPE, seam="qtip"), scale_refit=[{"changed_rows": 1}])
        q38.check_group(combo, g, self._args("/c", seam="qtip", scale_refit=True), 0)


if __name__ == "__main__":
    unittest.main()
