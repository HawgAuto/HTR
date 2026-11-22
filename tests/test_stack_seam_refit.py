"""Integration of the selected default seam with opt-in refit. CPU-only, synthetic weights.
Reuse the predecessor worker/assembler fixture, not its historical default choice.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import legacy_quantize_refit_cli as fixture

class StackedCli(unittest.TestCase):
    @classmethod
    def invoke(cls, label, args, **kwargs):
        return fixture.MainCli.invoke.__func__(cls, label, args, **kwargs)

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.d = Path(tmp.name)
        cls.src, cls.hd, cls.W, cls.H = fixture.make_fixture(cls.d)
        cls.calls = {}
        cls.routes = []
        original = fixture.legal_backend
        def observed(x, w, cb, k, seam='htr'):
            cls.routes.append(seam)
            return original(x, w, cb, k, seam=seam)
        with patch.object(fixture, 'legal_backend', observed):
            cls.invoke('worker', ['--worker-only','--cache','c-new'])
        cls.cache_before = fixture.cache_bytes(cls.d/'c-new')
        cls.invoke('default', ['--cache','c-new'], encode=False)
        cls.invoke('refit', ['--cache','c-new','--scale-refit'], encode=False)
        cls.invoke('explicit', ['--seam','qtip','--scale-refit'])

    def test_unflagged_worker_dispatches_qtip(self):
        self.assertGreater(len(self.routes),0)
        self.assertEqual(set(self.routes), {'qtip'})
        for i in range(len(fixture.GROUPS)):
            entry = torch.load(self.d/'c-new'/f'g{i:04d}.pt',weights_only=False,map_location='cpu')
            self.assertEqual(entry['seam'],'qtip')
            self.assertNotIn('scale_refit',entry)

    def test_refit_is_post_cache_and_changes_only_scales(self):
        self.assertEqual(fixture.cache_bytes(self.d/'c-new'), self.cache_before)
        for name in ('default','refit'):
            self.assertEqual((self.calls[name]['rc'], self.calls[name]['encode_calls']), (0,0))
        def packed(name):
            return {k:(str(v.dtype),tuple(v.shape),v.contiguous().view(torch.uint8).numpy().tobytes())
                    for k,v in fixture.load_tree(self.d/('out-'+name)).items()}
        base,fitted,explicit = [packed(x) for x in ('default','refit','explicit')]
        self.assertEqual(set(base),set(fitted)); self.assertEqual(fitted,explicit)
        changed = {k for k in base if base[k]!=fitted[k]}
        self.assertEqual(changed,{n+'.htr_scales' for n in fixture.SHAPES})
        self.assertFalse(torch.cuda.is_initialized())

    def test_checkpoint_records_default_and_opt_in_separately(self):
        for label,on in [('default',False),('refit',True),('explicit',True)]:
            with self.subTest(label=label):
                out=self.d/('out-'+label)
                cfg=json.loads((out/'config.json').read_bytes())['quantization_config']
                self.assertEqual(cfg['encoder'],{'seam':'qtip', **({'scale_refit':True} if on else {})})
                self.assertEqual((cfg['wire'],cfg['generator']),('HTR-W1','rdna24'))
                report=json.loads((out/'htr_report.json').read_bytes())
                self.assertIs(report['args']['scale_refit'],on); self.assertEqual(report['args']['seam'],'qtip')
                self.assertTrue(all(('scale_refit' in row)==on for row in report['modules'].values()))
