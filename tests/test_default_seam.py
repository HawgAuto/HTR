"""QTIP CLI default and explicit legacy-cache compatibility; CPU boundary fixtures.

HF construction/input capture are doubled. Parser, grouping, cache checks and
checkpoint writer are real. These tiny fixtures are NOT encoder/quality evidence.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from htr import quantize

NAME = 'model.layers.0.mlp.down_proj'


class Parsed(Exception):
    pass


class DefaultSeam(unittest.TestCase):
    def parsed(self, extra=()):
        original = argparse.ArgumentParser.parse_args
        def capture(parser, *a, **kw):
            raise Parsed(original(parser, *a, **kw))
        with patch.object(sys, 'argv', ['htr.quantize', '--model', 'unused', '--out', 'unused', '--calib', 'unused', *extra]), patch.object(argparse.ArgumentParser, 'parse_args', capture):
            with self.assertRaises(Parsed) as caught:
                quantize.main()
        return caught.exception.args[0]

    def test_default_is_qtip(self):
        self.assertEqual(self.parsed().seam, 'qtip')

    def test_explicit_old_seam_remains_selectable(self):
        self.assertEqual(self.parsed(['--seam', 'htr']).seam, 'htr')

    def test_explicit_qtip_remains_selectable(self):
        self.assertEqual(self.parsed(['--seam', 'qtip']).seam, 'qtip')

    def test_v1_qtip_rejected_before_input_access(self):
        argv = ['htr.quantize', '--model', 'unused', '--out', 'unused', '--calib', 'unused', '--viterbi', 'v1', '--seam', 'qtip']
        with patch.object(sys, 'argv', argv), patch.object(quantize, 'ckpt_index', side_effect=Parsed('unexpected input access')) as access, patch.object(quantize.encode, 'VITERBI', quantize.encode.VITERBI), contextlib.redirect_stderr(io.StringIO()):
            try:
                quantize.main()
            except (Parsed, SystemExit) as exc:
                result = exc
            else:
                result = None
        self.assertIsInstance(result, SystemExit)
        self.assertEqual(result.code, 2)
        self.assertEqual(access.call_count, 0)

    def test_v1_explicit_old_seam_not_rejected(self):
        argv = ['htr.quantize', '--model', 'unused', '--out', 'unused', '--calib', 'unused', '--viterbi', 'v1', '--seam', 'htr']
        with patch.object(sys, 'argv', argv), patch.object(quantize, 'ckpt_index', side_effect=Parsed('input reached')), patch.object(quantize.encode, 'VITERBI', quantize.encode.VITERBI):
            with self.assertRaises(Parsed):quantize.main()


class CachedMain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src, self.cache, self.hd, self.out = [self.root / x for x in ('src', 'cache', 'hessians', 'out')]
        for p in (self.src, self.cache, self.hd):p.mkdir()
        self.weight = torch.ones(128, 128, dtype=torch.bfloat16)
        save_file({NAME+'.weight': self.weight, 'model.norm.weight': torch.ones(128)}, str(self.src/'model.safetensors'))
        (self.src/'config.json').write_text(json.dumps(dict(architectures=['CPUFixture'])))
        torch.save({NAME: torch.eye(128)}, self.hd/'layer000.pt')
        self.packet = dict(qweight=torch.zeros((8,8,24),dtype=torch.int32),scales=torch.ones(128,dtype=torch.float16),su=torch.zeros(16,dtype=torch.uint8),sv=torch.zeros(16,dtype=torch.uint8),What=self.weight)

    def invoke(self, stored='qtip', requested=None, legacy=False, claim=False, bad=None):
        data = dict(names=[NAME],K=3,seam=stored,res=[self.packet],rel=[0.125])
        if legacy:data.pop('seam')
        if bad:data.update(bad)
        cpath=self.cache/'g0000.pt';torch.save(data,cpath)
        before=hashlib.sha256(cpath.read_bytes()).hexdigest()
        if claim:(self.cache/'g0000.pt.claim').touch()
        argv=['htr.quantize','--model',str(self.src),'--out',str(self.out),'--bits','3','--calib','unused','--hessians',str(self.hd),'--cache',str(self.cache),'--offload']
        if requested is not None:argv+=['--seam',requested]
        def model(*a,**kw):
            m=torch.nn.Module();p=m
            for part in NAME.split('.')[:-1]:
                child=torch.nn.Module();p.add_module(part,child);p=child
            mod=torch.nn.Linear(128,128,bias=False);mod.weight=torch.nn.Parameter(self.weight.clone(),requires_grad=False);p.add_module('down_proj',mod)
            return m
        from transformers import AutoModelForCausalLM,AutoTokenizer
        actual_exists=quantize.os.path.exists
        first=[claim]
        def exists(path):
            if str(path)==str(cpath) and first[0]:first[0]=False;return False
            return actual_exists(path)
        with contextlib.ExitStack() as st:
            st.enter_context(patch.object(sys,'argv',argv))
            st.enter_context(patch.object(AutoModelForCausalLM,'from_pretrained',side_effect=model))
            st.enter_context(patch.object(AutoTokenizer,'from_pretrained',return_value=object()))
            st.enter_context(patch.object(quantize,'capture',return_value=({}, {NAME:(0,0,(1,64,128))}, {})))
            st.enter_context(patch.object(quantize.encode,'VITERBI',quantize.encode.VITERBI))
            enc=st.enter_context(patch.object(quantize.encode,'encode_group',side_effect=RuntimeError('forbidden encode')))
            st.enter_context(patch.object(quantize.os.path,'exists',side_effect=exists))
            try:
                with contextlib.redirect_stdout(io.StringIO()):quantize.main()
                error=None
            except (AssertionError,ValueError) as exc:error=exc
            self.assertEqual(enc.call_count,0)
        self.assertEqual(hashlib.sha256(cpath.read_bytes()).hexdigest(),before)
        self.assertFalse(torch.cuda.is_initialized())
        return error

    def test_default_assembles_qtip_and_records_provenance(self):
        self.assertIsNone(self.invoke())
        cfg=json.loads((self.out/'config.json').read_text())
        self.assertEqual(cfg['quantization_config']['encoder'],{'seam':'qtip'})
        self.assertEqual(json.loads((self.out/'htr_report.json').read_text())['args']['seam'],'qtip')

    def test_explicit_old_seam_accepts_legacy_cache(self):
        self.assertIsNone(self.invoke(stored='htr',requested='htr',legacy=True))
        self.assertNotIn('encoder',json.loads((self.out/'config.json').read_text())['quantization_config'])

    def test_default_rejects_legacy_old_cache(self):
        error=self.invoke(stored='htr',legacy=True)
        self.assertIsInstance(error,ValueError)
        self.assertFalse(self.out.exists())

    def test_claim_completion_cannot_bypass_seam_check(self):
        error=self.invoke(stored='htr',requested='qtip',claim=True)
        self.assertIsInstance(error,ValueError)
        self.assertFalse(self.out.exists())

    def test_wrong_name_rejected_before_write(self):
        error=self.invoke(requested='qtip',bad={'names':['wrong']})
        self.assertIsInstance(error,ValueError)
        self.assertFalse(self.out.exists())


if __name__=='__main__':unittest.main()
