"""Dense worker/assembler input binding, using the existing legal-path CPU CLI fixture."""
import hashlib, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import torch
import legacy_quantize_refit_cli as f
from htr import quantize as q

class Inputs(unittest.TestCase):
    @classmethod
    def cli(cls, label, flags, fresh=False):
        return f.MainCli.invoke.__func__(cls, label, flags, encode=fresh)

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory(); cls.addClassCleanup(tmp.cleanup)
        cls.d = Path(tmp.name); cls.src, cls.hd, cls.W, cls.H = f.make_fixture(cls.d)
        cls.calls = {}
        if cls.cli('producer', ['--cache', 'c-base', '--worker-only'], True) != 0:
            raise AssertionError('producer fixture failed')

    def copy(self, cache, edit=None):
        dest = self.d / cache; dest.mkdir()
        for gi, p in enumerate(sorted((self.d/'c-base').glob('g*.pt'))):
            z = torch.load(p, weights_only=True)
            if edit: edit(z, gi)
            torch.save(z, dest/p.name)
        return dest

    def refuse(self, label, flags, regex):
        with self.assertRaisesRegex(ValueError, regex): self.cli(label, flags)
        self.assertFalse((self.d/('out-'+label)).exists())

    def test_worker_binds_every_member(self):
        def oracle(v):
            tag = f'{v.dtype}|{tuple(v.shape)}|{tuple(v.stride())}|'.encode()
            return hashlib.sha256(tag+v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for gi, names in enumerate(f.GROUPS):
            pack = torch.load(self.d/'c-base'/f'g{gi:04d}.pt', weights_only=True)
            self.assertEqual(pack.get('input_identity'), {'format':'typed-layout-sha256-v1','members':{
                n:{'weight':oracle(self.W[n]),'hessian':oracle(self.H[n])} for n in names}})

    def test_changed_weight_second_member(self):
        with patch.dict(self.W, {f.UP: -self.W[f.UP]}):
            self.refuse('stale-weight', ['--cache','c-base'], 'weight.*up_proj')

    def test_changed_hessian_from_late_worker(self):
        hd = self.d/'other-gram'; hd.mkdir(); h = self.H[f.UP].clone(); h[0,0] += 1
        torch.save({**self.H, f.UP:h, f.GATE:h}, hd/'layer000.pt')
        c = self.copy('c-late-gram'); p=c/'g0001.pt'; held=c/'held'; p.rename(held); (c/'g0001.pt.claim').touch()
        with patch.object(type(self), 'hd', hd), patch.object(q.time, 'sleep', side_effect=lambda s:held.rename(p)) as wait:
            self.refuse('late-gram',['--cache','c-late-gram','--scale-refit'],'hessian.*gate_proj')
        wait.assert_called_once_with(30)

    def test_typed_corruption_is_not_legacy(self):
        self.copy('c-invalid-inputs', lambda z, g:z.update(input_identity=None))
        self.refuse('invalid-inputs', ['--cache','c-invalid-inputs'], 'input identity')

    def test_legacy_warning_or_strict_refusal(self):
        self.copy('c-legacy-inputs', lambda z,g:z.pop('input_identity',None))
        self.assertEqual(self.cli('legacy',['--cache','c-legacy-inputs']),0)
        self.assertIn('unverified weights/Hessians', self.calls['legacy']['log'])
        self.refuse('strict-old', ['--cache','c-legacy-inputs','--require-cache-inputs'], 'legacy')

    def test_strict_content_not_path(self):
        c = self.copy('c-moved-inputs'); hd=self.d/'reserialized'; hd.mkdir()
        torch.save({n:v.clone() for n,v in reversed(list(self.H.items()))}, hd/'layer000.pt')
        raw=lambda name:{k:(v.dtype,v.shape,v.contiguous().view(torch.uint8).numpy().tobytes())
                         for k,v in f.load_tree(self.d/('out-'+name)).items()}
        for label,flags in [('plain',[]),('fit',['--scale-refit'])]:
            self.assertEqual(self.cli('fresh-'+label,flags,True),0)
            with patch.object(type(self),'hd',hd):
                self.assertEqual(self.cli(label,['--cache',c.name,'--require-cache-inputs',*flags]),0)
            self.assertEqual(self.calls[label]['encode_calls'],0)
            self.assertEqual(raw(label),raw('fresh-'+label))

    def test_offload_cache_hit_does_not_move_hessian_to_device(self):
        with patch.object(q.LayerDirHessians,'__getitem__',side_effect=AssertionError('unnecessary device copy')):
            self.assertEqual(self.cli('host-only',['--cache','c-base','--require-cache-inputs']),0)

    def test_equal_gram_with_different_stride_refused(self):
        hd = self.d/'colmajor'; hd.mkdir()
        changed = {n:v.T.contiguous().T for n,v in self.H.items()}
        self.assertTrue(all(torch.equal(changed[n],v) and changed[n].stride()!=v.stride() for n,v in self.H.items()))
        torch.save(changed, hd/'layer000.pt')
        with patch.object(type(self),'hd',hd):
            self.refuse('layout',['--cache','c-base'],'hessian.*down_proj')
