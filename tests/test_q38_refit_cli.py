"""Real Qwen worker/assembler CPU branches on synthetic 312-target safetensors.
Only CUDA placement/hardware and Viterbi search are doubled; grouping, H loading/mixture,
encode/refit, packing, proxy sums, cache, assembly and reopened output bytes are real.
Not native GPU, real-checkpoint, quality or performance qualification.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from htr import encode, vit_v2, wire
from htr.scale_refit import refit_packed
from test_quantize_refit_cli import legal_backend

ROOT = Path(__file__).resolve().parents[1]
PIN = '22d0feed6845d0c3f1ae6e2b2e36c445d50feb0c'
FIELDS = ('qweight', 'scales', 'su', 'sv')
RECIPE = dict(scale_refit=True, refit_row_batch=128, damp=0.01)
torch.set_num_threads(2)


def module(pinned=False):
    text = (subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{PIN}:htr/encode_q38.py'], text=True)
            if pinned else (ROOT / 'htr/encode_q38.py').read_text())
    if text.count('.to("cuda")') != 3:
        raise AssertionError('device-only adapter boundary changed')
    m = types.ModuleType('htr._q38_cpu_fixture')
    m.__package__ = 'htr'
    exec(compile(text.replace('.to("cuda")', '.to("cpu")'), '<Qwen CPU-placement adapter>', 'exec'), m.__dict__)
    return m


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class QwenRefit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise AssertionError('CUDA was initialized')
        cls.shared = tempfile.TemporaryDirectory()
        cls.base = Path(cls.shared.name)
        cls.model = cls.base / 'model'
        cls.model.mkdir()
        rng = torch.Generator().manual_seed(786)
        shards = {'selected.safetensors': {}, 'other.safetensors': {}}
        wm = {}
        for li in range(48):
            attn = (['self_attn.' + n for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'indexer.index_qk_proj')]
                    if li % 4 == 3 else ['linear_attn.' + n for n in ('in_proj_qkv', 'in_proj_z', 'out_proj')])
            for sub in attn + ['mlp.shared_expert.' + n for n in ('gate_proj', 'up_proj', 'down_proj')]:
                name = f'model.language_model.layers.{li}.{sub}.weight'
                sf = 'selected.safetensors' if li == 0 else 'other.safetensors'
                wm[name] = sf
                shards[sf][name] = (torch.randn(128, 128, generator=rng) * .03).bfloat16()
        assert len(wm) == 312
        wm['model.language_model.norm.weight'] = 'selected.safetensors'
        shards['selected.safetensors']['model.language_model.norm.weight'] = torch.ones(128, dtype=torch.bfloat16)
        for name, data in shards.items():
            save_file(data, str(cls.model / name), metadata={'format': 'pt'})
        (cls.model / 'model.safetensors.index.json').write_text(json.dumps(dict(metadata={}, weight_map=wm)))
        (cls.model / 'config.json').write_text(json.dumps(dict(quantization_config=dict(quant_method='quark', protected='expert-recipe'), model_type='qwen4_exp')))
        (cls.model / 'tokenizer.json').write_text('{"synthetic": true}')
        cls.hdirs = [cls.base / 'h0', cls.base / 'h1']
        _, groups = module(True).plan(str(cls.model))
        for i, hd in enumerate(cls.hdirs):
            hd.mkdir()
            for gi in (0, 3):
                x = torch.randn(256, 128, generator=rng)
                h = x.T @ x / 256 + torch.eye(128) * (i + .5)
                g = groups[gi]
                torch.save(h, hd / f"language_model.model.layers.{g['layer']}.{g['group']}.pt")
        cls.source_hashes = hashes(cls.model)

    @classmethod
    def tearDownClass(cls):
        assert hashes(cls.model) == cls.source_hashes
        assert not torch.cuda.is_initialized()
        cls.shared.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.m, self.old = module(), module(True)

    def invoke(self, cache, *, cmd='worker', enabled=False, pinned=False, extra=(), forbid_device=False, forbid_encode=False, on_device=None):
        m = self.old if pinned else self.m
        argv = ['htr.encode_q38', cmd, '--model', str(self.model), '--hdir', str(self.hdirs[0]), '--cache', str(cache), '--bits', '4']
        argv += ['--groups', '0,3'] if cmd == 'worker' else ['--partial']
        if enabled:
            argv += ['--scale-refit']
        argv += list(extra)
        calls = []
        real = encode.encode_group
        def counted(jobs, k, **kw):
            if forbid_encode:
                raise AssertionError('unexpected encoder call')
            result = real(jobs, k, **kw)
            calls.append(dict(jobs=jobs, K=k, kwargs=kw, result=result))
            return result
        clock = types.SimpleNamespace(time=lambda: 100., strftime=lambda *a: 'FIXED_TIME')
        with contextlib.ExitStack() as st:
            st.enter_context(patch.object(m, 'time', clock))
            st.enter_context(patch.object(vit_v2, 'viterbi_auto', legal_backend))
            st.enter_context(patch.object(encode, 'encode_group', counted))
            st.enter_context(patch.object(encode, 'VITERBI', encode.VITERBI))
            st.enter_context(patch.object(torch.cuda, 'get_device_name', side_effect=AssertionError('unexpected device query') if forbid_device else on_device, return_value='CPU_TEST'))
            st.enter_context(patch.object(torch.cuda, 'empty_cache', return_value=None))
            st.enter_context(patch.object(sys, 'argv', argv))
            st.enter_context(contextlib.redirect_stdout(io.StringIO()))
            st.enter_context(contextlib.redirect_stderr(io.StringIO()))
            try:
                m.main()
            except SystemExit as exc:
                if enabled and exc.code != 0 and '--scale-refit' not in (ROOT / 'htr/encode_q38.py').read_text():
                    self.fail('missing --scale-refit CLI feature')
                raise
        self.assertFalse(torch.cuda.is_initialized())
        return calls

    def packet(self, cache, gi):
        return torch.load(cache / f'g{gi:04d}.pt', weights_only=False, map_location='cpu')

    def assert_bytes(self, a, b):
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))

    def test_help_exposes_opt_in(self):
        r = subprocess.run([sys.executable, '-B', '-m', 'htr.encode_q38', '--help'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('--scale-refit', r.stdout)

    def test_default_matches_parent_worker_and_both_assemblers(self):
        ca, cb = self.root / 'old', self.root / 'new'
        old, new = self.invoke(ca, pinned=True), self.invoke(cb)
        self.assertEqual((ca / 'plan.json').read_bytes(), (cb / 'plan.json').read_bytes())
        self.assertEqual([c['kwargs'] for c in old], [c['kwargs'] for c in new])
        for gi in (0, 3):
            a, b = self.packet(ca, gi), self.packet(cb, gi)
            self.assertEqual(set(a), set(b))
            for key in ('names', 'K', 'seed', 'rel', 'host', 'seconds'):
                self.assertEqual(a[key], b[key])
            for ra, rb in zip(a['res'], b['res']):
                for key in FIELDS + ('What',):
                    self.assert_bytes(ra[key], rb[key])
        for fake in (False, True):
            oa, ob = self.root / f'parent-{fake}', self.root / f'new-{fake}'
            flag = '--fake-out' if fake else '--out'
            self.invoke(ca, cmd='assemble', pinned=True, extra=(flag, str(oa)))
            self.invoke(cb, cmd='assemble', extra=(flag, str(ob)))
            self.assertEqual(hashes(oa), hashes(ob))
            self.assertEqual((ob / 'other.safetensors').stat().st_ino, (self.model / 'other.safetensors').stat().st_ino)

    def test_all_widths_direct_refit_and_reopened_assembly(self):
        changed = 0
        for k in (2, 3, 4, 5):
            with self.subTest(K=k):
                cache = self.root / f'k{k}'
                mix = f'{self.hdirs[0]}=1,{self.hdirs[1]}=3'
                calls = self.invoke(cache, enabled=True, extra=('--bits', str(k), '--hmix', mix))
                self.assertEqual(len(calls), 2)
                self.assertIs(json.loads((cache / 'plan.json').read_text())['scale_refit'], True)
                for gi, call in zip((0, 3), calls):
                    self.assertEqual(call['kwargs'], dict(seed=1234+104729*gi, shared_su=True, **RECIPE))
                    with patch.object(encode, 'VITERBI', legal_backend):
                        raw = encode.encode_group(call['jobs'], k, seed=1234+104729*gi, shared_su=True)
                    direct = [refit_packed(j['W'], j['H'], r, damp=.01, row_batch=128) for j, r in zip(call['jobs'], raw)]
                    packet = self.packet(cache, gi)
                    self.assertEqual(packet['encoder'], RECIPE)
                    self.assertEqual(packet['scale_refit'], [r['scale_refit'] for r in direct])
                    for fitted, expected, original, saved in zip(call['result'], direct, raw, packet['res']):
                        for field in FIELDS + ('What', 'Q'):
                            self.assert_bytes(fitted[field], expected[field])
                        for field in ('qweight', 'su', 'sv'):
                            self.assert_bytes(fitted[field], original[field])
                        for field in FIELDS:
                            self.assert_bytes(saved[field], expected[field])
                        self.assert_bytes(saved['What'], expected['What'].bfloat16())
                        changed += expected['scale_refit']['changed_rows']
                packed, fake = self.root / f'packed{k}', self.root / f'fake{k}'
                self.invoke(cache, enabled=True, cmd='assemble', extra=('--bits', str(k), '--out', str(packed), '--fake-out', str(fake)), forbid_encode=True, forbid_device=True)
                pt, ft = load_file(str(packed / 'selected.safetensors')), load_file(str(fake / 'selected.safetensors'))
                cfg = json.loads((packed / 'config.json').read_text())['quantization_config']
                self.assertEqual(cfg['protected'], 'expert-recipe')
                self.assertEqual(cfg['htr_overlay']['encoder'], RECIPE)
                # Existing assembler reformats even an unchanged fake config with indent=2.
                self.assertEqual((fake / 'config.json').read_bytes(), json.dumps(json.loads((self.model / 'config.json').read_text()), indent=2).encode())
                for folder in (packed, fake):
                    report = json.loads((folder / 'htr_report.json').read_text())
                    self.assertEqual(report['encoder'], RECIPE)
                for gi in (0, 3):
                    packet = self.packet(cache, gi)
                    for n, res in zip(packet['names'], packet['res']):
                        for field in FIELDS:
                            self.assert_bytes(pt[n[:-7]+'.htr_'+field], res[field])
                        self.assert_bytes(ft[n], res['What'])
        self.assertGreater(changed, 0, 'vacuous scale fixture')

    def test_enabled_report_uses_worker_hessian_provenance(self):
        cache = self.root / 'mixed-hessian-report'
        mixture = f'{self.hdirs[0]}=1,{self.hdirs[1]}=3'
        self.invoke(cache, enabled=True, extra=('--hmix', mixture))
        out = self.root / 'report-out'
        self.invoke(cache, enabled=True, cmd='assemble', extra=('--out', str(out)), forbid_device=True)
        report = json.loads((out / 'htr_report.json').read_text())
        self.assertEqual(report['hdir'], str(self.hdirs[0]))
        self.assertEqual(report['hmix'], mixture)

    def test_enabled_and_legacy_resume_do_not_rewrite(self):
        for enabled in (False, True):
            cache = self.root / str(enabled)
            self.invoke(cache, enabled=enabled, pinned=not enabled)
            before = hashes(cache)
            self.invoke(cache, enabled=enabled, forbid_encode=True)
            self.assertEqual(before, hashes(cache))

    def test_wrong_plan_recipe_before_device_and_no_mutation(self):
        for stored, requested in ((None, True), (False, True), (True, False), ('true', False), (1, True)):
            with self.subTest(stored=stored, requested=requested):
                cache = self.root / f'{stored}-{requested}'
                self.invoke(cache, pinned=True)
                p = json.loads((cache / 'plan.json').read_text())
                if stored is not None:
                    p['scale_refit'] = stored
                (cache / 'plan.json').write_text(json.dumps(p))
                before = hashes(cache)
                with self.assertRaisesRegex(ValueError, 'scale.refit'):
                    self.invoke(cache, enabled=requested, forbid_device=True)
                self.assertEqual(before, hashes(cache))

    def test_worker_identity_mismatch_before_device(self):
        cache = self.root / 'identity'
        self.invoke(cache)
        before = hashes(cache)
        for extra in (('--bits', '3'), ('--seed', '901'), ('--hdir', 'WRONG'), ('--hmix', 'WRONG=1')):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.invoke(cache, extra=extra, forbid_device=True)
            self.assertEqual(before, hashes(cache))

    def test_enabled_worker_backend_identity_is_not_interchangeable(self):
        cache = self.root / 'backend-identity'
        self.invoke(cache, enabled=True)
        before = hashes(cache)
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.invoke(cache, enabled=True, extra=('--viterbi', 'ref'), forbid_device=True)
        self.assertEqual(before, hashes(cache))

    def test_orphan_worker_cache_rejected(self):
        cache = self.root / 'orphan'
        cache.mkdir()
        (cache / 'g0000.pt').write_bytes(b'unknown origin')
        before = hashes(cache)
        with self.assertRaisesRegex(ValueError, 'provenance|plan'):
            self.invoke(cache, enabled=True, forbid_device=True)
        self.assertEqual(before, hashes(cache))

    def test_atomic_plan_creator_and_race_rejection(self):
        publish = getattr(self.m, 'write_plan_if_absent', None)
        self.assertTrue(callable(publish), 'missing atomic publisher')
        path = self.root / 'plan.json'
        candidates = [dict(writer=i, payload=str(i)*1000) for i in range(8)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda p: publish(str(path), p), candidates))
        self.assertIn(json.loads(path.read_text()), candidates)
        old = path.read_bytes()
        publish(str(path), dict(other=True))
        self.assertEqual(path.read_bytes(), old)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['plan.json'])
        cache = self.root / 'race'
        def competing(path, plan):
            publish(path, dict(plan, scale_refit=not plan.get('scale_refit', False)))
        with patch.object(self.m, 'write_plan_if_absent', competing):
            with self.assertRaisesRegex(ValueError, 'scale.refit'):
                self.invoke(cache, forbid_device=True)
        self.assertFalse((cache / 'g0000.pt').exists())

    def test_assembly_rejects_recipe_or_cardinality_before_output(self):
        cache = self.root / 'bad-packets'
        self.invoke(cache, enabled=True)
        original = (cache / 'g0000.pt').read_bytes()
        for name in ('metadata', 'diagnostics', 'members', 'width', 'boolean'):
            with self.subTest(name=name):
                packet = self.packet(cache, 0)
                if name == 'metadata':
                    packet.pop('encoder')
                elif name == 'diagnostics':
                    packet['scale_refit'] = []
                elif name == 'members':
                    packet['res'] = packet['res'][:1]
                elif name == 'width':
                    packet['K'] = 6
                else:
                    packet['encoder']['scale_refit'] = 1
                torch.save(packet, cache / 'g0000.pt')
                out = self.root / ('bad-' + name)
                with self.assertRaises(ValueError):
                    self.invoke(cache, enabled=True, cmd='assemble', extra=('--mixed', '--out', str(out)), forbid_device=True)
                self.assertFalse(out.exists())
                self.assertFalse(Path(str(out)+'.partial').exists())
                (cache / 'g0000.pt').write_bytes(original)
        out = self.root / 'default-cannot-read-refit'
        with self.assertRaisesRegex(ValueError, 'scale.refit'):
            self.invoke(cache, cmd='assemble', extra=('--out', str(out)), forbid_device=True)
        self.assertFalse(out.exists())

    def test_enabled_assemble_requires_plan_but_legacy_orphans_remain_readable(self):
        for enabled in (False, True):
            cache = self.root / str(enabled)
            self.invoke(cache, enabled=enabled)
            (cache / 'plan.json').unlink()
            out = self.root / f'out-{enabled}'
            if enabled:
                with self.assertRaisesRegex(ValueError, 'plan|provenance'):
                    self.invoke(cache, enabled=True, cmd='assemble', extra=('--out', str(out)), forbid_device=True)
                self.assertFalse(out.exists())
            else:
                self.invoke(cache, cmd='assemble', extra=('--out', str(out)), forbid_device=True)
                self.assertTrue(out.is_dir())

    def test_mixed_width_assembly_and_nonoverwrite(self):
        cache = self.root / 'mixed'
        self.invoke(cache, enabled=True, extra=('--bits', '3'))
        other = self.root / 'other'
        self.invoke(other, enabled=True, extra=('--bits', '5'))
        shutil.copyfile(other / 'g0003.pt', cache / 'g0003.pt')
        out = self.root / 'mixed-out'
        self.invoke(cache, enabled=True, cmd='assemble', extra=('--mixed', '--out', str(out)), forbid_device=True)
        report = json.loads((out / 'htr_report.json').read_text())
        self.assertEqual({v['K'] for v in report['modules'].values()}, {3, 5})
        before = hashes(out)
        with self.assertRaises((ValueError, AssertionError, FileExistsError)):
            self.invoke(cache, enabled=True, cmd='assemble', extra=('--mixed', '--out', str(out)), forbid_device=True)
        self.assertEqual(before, hashes(out))

    def test_unsupported_widths_reject_opt_in_only(self):
        for k in (6, 8):
            cache = self.root / f'bad{k}'
            with self.assertRaises(SystemExit) as cm:
                self.invoke(cache, enabled=True, extra=('--bits', str(k)), forbid_device=True)
            self.assertEqual(cm.exception.code, 2)
            self.assertFalse(cache.exists())
            with self.assertRaisesRegex(AssertionError, 'unexpected encoder call'):
                self.invoke(cache, extra=('--bits', str(k)), forbid_encode=True)
            self.assertNotIn('scale_refit', json.loads((cache / 'plan.json').read_text()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
