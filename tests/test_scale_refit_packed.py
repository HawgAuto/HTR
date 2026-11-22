"""Opt-in encoder/packed adapter tests; no model load or GPU."""
import importlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import types
import unittest

import torch
from safetensors.torch import load_file, save, save_file
from htr import encode, wire

torch.set_num_threads(2)


def packed_fixture(k=3):
    g = torch.Generator().manual_seed(123)
    return {'qweight': torch.randint(-2**31, 2**31, (8, 8, 8*k), generator=g, dtype=torch.int32),
            'scales': torch.ones(128, dtype=torch.float16),
            'su': wire.pack_signs((torch.randint(0, 2, (128,), generator=g)*2-1).float()),
            'sv': wire.pack_signs((torch.randint(0, 2, (128,), generator=g)*2-1).float())}


def legal_path_backend(x, w, cb, k):
    # Test-only valid circular-path backend, not a quality/optimality oracle.
    words = torch.arange(x.shape[0]*8*k, device=x.device, dtype=torch.int32).reshape(x.shape[0], 8*k)
    return wire.unpack_states(words, k)


class PackedAdapterTests(unittest.TestCase):
    def adapter(self):
        module = importlib.import_module('htr.scale_refit')
        self.assertTrue(hasattr(module, 'refit_packed'), 'packed refit adapter is missing')
        return module.refit_packed

    def test_codes_signs_bytes_preserved_and_reopened_decode_matches(self):
        fn = self.adapter()
        for k in (2, 3, 4, 5):
            with self.subTest(k=k):
                p = packed_fixture(k)
                frozen = {name: value.clone() for name, value in p.items()}
                desired = torch.full((128,), 1.5, dtype=torch.float16)
                w = wire.dequant(p['qweight'], desired, p['su'], p['sv'])
                result = fn(w, torch.eye(128), p, damp=0., row_batch=17)
                selected = {name: result[name] for name in p}
                self.assertEqual(len(save(p)), len(save(selected)))
                for name in ('qweight', 'su', 'sv'):
                    self.assertTrue(torch.equal(result[name], p[name]))
                self.assertTrue(all(torch.equal(p[name], frozen[name]) for name in p))
                self.assertTrue(torch.equal(result['scales'], desired))
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp)/'refit.safetensors'
                    save_file(selected, str(path))
                    readback = load_file(str(path))
                    actual = wire.dequant(readback['qweight'], readback['scales'], readback['su'], readback['sv'])
                self.assertTrue(torch.equal(actual, result['What']))
                self.assertTrue(torch.equal(wire.dequant_Q(result['qweight'], result['scales']), result['Q']))
                self.assertLess(result['scale_refit']['after'], result['scale_refit']['before'])
                self.assertEqual(result['scale_refit']['changed_rows'], 128)

    def test_invalid_wire_shapes_types_and_damping_rejected(self):
        fn = self.adapter()
        p = packed_fixture()
        w, h = torch.zeros(128, 128), torch.eye(128)
        bad = [dict(p, qweight=p['qweight'].long()), dict(p, su=p['su'].long()),
               dict(p, sv=p['sv'][:1]), dict(p, scales=p['scales'].float()),
               dict(p, qweight=p['qweight'][:, :, :23])]
        for value in bad:
            with self.subTest(keys=list(value)):
                with self.assertRaises((ValueError, TypeError)):
                    fn(w, h, value)
        for damp in (-.1, float('nan'), float('inf')):
            with self.subTest(damp=damp):
                with self.assertRaises(ValueError):
                    fn(w, h, p, damp=damp)

    def test_zero_target_zero_scales_round_trip(self):
        fn = self.adapter()
        p = packed_fixture()
        p['scales'].zero_()
        result = fn(torch.zeros(128, 128), torch.zeros(128, 128), p)
        self.assertTrue(torch.equal(result['scales'], p['scales']))
        self.assertTrue(torch.equal(result['What'], torch.zeros(128, 128)))


class EncoderIntegrationTests(unittest.TestCase):
    def test_opt_in_preserves_default_and_legacy_codes(self):
        import inspect
        self.assertIn('scale_refit', inspect.signature(encode.encode_group).parameters,
                      'encode_group has no opt-in refit integration')
        root = Path(__file__).resolve().parents[1]
        legacy = types.ModuleType('htr._legacy_encoder')
        legacy.__package__ = 'htr'
        source = subprocess.check_output(['git', '-C', str(root), 'show', 'bd0b92c:htr/encode.py'], text=True)
        exec(compile(source, '<pinned-original-encoder>', 'exec'), legacy.__dict__)
        legacy.VITERBI = legal_path_backend
        previous = encode.VITERBI
        g = torch.Generator().manual_seed(772)
        jobs = [{'W': torch.randn(128, 128, generator=g)*.02, 'H': torch.eye(128)}]
        try:
            encode.VITERBI = legal_path_backend
            old = legacy.encode_group(jobs, 3)[0]
            default = encode.encode_group(jobs, 3)[0]
            disabled = encode.encode_group(jobs, 3, scale_refit=False)[0]
            enabled = encode.encode_group(jobs, 3, scale_refit=True)[0]
        finally:
            encode.VITERBI = previous
        self.assertEqual(set(default), set(old))
        for name in old:
            self.assertTrue(torch.equal(default[name], old[name]), name)
            self.assertTrue(torch.equal(disabled[name], old[name]), name)
        for name in ('qweight', 'su', 'sv'):
            self.assertTrue(torch.equal(enabled[name], old[name]), name)
        self.assertTrue(torch.equal(enabled['What'], wire.dequant(enabled['qweight'], enabled['scales'], enabled['su'], enabled['sv'])))
        self.assertLessEqual(enabled['scale_refit']['after'], enabled['scale_refit']['before'])


if __name__ == '__main__':
    unittest.main()
