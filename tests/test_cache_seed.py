"""Seed/rotation compatibility of trusted group caches. Uses the existing CPU CLI fixture."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import legacy_quantize_refit_cli as fixture
from htr import quantize


class CacheSeed(unittest.TestCase):
    @classmethod
    def cli(cls, label, flags, fresh=False):
        return fixture.MainCli.invoke.__func__(cls, label, flags, encode=fresh)

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.d = Path(tmp.name)
        cls.src, cls.hd, cls.W, cls.H = fixture.make_fixture(cls.d)
        cls.calls = {}
        if cls.cli('worker', ['--worker-only', '--cache', 'c-base'], fresh=True) != 0:
            raise AssertionError('worker fixture failed')

    def copy_cache(self, label, *, legacy=False, edit=None):
        dest = self.d / ('c-' + label)
        dest.mkdir()
        for gi in range(len(fixture.GROUPS)):
            name = f'g{gi:04d}.pt'
            group = torch.load(self.d / 'c-base' / name, map_location='cpu', weights_only=True)
            if legacy:
                group.pop('group_seed', None)
            if edit:
                edit(group, gi)
            torch.save(group, dest / name)
        return dest

    def refuse(self, label, flags):
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.cli(label, flags)
        self.assertFalse((self.d / ('out-' + label)).exists())

    def test_worker_records_group_seeds(self):
        for gi in range(len(fixture.GROUPS)):
            g = torch.load(self.d / 'c-base' / f'g{gi:04d}.pt', weights_only=True)
            self.assertIs(type(g.get('group_seed')), int)
            self.assertEqual(g['group_seed'], 1234 + 104729 * gi)

    def test_wrong_seed(self):
        self.refuse('wrong', ['--cache', 'c-base', '--seed', '4321'])

    def test_legacy_wrong_seed_with_refit(self):
        self.copy_cache('legacy-wrong', legacy=True)
        self.refuse('legacy-wrong', ['--cache', 'c-legacy-wrong', '--seed', '4321', '--scale-refit'])

    def test_exact_seed_with_aliased_generator(self):
        self.refuse('alias', ['--cache', 'c-base', '--seed', str(1234 + 2**32)])

    def test_claim_waiter_checks_signs(self):
        dest = self.copy_cache('late', legacy=True)
        packet, held = dest / 'g0000.pt', dest / 'held'
        packet.rename(held)
        (dest / 'g0000.pt.claim').touch()
        with patch.object(quantize.time, 'sleep', side_effect=lambda _: held.rename(packet)) as sleeper:
            self.refuse('late', ['--cache', 'c-late', '--seed', '4321'])
        sleeper.assert_called_once_with(30)

    def test_checks_second_member(self):
        for field in ('su', 'sv'):
            with self.subTest(field=field):
                def change(group, gi):
                    if gi == 1:
                        group['res'][1][field] = group['res'][1][field].clone()
                        group['res'][1][field][0] ^= 1
                label = 'bad-' + field
                self.copy_cache(label, edit=change)
                self.refuse(label, ['--cache', 'c-' + label])

    def test_malformed_metadata(self):
        for val in (True, 1234.0, None):
            with self.subTest(value=val):
                def change(group, gi):
                    if gi == 0:
                        group['group_seed'] = val
                label = 'bad-' + type(val).__name__
                self.copy_cache(label, edit=change)
                self.refuse(label, ['--cache', 'c-' + label])

    def test_matching_caches_equal_fresh_output(self):
        self.copy_cache('legacy-good', legacy=True)
        def fields(label):
            return {k: (v.dtype, tuple(v.shape), v.contiguous().view(torch.uint8).numpy().tobytes())
                    for k, v in fixture.load_tree(self.d / ('out-' + label)).items()}
        for mode, flags in [('plain', []), ('refit', ['--scale-refit'])]:
            with self.subTest(mode=mode):
                self.assertEqual(self.cli(mode, flags, fresh=True), 0)
                expected = fields(mode)
                for cache in ('c-base', 'c-legacy-good'):
                    label = mode + cache
                    self.assertEqual(self.cli(label, ['--cache', cache, *flags]), 0)
                    self.assertEqual(self.calls[label]['encode_calls'], 0)
                    self.assertEqual(fields(label), expected)
