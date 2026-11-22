"""Run every pre-QTIP refit assertion with its original comparison recipe.

The old test module is preserved byte-for-byte as legacy_quantize_refit_cli.py.
Its d4ae293 oracle expects the HTR seam. Only previously implicit old-seam
arguments are made explicit; explicit QTIP cases and all assertions are unchanged.
Default behavior of the new CLI is covered separately in test_default_seam and
test_stack_seam_refit, without this adapter.
"""
import legacy_quantize_refit_cli as _legacy

# The routed- and shared-expert suites import this fixture by its original path.
legal_backend = _legacy.legal_backend
Help = _legacy.Help
Config = _legacy.Config
RefitGroup = _legacy.RefitGroup


class MainCli(_legacy.MainCli):
    @classmethod
    def invoke(cls, label, extra, pinned=False, encode=True, policy=None):
        extra = list(extra)
        if '--seam' not in extra:
            extra.extend(['--seam', 'htr'])
        return super().invoke(label, extra, pinned=pinned, encode=encode, policy=policy)
