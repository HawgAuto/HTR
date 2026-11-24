"""Run the original default/legacy-seam checks using a seed-consistent cached packet."""
import torch
import legacy_default_seam as original


class DefaultSeam(original.DefaultSeam):
    pass


class CachedMain(original.CachedMain):
    def setUp(self):
        super().setUp()
        # The old synthetic cache used all-plus signs. Use the fixture's actual seed;
        # pack independently of the product checker. All inherited assertions stand.
        for field, seed in [('su', 1234), ('sv', 1234 + 7919)]:
            gen = torch.Generator().manual_seed(seed)
            neg = torch.randint(0, 2, (128,), generator=gen).eq(0).tolist()
            self.packet[field] = torch.tensor([
                sum(int(neg[i + j]) << j for j in range(8)) for i in range(0, 128, 8)
            ], dtype=torch.uint8)
