"""Dataset, highly-uneven sharding and microbatch planning.

A sample has a varying sequence length; positions at/after its ``valid_len``
are padded inputs and ``IGNORE`` labels, i.e. they contribute no valid token.
"""
from dataclasses import dataclass

import torch

IGNORE = -100


def make_dataset(seed=1234, n=22, vocab=4):
    g = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(n):
        valid_len = int(torch.randint(1, 7, (1,), generator=g).item())
        seq_len = valid_len + int(torch.randint(0, 4, (1,), generator=g).item())
        x = torch.randint(0, vocab, (seq_len,), generator=g)
        y = torch.randint(0, vocab, (seq_len,), generator=g)
        y[valid_len:] = IGNORE
        samples.append((x, y))
    return samples


def count_valid(samples):
    return int(sum(int((y != IGNORE).sum().item()) for _, y in samples))


@dataclass
class Plan:
    """What one logical update contains for one rank."""

    microbatches: list  # list[list[(x, y)]]

    @property
    def num_microbatches(self):
        return len(self.microbatches)

    @property
    def is_empty(self):
        return self.num_microbatches == 0

    def samples(self):
        out = []
        for mb in self.microbatches:
            out.extend(mb)
        return out


def shard_dataset(samples, split_idx):
    return samples[:split_idx], samples[split_idx:]


def make_plan(shard, mb_sizes):
    """Split one rank's shard into microbatches of given sizes.

    An empty ``mb_sizes`` (or all-zero) gives the fully-empty-rank case:
    the rank still owns no data but must participate in every collective.
    """
    mbs, pos = [], 0
    for size in mb_sizes:
        size = int(size)
        if size == 0:
            continue
        mbs.append(shard[pos:pos + size])
        pos += size
    assert pos == len(shard), (pos, len(shard))
    if not shard:
        # Entire rank owns no data: every logical update for it is empty.
        return Plan([])
    return Plan(mbs)


def oracle_plan(samples):
    """Oracle consumes the whole global dataset, one sample per microbatch."""
    return Plan([[s] for s in samples])
