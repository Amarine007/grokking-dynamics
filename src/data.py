"""Modular addition dataset: the full Cayley table of (a + b) mod p.

Every example is the token sequence [a, b, =] with a, b in 0..p-1 and "=" encoded
as token p, so the input vocabulary has p + 1 tokens. The label is (a + b) mod p,
predicted at the final position only.

The train/test split is a seeded permutation of all p^2 pairs, so it is fully
determined by (p, train_frac, split_seed). Everything is full-batch: no
DataLoader, no shuffling, no augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ModularAdditionData:
    p: int
    all_x: torch.Tensor  # (p^2, 3) long, row index = a * p + b
    all_y: torch.Tensor  # (p^2,) long
    train_idx: torch.Tensor  # indices into all_x, in permutation order
    test_idx: torch.Tensor

    @property
    def train_x(self) -> torch.Tensor:
        return self.all_x[self.train_idx]

    @property
    def train_y(self) -> torch.Tensor:
        return self.all_y[self.train_idx]

    @property
    def test_x(self) -> torch.Tensor:
        return self.all_x[self.test_idx]

    @property
    def test_y(self) -> torch.Tensor:
        return self.all_y[self.test_idx]


def make_cayley_table(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All p^2 inputs [a, b, =] and labels (a + b) mod p, ordered by a * p + b."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)
    x = torch.stack([a, b, eq], dim=1)
    y = (a + b) % p
    return x, y


def split_indices(p: int, train_frac: float, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/test split from a seeded permutation of all p^2 pairs."""
    n = p * p
    perm = np.random.default_rng(split_seed).permutation(n)
    n_train = int(train_frac * n)
    return perm[:n_train], perm[n_train:]


def make_data(p: int, train_frac: float, split_seed: int) -> ModularAdditionData:
    x, y = make_cayley_table(p)
    train_idx, test_idx = split_indices(p, train_frac, split_seed)
    return ModularAdditionData(
        p=p,
        all_x=x,
        all_y=y,
        train_idx=torch.from_numpy(train_idx).long(),
        test_idx=torch.from_numpy(test_idx).long(),
    )
