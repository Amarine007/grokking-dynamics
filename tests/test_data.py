import hashlib

import numpy as np
import torch

from src.data import make_cayley_table, make_data, split_indices

P = 113


def test_cayley_table_contents():
    x, y = make_cayley_table(P)
    assert x.shape == (P * P, 3) and y.shape == (P * P,)
    assert x.dtype == torch.long
    assert (x[:, 2] == P).all(), "final token must be '=' (token p)"
    assert x[:, :2].max() == P - 1 and x[:, :2].min() == 0
    assert torch.equal(y, (x[:, 0] + x[:, 1]) % P)
    # every (a, b) pair appears exactly once, ordered by a * p + b
    assert torch.equal(x[:, 0] * P + x[:, 1], torch.arange(P * P))


def test_split_sizes_disjoint_and_complete():
    train, test = split_indices(P, 0.3, 0)
    assert len(train) == int(0.3 * P * P) == 3830
    assert len(test) == P * P - 3830
    assert len(np.intersect1d(train, test)) == 0
    assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(P * P))


def test_split_is_deterministic_in_seed():
    a = make_data(P, 0.3, 0)
    b = make_data(P, 0.3, 0)
    assert torch.equal(a.train_idx, b.train_idx)
    assert torch.equal(a.test_idx, b.test_idx)
    c = make_data(P, 0.3, 1)
    assert not torch.equal(a.train_idx, c.train_idx)


def test_split_matches_committed_results():
    """Pin the exact split for split_seed=0. If this fails, the RNG stream changed
    (e.g. a numpy upgrade) and committed results were produced on a different split."""
    train, _ = split_indices(P, 0.3, 0)
    digest = hashlib.sha256(train.astype(np.int64).tobytes()).hexdigest()
    assert digest == SPLIT_SEED0_SHA256


SPLIT_SEED0_SHA256 = "9eb90b8c95321a373927ac373272d2489cda22b61d989ff3f6f2c3e066c47006"
