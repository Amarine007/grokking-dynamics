"""Restricted/excluded loss must be exact on logits whose Fourier content is known."""

import math

import pytest
import torch

from src.data import make_data
from src.measures import (
    constant_direction,
    excluded_logits,
    first_step,
    key_directions,
    phase_boundaries,
    progress_measures,
    project,
    restricted_logits,
    step_of_extreme,
)

SMALL = 23  # odd, small: every test here is exact linear algebra, so size is irrelevant
KEY = [2, 7]


def _grid(fn, p: int = SMALL, c: int = SMALL + 1) -> torch.Tensor:
    """[p, p, c] grid with value fn(a, b, c)."""
    a = torch.arange(p, dtype=torch.float64)[:, None, None]
    b = torch.arange(p, dtype=torch.float64)[None, :, None]
    cc = torch.arange(c, dtype=torch.float64)[None, None, :]
    return fn(a, b, cc).expand(p, p, c).clone()


def _w(k: int, p: int = SMALL) -> float:
    return 2 * math.pi * k / p


@pytest.mark.parametrize("variant", ["sum", "products"])
def test_directions_are_orthonormal_and_orthogonal_to_constant(variant):
    d = torch.cat([constant_direction(SMALL), key_directions(SMALL, KEY, variant)])
    gram = torch.einsum("iab,jab->ij", d, d)
    assert (gram - torch.eye(len(d), dtype=torch.float64)).abs().max() < 1e-12


def test_sum_directions_lie_inside_the_product_span():
    s, prod = key_directions(SMALL, KEY, "sum"), key_directions(SMALL, KEY, "products")
    assert torch.allclose(project(s.permute(1, 2, 0), prod), s.permute(1, 2, 0), atol=1e-12)


def test_clock_logits_survive_restriction_and_vanish_when_excluded():
    """cos(w_k(a+b-c)) plus a per-class bias is exactly what restricted loss keeps."""
    bias = _grid(lambda a, b, c: 0.3 * c + 0 * a * b)
    clock = _grid(lambda a, b, c: sum(torch.cos(_w(k) * (a + b - c)) for k in KEY))
    grid = bias + clock
    for variant in ("sum", "products"):
        assert torch.allclose(restricted_logits(grid, KEY, variant), grid, atol=1e-10)
        assert torch.allclose(excluded_logits(grid, KEY, variant), bias, atol=1e-10)


def test_a_minus_b_terms_separate_the_two_variants():
    """cos(w(a-b)) is at a key frequency but not a function of a+b: only `products` keeps it."""
    grid = _grid(lambda a, b, c: torch.cos(_w(KEY[0]) * (a - b)) + 0 * c)
    assert restricted_logits(grid, KEY, "sum").abs().max() < 1e-10
    assert torch.allclose(restricted_logits(grid, KEY, "products"), grid, atol=1e-10)


def test_non_key_frequencies_are_left_to_excluded():
    grid = _grid(lambda a, b, c: torch.sin(_w(5) * a) * torch.cos(_w(9) * (a + b)) + 0 * c)
    assert restricted_logits(grid, KEY).abs().max() < 1e-10
    assert torch.allclose(excluded_logits(grid, KEY), grid, atol=1e-10)


def test_restricted_and_excluded_partition_random_logits():
    """restricted + excluded = logits + constant part, exactly, for any input."""
    torch.manual_seed(0)
    grid = torch.randn(SMALL, SMALL, SMALL + 1, dtype=torch.float64)
    for variant in ("sum", "products"):
        total = restricted_logits(grid, KEY, variant) + excluded_logits(grid, KEY, variant)
        assert torch.allclose(total, grid + project(grid, constant_direction(SMALL)), atol=1e-10)


def test_clean_losses_match_direct_cross_entropy():
    data = make_data(SMALL, 0.3, 0)
    torch.manual_seed(1)
    grid = torch.randn(SMALL, SMALL, SMALL + 1, dtype=torch.float64)
    m = progress_measures(grid, data, KEY)
    flat = grid.reshape(-1, SMALL + 1)
    expected = torch.nn.functional.cross_entropy(flat[data.train_idx], data.train_y).item()
    assert m["train_loss"] == pytest.approx(expected, rel=1e-12)
    # A grid that is already pure clock + bias loses nothing under restriction.
    clock = _grid(lambda a, b, c: 5 * torch.cos(_w(KEY[0]) * (a + b - c)))
    m = progress_measures(clock, data, KEY)
    assert m["restricted_loss_test"] == pytest.approx(m["test_loss"], rel=1e-9)


def test_boundary_helpers():
    steps = [0, 10, 20, 30, 40]
    assert first_step(steps, [5, 4, 3, 2, 1], below=3) == 30
    assert first_step(steps, [0.0, 0.2, 0.6, 0.9, 1.0], at_least=0.9) == 30
    assert first_step(steps, [5, 5, 5, 5, 5], below=1) is None
    assert step_of_extreme(steps, [3, 1, 1, 2, 4], "min") == 10  # earliest on ties
    assert step_of_extreme(steps, [3, 1, 1, 2, 4], "max") == 40
    with pytest.raises(ValueError):
        first_step(steps, [1] * 5)


def test_phase_boundaries_on_synthetic_curves():
    p = SMALL
    u = math.log(p)
    rows = [
        dict(step=0, test_loss=u, test_acc=0.0, excluded_loss_train=u, restricted_loss_test=u),
        dict(step=100, test_loss=9.0, test_acc=0.0, excluded_loss_train=0.1, restricted_loss_test=6.0),
        dict(step=200, test_loss=8.0, test_acc=0.05, excluded_loss_train=2.0, restricted_loss_test=2.0),
        dict(step=300, test_loss=2.0, test_acc=0.6, excluded_loss_train=5.0, restricted_loss_test=0.5),
        dict(step=400, test_loss=0.1, test_acc=0.995, excluded_loss_train=5.0, restricted_loss_test=0.01),
    ]
    rules = {"restricted_split": "test", "cleanup_end_test_acc": 0.99, "jump_test_acc": 0.5}
    assert phase_boundaries(rows, p, rules) == {
        "memorization_end": 100,
        "cleanup_start": 300,
        "cleanup_end": 400,
        "restricted_peak": 100,
        "restricted_beats_uniform": 200,
        "test_acc_jump": 300,
    }
