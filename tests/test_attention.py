"""Attention helpers must be exact on patterns whose structure is known by construction."""

import math

import pytest
import torch

from src.attention import (
    factorisation_error,
    mean_pattern,
    variance_retained,
)

P = 23  # small odd p: these tests are about structure, not about the real model


def _factorised_pattern(p: int, n_heads: int = 2) -> torch.Tensor:
    """softmax over (f(a), g(b), const) -- exactly the form the query-is-constant argument predicts."""
    a = torch.arange(p, dtype=torch.float64)
    scores = torch.zeros(p, p, n_heads, 3, dtype=torch.float64)
    for h in range(n_heads):
        f = torch.sin(2 * math.pi * (h + 1) * a / p)
        g = torch.cos(2 * math.pi * (h + 2) * a / p)
        scores[:, :, h, 0] = f[:, None]
        scores[:, :, h, 1] = g[None, :]
        scores[:, :, h, 2] = 0.3 * (h + 1)
    return scores.softmax(dim=-1)


def test_factorisation_error_is_zero_for_a_factorised_pattern():
    err_a, err_b = factorisation_error(_factorised_pattern(P))
    assert err_a < 1e-12 and err_b < 1e-12


def test_factorisation_error_detects_an_interaction_term():
    """A score that depends on a AND b must not pass the factorisation check."""
    a = torch.arange(P, dtype=torch.float64)
    scores = torch.zeros(P, P, 1, 3, dtype=torch.float64)
    scores[:, :, 0, 0] = a[:, None] * a[None, :] / P  # genuinely joint in (a, b)
    err_a, _ = factorisation_error(scores.softmax(dim=-1))
    assert err_a > 0.05


def test_mean_pattern_is_a_distribution_per_head():
    mean = mean_pattern(_factorised_pattern(P, n_heads=3))
    assert mean.shape == (3, 3)
    assert torch.allclose(mean.sum(-1), torch.ones(3, dtype=torch.float64))


def test_variance_retained_bounds():
    clean = torch.randn(P, P, P, dtype=torch.float64)
    assert variance_retained(clean, clean.clone()) == pytest.approx(1.0)
    # adding a per-example constant changes no prediction, so nothing is lost
    shifted = clean + torch.randn(P, P, 1, dtype=torch.float64)
    assert variance_retained(clean, shifted) == pytest.approx(1.0)
    # predicting the clean mean everywhere retains none of the variance
    assert variance_retained(clean, torch.zeros_like(clean)) == pytest.approx(0.0, abs=1e-9)
