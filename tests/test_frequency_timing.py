"""Frequency-timing helpers must be exact on spectra and weights with known structure."""

import numpy as np
import pytest
import torch

from src.fourier import fourier_basis
from src.frequency_timing import (
    frequency_ranks,
    neuron_logit_map,
    rank_permutation_test,
    settled_step,
    spectra,
    top_frequencies,
)
from src.model import ModelConfig, Transformer

P = 23


def test_ranks_and_top_frequencies():
    spec = np.array([0.5, 0.01, 0.2, 0.05, 0.2, 0.04])  # constant first; k=2 and k=4 tie
    assert frequency_ranks(spec) == {2: 1, 4: 2, 3: 3, 5: 4, 1: 5}
    assert top_frequencies(spec, 3) == [2, 3, 4]


def test_settled_step():
    steps = [0, 10, 20, 30, 40]
    assert settled_step(steps, [True, False, True, True, True]) == 20
    assert settled_step(steps, [True] * 5) == 0
    assert settled_step(steps, [True, True, True, True, False]) is None


def test_spectra_find_planted_frequencies_in_both_matrices():
    torch.manual_seed(0)
    cfg = ModelConfig(d_vocab=P + 1, d_vocab_out=P + 1, n_ctx=3, d_model=16, n_heads=2, d_head=8, d_mlp=12)
    model = Transformer(cfg)
    basis, _ = fourier_basis(P, torch.float32)
    with torch.no_grad():
        # Embedding: rows are cos/sin at k=3 only. W_L: output columns at k=7 only.
        model.W_E.zero_()
        model.W_E[:P, 0] = basis[2 * 3 - 1]
        model.W_E[:P, 1] = basis[2 * 3]
        model.W_out.zero_()
        model.W_out[0, 0] = 1.0
        model.W_U.zero_()
        model.W_U[0, :P] = basis[2 * 7]
    s = spectra(model, P)
    assert top_frequencies(s["W_E"], 1) == [3] and s["W_E"][3] == pytest.approx(1.0, abs=1e-5)
    assert top_frequencies(s["W_L"], 1) == [7] and s["W_L"][7] == pytest.approx(1.0, abs=1e-5)
    assert neuron_logit_map(model, P).shape == (P, cfg.d_mlp)


def test_permutation_test_null_and_signal():
    n = 56
    ranks = {k: k for k in range(1, n + 1)}  # frequency k has rank k
    # Keys at the top ranks in every seed: strong signal.
    top = rank_permutation_test([ranks] * 10, [[1, 2, 3, 4, 5]] * 10, n_draws=20000, seed=0)
    assert top["p_value"] < 1e-3
    assert top["null_mean_rank"] == pytest.approx((n + 1) / 2, abs=0.2)
    # Keys at the median ranks: no signal.
    mid = rank_permutation_test([ranks] * 10, [[26, 27, 28, 29, 30]] * 10, n_draws=20000, seed=0)
    assert 0.3 < mid["p_value"] < 0.7
    # Keys at the bottom: p near 1 (one-sided test).
    bottom = rank_permutation_test([ranks] * 10, [[52, 53, 54, 55, 56]] * 10, n_draws=20000, seed=0)
    assert bottom["p_value"] > 0.99
