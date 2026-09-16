"""The Fourier toolkit must be exact on signals whose answer is known analytically."""

import math

import pytest
import torch

from src.data import make_data
from src.fourier import (
    basis_frequencies,
    center_logits,
    embedding_spectrum,
    evaluate_logit_grid,
    fourier_2d,
    fourier_basis,
    frequency_masks,
    group_by_frequency,
    inverse_fourier_2d,
    key_frequencies,
    misclassified_pairs,
    neuron_frequency_fit,
    neuron_sum_fit,
    single_frequency_mask,
    sum_basis,
    variance_fraction,
)

P = 113
SMALL = 23  # a second odd p, so nothing silently depends on 113


# --- the basis ------------------------------------------------------------------
@pytest.mark.parametrize("p", [SMALL, P])
def test_basis_is_orthonormal(p):
    basis, names = fourier_basis(p)
    assert basis.shape == (p, p)
    assert len(names) == p and names[0] == "const"
    err = (basis @ basis.T - torch.eye(p, dtype=torch.float64)).abs().max().item()
    assert err < 1e-12, f"max |F F^T - I| = {err:.2e}"


def test_basis_rejects_even_p():
    with pytest.raises(ValueError):
        fourier_basis(8)


def test_basis_frequencies_pair_cos_and_sin():
    freqs = basis_frequencies(SMALL)
    assert freqs[0] == 0
    for k in range(1, SMALL // 2 + 1):
        assert freqs[2 * k - 1] == k and freqs[2 * k] == k


def test_group_by_frequency_sums_pairs():
    values = torch.ones(SMALL, dtype=torch.float64)
    grouped = group_by_frequency(values, SMALL)
    assert grouped[0] == 1.0  # the constant is alone
    assert (grouped[1:] == 2.0).all()  # every other frequency is a cos/sin pair


# --- the embedding spectrum -----------------------------------------------------
def test_spectrum_recovers_a_planted_frequency():
    """An embedding that is exactly cos/sin(w_7 x) must put all its mass at k = 7."""
    x = torch.arange(P, dtype=torch.float64)
    angle = 2 * math.pi * 7 * x / P
    W_E = torch.stack([torch.cos(angle), 3.0 * torch.sin(angle)], dim=1)
    spectrum = embedding_spectrum(W_E, P)
    assert spectrum.shape == (P // 2 + 1,)
    assert spectrum.sum().item() == pytest.approx(1.0)
    assert spectrum[7].item() == pytest.approx(1.0, abs=1e-12)


def test_spectrum_ignores_the_equals_row():
    """W_E has p + 1 rows; the "=" row must not enter the transform."""
    x = torch.arange(P, dtype=torch.float64)
    W_E = torch.cos(2 * math.pi * 7 * x / P)[:, None]
    with_eq = torch.cat([W_E, torch.full((1, 1), 99.0, dtype=torch.float64)])
    assert torch.allclose(embedding_spectrum(with_eq, P), embedding_spectrum(W_E, P))


def test_spectrum_of_a_constant_embedding_is_all_constant():
    W_E = torch.ones(P, 4, dtype=torch.float64)
    assert embedding_spectrum(W_E, P)[0].item() == pytest.approx(1.0)


def test_key_frequencies_threshold_and_order():
    spectrum = torch.zeros(P // 2 + 1, dtype=torch.float64)
    spectrum[0], spectrum[5], spectrum[9], spectrum[20] = 0.5, 0.3, 0.19, 0.01
    assert key_frequencies(spectrum, 0.05) == [5, 9]  # descending, constant excluded
    assert key_frequencies(spectrum, 0.005) == [5, 9, 20]
    assert key_frequencies(spectrum, 0.9) == []


# --- neurons as functions of (a + b) --------------------------------------------
def test_sum_basis_is_orthonormal_on_the_grid():
    basis, names = sum_basis(P, [5, 17])
    assert basis.shape == (4, P, P) and names == ["cos5", "sin5", "cos17", "sin17"]
    flat = basis.reshape(4, -1)
    err = (flat @ flat.T - torch.eye(4, dtype=torch.float64)).abs().max().item()
    assert err < 1e-12


def test_neuron_fit_recovers_a_planted_wave_exactly():
    """A neuron that IS cos(w_5(a+b) + phase) must come back with fve = 1."""
    p, k, phase = P, 5, 0.7
    s = (torch.arange(p)[:, None] + torch.arange(p)[None, :]) % p
    acts = torch.cos(2 * math.pi * k * s.double() / p + phase)[:, :, None]
    fit = neuron_sum_fit(acts, p, [k])
    assert fit["fve"][0].item() == pytest.approx(1.0, abs=1e-10)
    # atan2(sin_coef, cos_coef) of cos(t + phase) = cos(t)cos(phase) - sin(t)sin(phase)
    assert fit["phase"][0, 0].item() == pytest.approx(-phase, abs=1e-8)
    assert not fit["dead"][0]


def test_neuron_fit_offset_does_not_count_as_signal():
    """Adding a constant must not change the fraction of *variance* explained."""
    s = (torch.arange(P)[:, None] + torch.arange(P)[None, :]) % P
    wave = torch.cos(2 * math.pi * 5 * s.double() / P)[:, :, None]
    assert neuron_sum_fit(wave + 12.0, P, [5])["fve"][0].item() == pytest.approx(1.0, abs=1e-10)


def test_neuron_fit_splits_power_across_frequencies():
    s = (torch.arange(P)[:, None] + torch.arange(P)[None, :]) % P
    t = 2 * math.pi * s.double() / P
    acts = (torch.cos(5 * t) + 2.0 * torch.sin(17 * t))[:, :, None]
    fit = neuron_sum_fit(acts, P, [5, 17])
    assert fit["fve"][0].item() == pytest.approx(1.0, abs=1e-10)
    # power ratio 1 : 4, so the variance splits 0.2 / 0.8
    assert fit["fve_per_freq"][0, 0].item() == pytest.approx(0.2, abs=1e-10)
    assert fit["fve_per_freq"][1, 0].item() == pytest.approx(0.8, abs=1e-10)


def test_neuron_fit_rejects_a_function_of_a_minus_b():
    """cos(w(a - b)) is orthogonal to every function of a + b, so fve must be ~0."""
    d = (torch.arange(P)[:, None] - torch.arange(P)[None, :]) % P
    acts = torch.cos(2 * math.pi * 5 * d.double() / P)[:, :, None]
    assert neuron_sum_fit(acts, P, [5, 17, 24])["fve"][0].item() < 1e-10


def test_neuron_fit_marks_dead_neurons_instead_of_dividing_by_zero():
    acts = torch.zeros(P, P, 2, dtype=torch.float64)
    acts[:, :, 1] = 1.0  # constant, so also zero variance
    fit = neuron_sum_fit(acts, P, [5])
    assert fit["dead"].all()
    assert torch.isfinite(fit["fve"]).all() and (fit["fve"] == 0).all()


def test_product_neuron_is_single_frequency_but_only_half_a_sum():
    """cos(w a)cos(w b) pins both neuron fits at once.

    It is entirely built from frequency k, so the single-frequency fit must give 1.0.
    But cos(w a)cos(w b) = [cos(w(a+b)) + cos(w(a-b))] / 2, so exactly half its
    variance is a function of a + b and the sum fit must give 0.5. This is the
    textbook case the real neurons are being compared against.
    """
    k = 5
    a = torch.arange(P, dtype=torch.float64)[:, None]
    b = torch.arange(P, dtype=torch.float64)[None, :]
    w = 2 * math.pi * k / P
    acts = (torch.cos(w * a) * torch.cos(w * b))[:, :, None]
    assert neuron_frequency_fit(acts, P, [k])["best_fve"][0].item() == pytest.approx(1.0, abs=1e-10)
    assert neuron_sum_fit(acts, P, [k])["fve"][0].item() == pytest.approx(0.5, abs=1e-10)


def test_single_frequency_fit_picks_the_right_frequency():
    a = torch.arange(P, dtype=torch.float64)[:, None]
    b = torch.arange(P, dtype=torch.float64)[None, :]
    w = 2 * math.pi * 17 / P
    acts = (torch.cos(w * a) * torch.sin(w * b))[:, :, None]
    fit = neuron_frequency_fit(acts, P, [5, 17, 24])
    assert fit["best"][0].item() == 1  # index of 17 in the list
    assert fit["best_fve"][0].item() == pytest.approx(1.0, abs=1e-10)
    assert fit["fve"][0, 0].item() < 1e-10  # nothing at frequency 5


def test_single_frequency_mask_keeps_constant_and_k_only():
    mask = single_frequency_mask(P, 5)
    freqs = basis_frequencies(P)
    kept = torch.nonzero(mask)
    assert len(kept) == 9  # {const, cos5, sin5} on each axis
    for r, s in kept.tolist():
        assert freqs[r].item() in (0, 5) and freqs[s].item() in (0, 5)


# --- the 2D transform and ablation ----------------------------------------------
def test_fourier_2d_round_trips_and_obeys_parseval():
    grid = torch.randn(P, P, 4, dtype=torch.float64)
    coef = fourier_2d(grid, P)
    assert (inverse_fourier_2d(coef, P) - grid).abs().max().item() < 1e-10
    assert coef.pow(2).sum().item() == pytest.approx(grid.pow(2).sum().item())


def test_center_logits_removes_only_the_per_example_mean():
    grid = torch.randn(P, P, P, dtype=torch.float64)
    centred = center_logits(grid)
    assert centred.mean(-1).abs().max().item() < 1e-12
    # softmax-irrelevant: differences between answers are untouched
    assert ((grid[..., 1:] - grid[..., :1]) - (centred[..., 1:] - centred[..., :1])).abs().max() < 1e-10


def test_key_diag_mask_captures_a_planted_clock_signal():
    """Logits of the exact form cos(w_k(a + b - c)) must live entirely on the diagonal.

    This is the ablation's ground truth: if the model implements the clock algorithm,
    `key_diag` retains everything; if this test were wrong, a real model matching the
    hypothesis would look like a failure.
    """
    k = 5
    a = torch.arange(P)[:, None, None]
    b = torch.arange(P)[None, :, None]
    c = torch.arange(P)[None, None, :]
    grid = torch.cos(2 * math.pi * k * ((a + b - c) % P).double() / P)
    coef = fourier_2d(center_logits(grid), P)
    masks = frequency_masks(P, [k])
    assert variance_fraction(coef, masks["key_diag"]) == pytest.approx(1.0, abs=1e-10)
    assert variance_fraction(coef, masks["ablate_key"]) < 1e-10


def test_masks_have_the_expected_containment():
    masks = frequency_masks(P, [5, 17])
    assert (masks["key_diag"] & masks["key_block"]).equal(masks["key_diag"])
    # The only component a key-frequency reconstruction and a key-frequency ablation
    # share is the constant, which both keep on purpose: the ablation has to collapse
    # because the (a + b) structure is gone, not because a per-answer bias was removed.
    expected = torch.zeros(P, P, dtype=torch.bool)
    expected[0, 0] = True
    assert (masks["key_block"] & masks["ablate_key"]).equal(expected)
    assert masks["ablate_key"][0, 0], "the ablation must keep the constant component"


def test_variance_fraction_of_everything_is_one():
    coef = fourier_2d(torch.randn(P, P, 3, dtype=torch.float64), P)
    assert variance_fraction(coef, torch.ones(P, P, dtype=torch.bool)) == pytest.approx(1.0)
    assert variance_fraction(coef, torch.zeros(P, P, dtype=torch.bool)) == 0.0


# --- evaluation helpers ---------------------------------------------------------
def test_evaluate_and_misclassified_agree_on_a_perfect_grid():
    data = make_data(P, 0.3, 0)
    onehot = torch.zeros(P, P, P, dtype=torch.float64)
    s = (torch.arange(P)[:, None] + torch.arange(P)[None, :]) % P
    onehot.scatter_(2, s[:, :, None], 10.0)  # argmax is always the right answer
    metrics = evaluate_logit_grid(onehot, data)
    assert metrics["train_acc"] == 1.0 and metrics["test_acc"] == 1.0
    assert misclassified_pairs(onehot, data) == []


def test_misclassified_pairs_reports_the_planted_error():
    data = make_data(P, 0.3, 0)
    s = (torch.arange(P)[:, None] + torch.arange(P)[None, :]) % P
    grid = torch.zeros(P, P, P, dtype=torch.float64).scatter_(2, s[:, :, None], 10.0)
    flat_idx = data.test_idx[0].item()
    a, b = flat_idx // P, flat_idx % P
    grid[a, b] = 0.0
    grid[a, b, (a + b + 1) % P] = 99.0  # force one specific wrong answer
    wrong = misclassified_pairs(grid, data)
    assert wrong == [{"a": a, "b": b, "answer": (a + b) % P, "predicted": (a + b + 1) % P}]
