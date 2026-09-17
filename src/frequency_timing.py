"""When does a seed commit to its key frequencies? (Stage 4)

Stage 2 found that every seed uses a different set of 4-5 key frequencies; Stage 3 found
that the final set already carries useful logit signal a few hundred steps into
training. This module tracks the frequency content of the weights through training to
ask when that set is decided: latent in the initialization, chosen early, or settled
only after other frequencies compete.

Two weight matrices are tracked, both as a spectrum over the p number tokens:

* `W_E[:p]` -- the embedding, the input side. The primary matrix, and the one Stage 2
  used to define the key frequencies.
* `W_L = W_out @ W_U[:, :p]` -- the neuron-logit map, the output side. The paper uses it
  to find key frequencies on its additional seeds.

Everything is a pure function of a checkpoint or of spectra computed from checkpoints.
Ranks are 1-based with 1 = the frequency carrying the most squared norm; the constant
term is never ranked.
"""

from __future__ import annotations

import numpy as np
import torch

from src.fourier import embedding_spectrum


def neuron_logit_map(model, p: int) -> torch.Tensor:
    """W_L restricted to the number outputs, one row per output token: [p, d_mlp]."""
    return (model.W_out @ model.W_U[:, :p]).T.detach()


def spectra(model, p: int) -> dict[str, torch.Tensor]:
    """Per-frequency share of squared norm (index 0 = constant) for W_E and W_L."""
    return {"W_E": embedding_spectrum(model.W_E, p), "W_L": embedding_spectrum(neuron_logit_map(model, p), p)}


def frequency_ranks(spectrum: torch.Tensor | np.ndarray) -> dict[int, int]:
    """Rank of every non-constant frequency: {k: rank}, rank 1 = largest share, ties by lower k."""
    values = np.asarray(spectrum, dtype=np.float64)[1:]
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    return {i + 1: r + 1 for r, i in enumerate(order)}


def top_frequencies(spectrum: torch.Tensor | np.ndarray, n: int) -> list[int]:
    """The n non-constant frequencies with the largest share, sorted ascending."""
    ranks = frequency_ranks(spectrum)
    return sorted(k for k, r in ranks.items() if r <= n)


def settled_step(steps: list[int], ok: list[bool]) -> int | None:
    """First step from which `ok` holds at every later step; None if it fails at the last step."""
    if not ok or not ok[-1]:
        return None
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    return steps[i]


def rank_permutation_test(rank_maps: list[dict[int, int]], key_sets: list[list[int]],
                          n_draws: int, seed: int) -> dict[str, float]:
    """Are the eventual key frequencies already ranked high at this checkpoint?

    Null hypothesis: the key set is unrelated to the current ranking, so within each seed
    it is a uniformly random subset of the same size. The statistic is the mean rank of
    all key frequencies pooled over seeds (lower = more norm). One-sided p-value:
    P(null mean rank <= observed), with the +1 correction so it is never exactly 0.
    """
    rng = np.random.default_rng(seed)
    observed = np.mean([ranks[k] for ranks, key in zip(rank_maps, key_sets) for k in key])
    n_freq = len(rank_maps[0])
    total_keys = sum(len(k) for k in key_sets)
    null_sum = np.zeros(n_draws)
    for ranks, key in zip(rank_maps, key_sets):
        values = np.array([ranks[k] for k in range(1, n_freq + 1)], dtype=np.float64)
        # A uniformly random subset per draw: argsort of iid uniforms, keep the first len(key).
        picks = np.argsort(rng.random((n_draws, n_freq)), axis=1)[:, : len(key)]
        null_sum += values[picks].sum(axis=1)
    null_mean = null_sum / total_keys
    return {
        "observed_mean_rank": float(observed),
        "null_mean_rank": float(null_mean.mean()),
        "null_sd": float(null_mean.std()),
        "p_value": float((1 + np.sum(null_mean <= observed)) / (1 + n_draws)),
        "n_frequencies": int(total_keys),
        "n_draws": int(n_draws),
    }
