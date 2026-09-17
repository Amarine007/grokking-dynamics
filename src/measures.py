"""Progress measures through training (Stage 3): restricted and excluded loss.

Definitions follow Nanda et al. 2023, Section 5.1 (which translates the Section 4.4
logit ablations into measures), quoted here so the implementation can be checked
against the text rather than against memory:

    Restricted loss. ... we perform a 2D DFT on the logits to write them as a linear
    combination of waves in a and b, and set all terms besides the constant term and
    the 20 terms corresponding to cos(w_k(a+b)) and sin(w_k(a+b)) for the five key
    frequencies to 0. We then measure the loss of the ablated network.

    Excluded loss. Instead of keeping the important frequencies w_k, we next remove
    only those key frequencies from the logits but keep the rest. We measure this on
    the training data ...

The paper counts "20 terms" for five frequencies, i.e. four 2D components per
frequency, yet names only cos(w(a+b)) and sin(w(a+b)), which span just two directions
inside those four (cos a cos b - sin a sin b, and sin a cos b + cos a sin b). The other
two directions are cos/sin(w(a-b)). The text does not settle which is meant, so both
are implemented:

* **`sum`** (primary) -- only the cos/sin(w_k(a+b)) directions. This is what the words
  name, and what the algorithm predicts the logits use.
* **`products`** (sensitivity check) -- all four products at frequency k, so the
  a - b directions are kept (restricted) or removed (excluded) too.

Conventions:

* The DFT runs over the two *input* axes (a, b) of the logit grid, separately for each
  output class. All p + 1 output classes are kept, including "=", so the clean losses
  here are the same quantity as the Stage 1 CSV losses and can be checked against them.
* The "constant term" is the component constant in both a and b: a per-class bias.
* Key frequencies are fixed from the fully trained model and applied to every earlier
  checkpoint, as in the paper. A measure of how much an early model uses the *final*
  circuit is the point, not a flaw, but it does mean these measures cannot detect a
  circuit built on frequencies the final model abandoned.
"""

from __future__ import annotations

import math

import torch

from src.data import ModularAdditionData
from src.fourier import fourier_basis, sum_basis
from src.train import cross_entropy_high_precision

VARIANTS = ("sum", "products")


# --------------------------------------------------------------------------------
# Logit grid
# --------------------------------------------------------------------------------
def full_logit_grid(model, data: ModularAdditionData, batch_size: int = 4096) -> torch.Tensor:
    """Final-position logits for every (a, b) over ALL p + 1 outputs: [p, p, p + 1], float64."""
    p = data.p
    chunks = []
    for i in range(0, data.all_x.shape[0], batch_size):
        with torch.no_grad():
            chunks.append(model(data.all_x[i : i + batch_size], last_only=True).detach())
    return torch.cat(chunks).to(torch.float64).reshape(p, p, -1)


# --------------------------------------------------------------------------------
# The key-frequency subspace of the (a, b) grid
# --------------------------------------------------------------------------------
def key_directions(p: int, key: list[int], variant: str = "sum") -> torch.Tensor:
    """Orthonormal functions on the (a, b) grid spanning the key-frequency terms: [n, p, p].

    `sum`: cos/sin(w_k(a+b)), 2 per frequency. `products`: cos/sin(w_k a) x cos/sin(w_k b),
    4 per frequency (a superset: the `sum` directions lie in their span).
    """
    if variant == "sum":
        return sum_basis(p, key)[0]
    if variant == "products":
        basis, _ = fourier_basis(p)
        rows = []
        for k in key:
            if not 1 <= k <= p // 2:
                raise ValueError(f"frequency {k} out of range for p={p}")
            for r in (2 * k - 1, 2 * k):
                for s in (2 * k - 1, 2 * k):
                    rows.append(basis[r][:, None] * basis[s][None, :])
        return torch.stack(rows)
    raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


def constant_direction(p: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """The unit-norm function constant in both a and b: [1, p, p]."""
    return torch.full((1, p, p), 1.0 / p, dtype=dtype)


def project(grid: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection of a [p, p, c] grid onto orthonormal [n, p, p] directions."""
    coef = torch.einsum("nab,abc->nc", directions, grid)
    return torch.einsum("nab,nc->abc", directions, coef)


def restricted_logits(grid: torch.Tensor, key: list[int], variant: str = "sum") -> torch.Tensor:
    """Keep only the constant term and the key-frequency terms."""
    p = grid.shape[0]
    return project(grid, constant_direction(p, grid.dtype)) + project(grid, key_directions(p, key, variant))


def excluded_logits(grid: torch.Tensor, key: list[int], variant: str = "sum") -> torch.Tensor:
    """Remove only the key-frequency terms, keeping everything else (constant included)."""
    return grid - project(grid, key_directions(grid.shape[0], key, variant))


# --------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------
def score(grid: torch.Tensor, data: ModularAdditionData, split: str) -> tuple[float, float]:
    """(loss, accuracy) of a [p, p, c] logit grid on `split`: 'train', 'test' or 'all'."""
    flat = grid.reshape(-1, grid.shape[-1])
    if split == "all":
        idx = torch.arange(flat.shape[0])
    else:
        idx = getattr(data, f"{split}_idx")
    logits, y = flat[idx], data.all_y[idx]
    return cross_entropy_high_precision(logits, y).item(), (logits.argmax(-1) == y).double().mean().item()


def progress_measures(grid: torch.Tensor, data: ModularAdditionData, key: list[int]) -> dict[str, float]:
    """Every per-checkpoint number Stage 3 reports, from one [p, p, p + 1] logit grid.

    Clean train/test metrics, restricted loss/accuracy on train, test and all pairs, and
    excluded loss/accuracy on train (where the paper measures it) and test, for both
    variants. The `sum` variant's columns are unprefixed; `products` columns end `_products`.
    """
    out = {}
    for split in ("train", "test"):
        out[f"{split}_loss"], out[f"{split}_acc"] = score(grid, data, split)
    for variant in VARIANTS:
        tag = "" if variant == "sum" else f"_{variant}"
        restricted = restricted_logits(grid, key, variant)
        excluded = excluded_logits(grid, key, variant)
        for split in ("train", "test", "all"):
            out[f"restricted_loss_{split}{tag}"], out[f"restricted_acc_{split}{tag}"] = score(restricted, data, split)
        for split in ("train", "test"):
            out[f"excluded_loss_{split}{tag}"], out[f"excluded_acc_{split}{tag}"] = score(excluded, data, split)
    return out


# --------------------------------------------------------------------------------
# Phase boundaries (rules are declared in configs/progress.yaml, applied here)
# --------------------------------------------------------------------------------
def first_step(steps: list[int], values: list[float], below: float | None = None,
               at_least: float | None = None) -> int | None:
    """First step whose value is < `below` (or >= `at_least`); None if never."""
    if (below is None) == (at_least is None):
        raise ValueError("give exactly one of below / at_least")
    for s, v in zip(steps, values):
        if (below is not None and v < below) or (at_least is not None and v >= at_least):
            return s
    return None


def step_of_extreme(steps: list[int], values: list[float], kind: str) -> int:
    """Step of the minimum ('min') or maximum ('max') value; the earliest on ties."""
    if kind not in ("min", "max"):
        raise ValueError(f"kind must be 'min' or 'max', got {kind!r}")
    best = min(values) if kind == "min" else max(values)
    return steps[values.index(best)]


def phase_boundaries(rows: list[dict], p: int, rules: dict) -> dict[str, int | None]:
    """Apply the pre-registered boundary rules to one seed's per-checkpoint rows.

    `rows` are dicts with numeric `step` and the columns of `progress_measures`, sorted by
    step. Boundaries are checkpoint steps, so their resolution is the checkpoint spacing.

    * `memorization_end`: the step of minimum excluded (train) loss. The paper: excluded
      loss falls with train loss during memorization and rises during circuit formation.
    * `cleanup_start`: the first step whose test loss is below ln(p), the loss of a
      uniform guess over the p answers -- the full model starts beating chance on unseen
      pairs. The paper: cleanup is where "test loss suddenly drops".
    * `cleanup_end`: the first step with test accuracy >= `rules['cleanup_end_test_acc']`.
    * `restricted_peak`: the step of maximum restricted (test) loss, after which it falls.
    * `restricted_beats_uniform`: the first step whose restricted (test) loss is below
      ln(p): the key frequencies alone beat chance on unseen pairs.
    * `test_acc_jump`: the first step with test accuracy >= `rules['jump_test_acc']`, the
      visible jump the restricted-loss lead is measured against.
    """
    steps = [int(r["step"]) for r in rows]
    col = lambda name: [float(r[name]) for r in rows]  # noqa: E731
    uniform = math.log(p)
    restricted = col(f"restricted_loss_{rules['restricted_split']}")
    return {
        "memorization_end": step_of_extreme(steps, col("excluded_loss_train"), "min"),
        "cleanup_start": first_step(steps, col("test_loss"), below=uniform),
        "cleanup_end": first_step(steps, col("test_acc"), at_least=rules["cleanup_end_test_acc"]),
        "restricted_peak": step_of_extreme(steps, restricted, "max"),
        "restricted_beats_uniform": first_step(steps, restricted, below=uniform),
        "test_acc_jump": first_step(steps, col("test_acc"), at_least=rules["jump_test_acc"]),
    }
