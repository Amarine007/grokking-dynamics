"""Fourier analysis of a trained modular-addition transformer (Stage 2).

The hypothesis under test is the "clock" algorithm of Nanda et al. 2023: the model
embeds each number on a small set of circles, embedding a as cos/sin(w_k a) for a
handful of frequencies w_k = 2*pi*k/p, multiplies the a-terms and b-terms together
in attention and the MLP, and reads the answer off as cos(w_k(a + b - c)), which is
maximal exactly when c = a + b (mod p).

Everything here is a pure function of a model or of tensors it produces, computed in
float64 on CPU from a checkpoint. No GPU, no training, no optimizer state.

Three conventions, fixed here and relied on everywhere downstream:

* **The basis is orthonormal**, so Parseval holds exactly: the sum of squared
  coefficients equals the sum of squares of the original tensor. Every "fraction of
  variance" below is therefore an exact ratio of squared norms, not a regression R^2
  that happens to approximate one.
* **Only the p number tokens are analysed.** The vocabulary is p + 1 because "=" is a
  token, but a (p+1)-point transform over a cyclic group of order p is meaningless, so
  `W_E[:p]` and `logits[..., :p]` are used throughout.
* **Logits are centred over the answer axis** before analysis. Softmax is invariant to
  adding a constant to every logit of an example, so that component cannot affect a
  prediction and would otherwise inflate the "retained variance" numbers.
"""

from __future__ import annotations

import math

import torch

from src.data import ModularAdditionData, make_cayley_table
from src.train import cross_entropy_high_precision

HOOK_MLP_POST = "blocks.0.mlp.hook_post"

# Squared norms below this are treated as zero (e.g. a permanently dead ReLU neuron,
# whose "fraction of variance explained" is 0/0 and is reported as such, not as 1.0).
_TINY = 1e-12


# --------------------------------------------------------------------------------
# The basis
# --------------------------------------------------------------------------------
def fourier_basis(p: int, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, list[str]]:
    """Orthonormal Fourier basis of R^p: the constant, then cos/sin at k = 1..(p-1)/2.

    Returns `(F, names)` with F of shape [p, p], one basis vector per row, so that
    `F @ v` gives the coefficients of v and `F.T @ coef` reconstructs it. p must be odd
    (113 is), so there is no Nyquist row and every frequency has both a cos and a sin.
    """
    if p % 2 == 0:
        raise ValueError(f"p must be odd for this basis (cos/sin pairs); got {p}")
    x = torch.arange(p, dtype=dtype)
    rows = [torch.ones(p, dtype=dtype)]
    names = ["const"]
    for k in range(1, p // 2 + 1):
        angle = 2 * math.pi * k * x / p
        rows += [torch.cos(angle), torch.sin(angle)]
        names += [f"cos{k}", f"sin{k}"]
    basis = torch.stack(rows)
    basis = basis / basis.norm(dim=1, keepdim=True)
    return basis, names


def basis_frequencies(p: int) -> torch.Tensor:
    """The frequency k of each basis row: 0 for the constant, then k, k for cos_k, sin_k."""
    freqs = torch.zeros(p, dtype=torch.long)
    for k in range(1, p // 2 + 1):
        freqs[2 * k - 1] = k
        freqs[2 * k] = k
    return freqs


def group_by_frequency(values: torch.Tensor, p: int) -> torch.Tensor:
    """Sum a per-basis-row quantity into per-frequency totals; index 0 is the constant."""
    out = torch.zeros(p // 2 + 1, dtype=values.dtype)
    return out.index_add_(0, basis_frequencies(p), values)


# --------------------------------------------------------------------------------
# 1. The embedding spectrum
# --------------------------------------------------------------------------------
def embedding_spectrum(W_E: torch.Tensor, p: int) -> torch.Tensor:
    """Fraction of `W_E[:p]`'s squared Frobenius norm carried by each frequency.

    Returns a vector of length p//2 + 1 summing to 1, where index 0 is the constant
    (mean) component and index k is the cos_k and sin_k pair combined.
    """
    basis, _ = fourier_basis(p)
    coef = basis @ W_E[:p].detach().to(basis.dtype)
    per_freq = group_by_frequency(coef.pow(2).sum(1), p)
    return per_freq / per_freq.sum()


def key_frequencies(spectrum: torch.Tensor, threshold: float) -> list[int]:
    """Frequencies carrying at least `threshold` of the squared norm, largest first.

    The constant term is never a key frequency. The threshold is a declared parameter,
    not a fitted one: the full spectrum is always reported alongside so the cutoff and
    the gap around it are visible.
    """
    order = torch.argsort(spectrum, descending=True).tolist()
    return [k for k in order if k > 0 and spectrum[k].item() >= threshold]


# --------------------------------------------------------------------------------
# 2. MLP neurons as functions of (a + b)
# --------------------------------------------------------------------------------
def mlp_activations(hooked, p: int, batch_size: int = 4096) -> torch.Tensor:
    """Post-ReLU MLP activations at the final position for every (a, b): [p, p, d_mlp].

    Batched with only the one hook cached, so peak memory stays a few hundred MB. The
    model has no cross-example operations, so batching is numerically irrelevant.
    """
    x, _ = make_cayley_table(p)
    chunks = []
    for i in range(0, x.shape[0], batch_size):
        with torch.no_grad():
            _, cache = hooked.run_with_cache(
                x[i : i + batch_size], names_filter=lambda n: n == HOOK_MLP_POST
            )
        chunks.append(cache[HOOK_MLP_POST][:, -1].detach())
    return torch.cat(chunks).reshape(p, p, -1)


def sum_basis(p: int, freqs: list[int], dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, list[str]]:
    """cos/sin(w_k (a + b)) over the (a, b) grid, orthonormal: [2 * len(freqs), p, p].

    These are the functions the clock algorithm predicts the MLP computes. They are
    mutually orthogonal on the grid because (a + b) mod p takes each value exactly p
    times, which is what makes the projection below an exact variance decomposition.
    """
    basis, _ = fourier_basis(p, dtype)
    s = (torch.arange(p)[:, None] + torch.arange(p)[None, :]) % p
    rows, names = [], []
    for k in freqs:
        if not 1 <= k <= p // 2:
            raise ValueError(f"frequency {k} out of range for p={p}")
        rows += [basis[2 * k - 1][s] / math.sqrt(p), basis[2 * k][s] / math.sqrt(p)]
        names += [f"cos{k}", f"sin{k}"]
    return torch.stack(rows), names


def neuron_sum_fit(acts: torch.Tensor, p: int, freqs: list[int]) -> dict[str, torch.Tensor]:
    """Decompose each neuron's activation over (a, b) onto cos/sin(w_k(a + b)).

    `acts` is [p, p, n_neurons]. Activations are centred first, so the returned
    fractions are of *variance* and the neuron's mean is not counted as signal.

    Returns per neuron: `fve` (fraction of variance explained by all `freqs` jointly),
    `fve_per_freq` [len(freqs), n], `phase` [len(freqs), n] in radians, `amplitude`
    [len(freqs), n], and `dead` (a boolean mask of neurons with no variance at all).

    A high `fve` means the neuron is essentially a function of a + b alone, which is
    the strong form of the clock prediction; the a - b terms that individual neurons
    also carry are expected to show up as unexplained variance here and to cancel only
    across the population.
    """
    if acts.ndim != 3 or acts.shape[0] != p or acts.shape[1] != p:
        raise ValueError(f"expected acts of shape [{p}, {p}, n], got {tuple(acts.shape)}")
    basis, _ = sum_basis(p, freqs)
    centred = acts.to(torch.float64)
    centred = centred - centred.mean(dim=(0, 1), keepdim=True)
    total = centred.pow(2).sum(dim=(0, 1))
    coef = torch.einsum("fab,abn->fn", basis, centred)
    dead = total < _TINY
    denom = total.clamp_min(_TINY)
    cos_c, sin_c = coef[0::2], coef[1::2]
    power = cos_c.pow(2) + sin_c.pow(2)
    fve_per_freq = power / denom
    return {
        "fve": torch.where(dead, torch.zeros_like(total), fve_per_freq.sum(0)),
        "fve_per_freq": torch.where(dead, torch.zeros_like(fve_per_freq), fve_per_freq),
        "phase": torch.atan2(sin_c, cos_c),
        "amplitude": power.sqrt(),
        "dead": dead,
    }


def single_frequency_mask(p: int, k: int) -> torch.Tensor:
    """2D components whose two axis frequencies are each either the constant or k."""
    freqs = basis_frequencies(p)
    keep = (freqs == 0) | (freqs == k)
    return keep[:, None] & keep[None, :]


def neuron_frequency_fit(acts: torch.Tensor, p: int, freqs: list[int]) -> dict[str, torch.Tensor]:
    """Fraction of each neuron's variance that lives at a *single* frequency.

    `neuron_sum_fit` asks the strong question: is this neuron a function of a + b? This
    asks the weaker one the paper actually claims: is it built from one frequency,
    whatever it does with that frequency? Every 2D component with frequency 0 or k on
    each axis is kept, so terms in a alone, b alone, a + b and a - b all count.

    The gap between the two is the point. A ReLU neuron computing cos(w a)cos(w b) is
    entirely single-frequency, yet only half its variance is a function of a + b,
    because cos(w a)cos(w b) = [cos(w(a+b)) + cos(w(a-b))] / 2. The a - b half is real
    structure that cancels across the population rather than inside any one neuron.

    (Uses the 2D transform defined in section 3 below.)

    Returns `fve` [len(freqs), n_neurons], plus `best` (index into `freqs`) and
    `best_fve` per neuron.
    """
    centred = acts.to(torch.float64)
    centred = centred - centred.mean(dim=(0, 1), keepdim=True)
    coef = fourier_2d(centred, p)
    total = coef.pow(2).sum(dim=(0, 1)).clamp_min(_TINY)
    fve = torch.stack(
        [
            (coef * single_frequency_mask(p, k)[:, :, None]).pow(2).sum(dim=(0, 1)) / total
            for k in freqs
        ]
    )
    best = fve.argmax(0)
    return {"fve": fve, "best": best, "best_fve": fve.gather(0, best[None]).squeeze(0)}


# --------------------------------------------------------------------------------
# 3. Fourier-space ablation of the logits
# --------------------------------------------------------------------------------
def logit_grid(model, p: int, batch_size: int = 4096) -> torch.Tensor:
    """Final-position logits over the p numeric answers for every (a, b): [p, p, p].

    The "=" output column is dropped. That never changes a prediction on a trained
    model (its argmax is always a number), and `experiments/02_fourier.py` asserts so
    before relying on it.
    """
    x, _ = make_cayley_table(p)
    chunks = []
    for i in range(0, x.shape[0], batch_size):
        with torch.no_grad():
            chunks.append(model(x[i : i + batch_size], last_only=True)[:, :p].detach())
    return torch.cat(chunks).reshape(p, p, p).to(torch.float64)


def center_logits(logits: torch.Tensor) -> torch.Tensor:
    """Remove the per-example mean over the answer axis, which softmax ignores anyway."""
    return logits - logits.mean(-1, keepdim=True)


def fourier_2d(grid: torch.Tensor, p: int) -> torch.Tensor:
    """Transform a [p, p, c] grid over its two input axes into Fourier space."""
    basis, _ = fourier_basis(p, grid.dtype)
    return torch.einsum("ra,sb,abc->rsc", basis, basis, grid)


def inverse_fourier_2d(coef: torch.Tensor, p: int) -> torch.Tensor:
    """Invert `fourier_2d`."""
    basis, _ = fourier_basis(p, coef.dtype)
    return torch.einsum("ra,sb,rsc->abc", basis, basis, coef)


def frequency_masks(p: int, key: list[int]) -> dict[str, torch.Tensor]:
    """Boolean [p, p] masks over 2D Fourier components, keyed by what they keep.

    * `key_diag` -- only components at the *same* key frequency on both axes, plus the
      constant. This is what the clock algorithm predicts carries the whole
      computation, since cos(w_k(a + b - c)) expands into products of a-terms and
      b-terms at one shared frequency k. It is the sharpest test of the hypothesis.
    * `key_block` -- any component whose two axis frequencies are both key (or
      constant). A superset of `key_diag` that also allows cross-frequency terms.
    * `ablate_key` -- the opposite: everything *not* touching a key frequency on either
      axis. If the key frequencies carry the algorithm, this should destroy it.

    `key_block` and `ablate_key` are not complements: components with a key frequency
    on one axis and a non-key frequency on the other are in neither, and
    `experiments/02_fourier.py` reports how much mass falls in that gap.
    """
    freqs = basis_frequencies(p)
    is_const = freqs == 0
    is_key = torch.zeros(p, dtype=torch.bool)
    for k in key:
        is_key |= freqs == k

    diag = is_const[:, None] & is_const[None, :]
    for k in key:
        row = freqs == k
        diag = diag | (row[:, None] & row[None, :])
    keep = is_key | is_const
    return {
        "key_diag": diag,
        "key_block": keep[:, None] & keep[None, :],
        "ablate_key": (~is_key)[:, None] & (~is_key)[None, :],
    }


def apply_mask(coef: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero every 2D Fourier component outside `mask`."""
    return coef * mask[:, :, None].to(coef.dtype)


def variance_fraction(coef: torch.Tensor, mask: torch.Tensor) -> float:
    """Fraction of total squared norm inside `mask`, exact by Parseval."""
    total = coef.pow(2).sum()
    if total < _TINY:
        return 0.0
    return (apply_mask(coef, mask).pow(2).sum() / total).item()


def evaluate_logit_grid(grid: torch.Tensor, data: ModularAdditionData) -> dict[str, float]:
    """Train/test loss and accuracy for a [p, p, p] logit grid.

    Losses are over p classes, not p + 1, so they are comparable across ablations but
    sit slightly below the p + 1 losses in the Stage 1 CSVs (the "=" logit contributes
    to the softmax denominator there). The clean row is computed the same way.
    """
    flat = grid.reshape(-1, grid.shape[-1])
    out = {}
    for split in ("train", "test"):
        idx = getattr(data, f"{split}_idx")
        y = getattr(data, f"{split}_y")
        logits = flat[idx]
        out[f"{split}_loss"] = cross_entropy_high_precision(logits, y).item()
        out[f"{split}_acc"] = (logits.argmax(-1) == y).double().mean().item()
    return out


def misclassified_pairs(grid: torch.Tensor, data: ModularAdditionData, split: str = "test") -> list[dict]:
    """Every (a, b) in `split` the logit grid gets wrong, with what it predicted."""
    p = grid.shape[0]
    flat = grid.reshape(-1, grid.shape[-1])
    idx = getattr(data, f"{split}_idx")
    y = getattr(data, f"{split}_y")
    pred = flat[idx].argmax(-1)
    wrong = (pred != y).nonzero().flatten()
    rows = []
    for i in wrong.tolist():
        flat_idx = idx[i].item()
        rows.append(
            {
                "a": flat_idx // p,
                "b": flat_idx % p,
                "answer": y[i].item(),
                "predicted": pred[i].item(),
            }
        )
    return rows
