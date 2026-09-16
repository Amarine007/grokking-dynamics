"""Is attention part of the algorithm, or just a fixed router? (Stage 2, minimal.)

Stage 2 showed the circuit runs embedding -> MLP -> logits on a sparse set of
frequencies, but never asked what attention contributes. This module answers the
narrow version of that question.

There is a structural reason it is cheap here. The final position always holds the
same token ("="), and the model has **no LayerNorm**, so the residual stream at the
query position before attention is W_E[p] + W_pos[2] -- identical for every input.
The query vector is therefore a constant, and the three attention scores at the final
position are:

    score(pos 0) = q . k(a, pos 0)   a function of a alone
    score(pos 1) = q . k(b, pos 1)   a function of b alone
    score(pos 2) = q . k(=, pos 2)   a constant

So attention is fully described by two 1-D functions of p values plus a scalar, per
head. `query_spread` and `factorisation_error` check that empirically rather than
trusting the algebra, and everything else here measures what happens when attention is
prevented from varying with the input.
"""

from __future__ import annotations

import torch

from src.data import make_cayley_table
from src.fourier import center_logits

HOOK_PATTERN = "blocks.0.attn.hook_pattern"
HOOK_Q = "blocks.0.attn.hook_q"
HOOK_Z = "blocks.0.attn.hook_z"

_TINY = 1e-12


def _cache_final(hooked, p: int, name: str, batch_size: int) -> torch.Tensor:
    """Cache one hook at the final position over every (a, b): [p * p, ...]."""
    x, _ = make_cayley_table(p)
    chunks = []
    for i in range(0, x.shape[0], batch_size):
        with torch.no_grad():
            _, cache = hooked.run_with_cache(
                x[i : i + batch_size], names_filter=lambda n, want=name: n == want
            )
        value = cache[name]
        # hook_pattern is [batch, head, query, key]; q and z are [batch, pos, head, ...].
        chunks.append((value[:, :, -1] if name == HOOK_PATTERN else value[:, -1]).detach())
    return torch.cat(chunks)


def final_patterns(hooked, p: int, batch_size: int = 4096) -> torch.Tensor:
    """Attention paid by the final position to (a, b, =), per head: [p, p, n_heads, 3]."""
    flat = _cache_final(hooked, p, HOOK_PATTERN, batch_size)
    return flat.reshape(p, p, flat.shape[1], flat.shape[2])


def query_spread(hooked, p: int, batch_size: int = 4096) -> float:
    """Largest standard deviation of any final-position query component over all inputs.

    Should be 0 to floating-point noise: the query cannot depend on a or b. A non-zero
    value would mean the reasoning in this module's docstring is wrong.
    """
    q = _cache_final(hooked, p, HOOK_Q, batch_size)  # [p*p, n_heads, d_head]
    return q.std(dim=0).max().item()


def factorisation_error(patterns: torch.Tensor) -> tuple[float, float]:
    """How much the a-score depends on b, and the b-score depends on a.

    Working in log space, log(alpha_j / alpha_2) equals score_j - score_2, which the
    structure above says is a function of a alone (j = 0) or b alone (j = 1). Each is
    returned as the largest deviation from its own row/column mean, divided by that
    quantity's overall spread, so 0 means perfect factorisation and 1 means none.
    """
    logs = patterns.clamp_min(_TINY).log().to(torch.float64)
    gap_a = logs[..., 0] - logs[..., 2]  # [p, p, head]; should not vary along b (dim 1)
    gap_b = logs[..., 1] - logs[..., 2]  # should not vary along a (dim 0)
    out = []
    for gap, dim in ((gap_a, 1), (gap_b, 0)):
        deviation = (gap - gap.mean(dim=dim, keepdim=True)).abs().max().item()
        spread = (gap.max() - gap.min()).clamp_min(_TINY).item()
        out.append(deviation / spread)
    return out[0], out[1]


def mean_pattern(patterns: torch.Tensor) -> torch.Tensor:
    """The average attention pattern over all inputs: [n_heads, 3]."""
    return patterns.mean(dim=(0, 1))


def head_mean_z(hooked, p: int, batch_size: int = 4096) -> torch.Tensor:
    """Each head's average final-position output vector: [n_heads, d_head]."""
    return _cache_final(hooked, p, HOOK_Z, batch_size).mean(0)


def hooked_logit_grid(hooked, p: int, fwd_hooks: list, batch_size: int = 4096) -> torch.Tensor:
    """Final-position logits over the p numeric answers, under `fwd_hooks`: [p, p, p]."""
    x, _ = make_cayley_table(p)
    chunks = []
    for i in range(0, x.shape[0], batch_size):
        with torch.no_grad():
            logits = hooked.run_with_hooks(x[i : i + batch_size], fwd_hooks=fwd_hooks)
        chunks.append(logits[:, -1, :p].detach())
    return torch.cat(chunks).reshape(p, p, p).to(torch.float64)


def freeze_pattern_hook(pattern_mean: torch.Tensor):
    """Replace the final position's attention with a fixed pattern, ignoring the input.

    If the model still works under this, attention carries no information about a or b
    and is only a router; if it collapses, attention is part of the computation.
    """

    def hook(pattern, hook):  # pattern: [batch, head, query, key]
        pattern[:, :, -1, :] = pattern_mean.to(pattern.dtype)
        return pattern

    return [(HOOK_PATTERN, hook)]


def ablate_head_hook(head: int, z_mean: torch.Tensor):
    """Replace one head's final-position output with its average over all inputs.

    Mean-ablation rather than zeroing: it removes the head's input-dependent signal
    while leaving its average contribution in place, so any drop is due to lost
    information and not to a missing bias.
    """

    def hook(z, hook):  # z: [batch, pos, head, d_head]
        z[:, -1, head] = z_mean[head].to(z.dtype)
        return z

    return [(HOOK_Z, hook)]


def variance_retained(clean: torch.Tensor, other: torch.Tensor) -> float:
    """Fraction of the clean model's centred logit variance that `other` reproduces.

    1.0 is identical; 0.0 is no better than predicting the clean mean. Negative values
    are possible and are reported as they come, not clipped.
    """
    c, o = center_logits(clean), center_logits(other)
    total = c.pow(2).sum()
    if total < _TINY:
        return 0.0
    return (1.0 - (c - o).pow(2).sum() / total).item()
