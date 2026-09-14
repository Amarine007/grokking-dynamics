"""One-layer transformer for modular addition, written from scratch.

Architecture (Nanda et al. 2023): token + learned positional embeddings, one
block of causal multi-head attention followed by a ReLU MLP, then an unembedding.
There is deliberately NO LayerNorm, NO biases and NO dropout: they would make the
weight-space Fourier analysis in Stage 2 much harder to interpret, and the model
groks without them.

Weight shapes follow TransformerLens conventions (W_Q is [n_heads, d_model, d_head],
W_in is [d_model, d_mlp], ...) so that `to_hooked_transformer` is a direct copy.
Initialisation follows the reference implementation accompanying the paper: every
weight ~ N(0, 1/d_model) except W_U ~ N(0, 1/d_vocab).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    d_vocab: int  # input vocab: p numbers + "="
    d_vocab_out: int
    n_ctx: int
    d_model: int
    n_heads: int
    d_head: int
    d_mlp: int
    act_fn: str = "relu"
    n_layers: int = 1

    def __post_init__(self):
        if self.n_layers != 1:
            raise ValueError("only n_layers=1 is implemented")
        if self.act_fn != "relu":
            raise ValueError("only act_fn='relu' is implemented")

    def to_dict(self) -> dict:
        return asdict(self)


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg
        self.W_E = nn.Parameter(torch.empty(c.d_vocab, c.d_model))
        self.W_pos = nn.Parameter(torch.empty(c.n_ctx, c.d_model))
        self.W_Q = nn.Parameter(torch.empty(c.n_heads, c.d_model, c.d_head))
        self.W_K = nn.Parameter(torch.empty(c.n_heads, c.d_model, c.d_head))
        self.W_V = nn.Parameter(torch.empty(c.n_heads, c.d_model, c.d_head))
        self.W_O = nn.Parameter(torch.empty(c.n_heads, c.d_head, c.d_model))
        self.W_in = nn.Parameter(torch.empty(c.d_model, c.d_mlp))
        self.W_out = nn.Parameter(torch.empty(c.d_mlp, c.d_model))
        self.W_U = nn.Parameter(torch.empty(c.d_model, c.d_vocab_out))
        self.register_buffer("causal_mask", torch.tril(torch.ones(c.n_ctx, c.n_ctx, dtype=torch.bool)))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.cfg.d_model)
        for name, p in self.named_parameters():
            p.normal_(0.0, std)
        self.W_U.normal_(0.0, 1.0 / math.sqrt(self.cfg.d_vocab))

    def forward(self, tokens: torch.Tensor, last_only: bool = False) -> torch.Tensor:
        """Logits for `tokens` [batch, pos].

        Returns [batch, pos, d_vocab_out], or [batch, d_vocab_out] if `last_only`.
        `last_only` computes queries, the MLP and the unembedding only at the final
        position. With causal attention in a single layer this gives exactly the
        final-position logits of the full forward pass while skipping work that
        cannot affect them; it is used for training and evaluation.
        """
        b, n = tokens.shape
        h, e = self.cfg.n_heads, self.cfg.d_head
        # With the final query attending to every key, last_only needs no mask.
        q_tokens = tokens[:, -1:] if last_only else tokens
        q_pos = slice(n - 1, n) if last_only else slice(0, n)
        resid = F.embedding(q_tokens, self.W_E) + self.W_pos[q_pos]  # [b, nq, d]
        k = self._embed_project(tokens, slice(0, n), self.W_K)  # [b, n, h, e]
        v = self._embed_project(tokens, slice(0, n), self.W_V)
        q = self._embed_project(q_tokens, q_pos, self.W_Q)  # [b, nq, h, e]
        # Contractions over d_head and over the <=3 key positions are tiny, so they
        # are done as broadcasted products rather than thousands of small bmm calls.
        scores = (q[:, :, None] * k[:, None]).sum(-1) / math.sqrt(e)  # [b, nq, nk, h]
        if not last_only:
            scores = scores.masked_fill(~self.causal_mask[:n, :n, None], float("-inf"))
        pattern = scores.softmax(dim=2)
        z = (pattern[..., None] * v[:, None]).sum(2)  # [b, nq, h, e]
        resid = resid + z.reshape(b, -1, h * e) @ self.W_O.reshape(h * e, -1)
        resid = resid + F.relu(resid @ self.W_in) @ self.W_out
        logits = resid @ self.W_U
        return logits[:, -1] if last_only else logits

    def _embed_project(self, tokens: torch.Tensor, pos: slice, W: torch.Tensor) -> torch.Tensor:
        """(W_E[tokens] + W_pos[pos]) @ W, shaped [b, n, h, e].

        The block input is always one token embedding plus one positional embedding,
        so this is computed exactly as (W_E @ W)[tokens] + (W_pos @ W)[pos]: projecting
        the d_vocab + n_ctx embedding rows once instead of every example's residual.
        """
        h, d, e = W.shape
        W_flat = W.permute(1, 0, 2).reshape(d, h * e)
        out = F.embedding(tokens, self.W_E @ W_flat) + self.W_pos[pos] @ W_flat
        return out.reshape(*tokens.shape, h, e)


def model_config_for_p(p: int, model_cfg: dict) -> ModelConfig:
    """Build the ModelConfig from the YAML `model:` section for the mod-p task."""
    return ModelConfig(d_vocab=p + 1, d_vocab_out=p + 1, n_ctx=3, **model_cfg)


def to_hooked_transformer(model: Transformer):
    """Port trained weights into an equivalent TransformerLens HookedTransformer.

    All TransformerLens bias parameters are set to zero (the from-scratch model has
    none), and no LayerNorm is used, so the two models compute the same function.
    """
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    c = model.cfg
    device = next(model.parameters()).device
    tl_cfg = HookedTransformerConfig(
        n_layers=c.n_layers,
        d_model=c.d_model,
        d_head=c.d_head,
        n_heads=c.n_heads,
        d_mlp=c.d_mlp,
        d_vocab=c.d_vocab,
        d_vocab_out=c.d_vocab_out,
        n_ctx=c.n_ctx,
        act_fn=c.act_fn,
        normalization_type=None,
        attention_dir="causal",
        positional_embedding_type="standard",
        default_prepend_bos=False,
        device=str(device),
    )
    hooked = HookedTransformer(tl_cfg)

    ours = {
        "embed.W_E": model.W_E,
        "pos_embed.W_pos": model.W_pos,
        "blocks.0.attn.W_Q": model.W_Q,
        "blocks.0.attn.W_K": model.W_K,
        "blocks.0.attn.W_V": model.W_V,
        "blocks.0.attn.W_O": model.W_O,
        "blocks.0.mlp.W_in": model.W_in,
        "blocks.0.mlp.W_out": model.W_out,
        "unembed.W_U": model.W_U,
    }
    params = dict(hooked.named_parameters())
    unmapped = [n for n in params if n not in ours and not n.split(".")[-1].startswith("b_")]
    if unmapped:
        raise RuntimeError(f"HookedTransformer has non-bias parameters with no source: {unmapped}")
    with torch.no_grad():
        for name, p in params.items():
            if name in ours:
                if p.shape != ours[name].shape:
                    raise RuntimeError(f"shape mismatch for {name}: {tuple(p.shape)} vs {tuple(ours[name].shape)}")
                p.copy_(ours[name])
            else:
                p.zero_()
    hooked.eval()
    return hooked
