"""The TransformerLens port must reproduce the from-scratch model's logits to 1e-4."""

from pathlib import Path

import pytest
import torch

from src.data import make_cayley_table
from src.model import Transformer, model_config_for_p, to_hooked_transformer
from src.train import REPO_ROOT, load_checkpoint, load_config

P = 113
TOL = 1e-4
BASELINE = REPO_ROOT / "configs" / "baseline.yaml"


def _model(seed: int, scale: float = 1.0) -> Transformer:
    torch.manual_seed(seed)
    m = Transformer(model_config_for_p(P, load_config(BASELINE)["model"]))
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(scale)
    return m.eval()


def _assert_port_matches(model: Transformer) -> float:
    x, _ = make_cayley_table(P)
    hooked = to_hooked_transformer(model)
    with torch.no_grad():
        ours = model(x)
        theirs = hooked(x)
    assert ours.shape == theirs.shape == (P * P, 3, P + 1)
    err = (ours - theirs).abs().max().item()
    assert err < TOL, f"max |logit diff| = {err:.2e}"
    return err


def test_port_random_init():
    _assert_port_matches(_model(0))


def test_port_large_weights():
    # Scaled-up weights give sharper attention and logits of magnitude ~10, closer to a
    # trained model. (Much larger scales make |logits| ~ 1e3, where float32 rounding
    # alone exceeds an absolute 1e-4 bound for either implementation.)
    _assert_port_matches(_model(1, scale=1.5))


def test_port_has_no_active_biases():
    hooked = to_hooked_transformer(_model(0))
    for name, p in hooked.named_parameters():
        if name.split(".")[-1].startswith("b_"):
            assert torch.count_nonzero(p) == 0, name


def test_last_only_matches_full_forward():
    m = _model(2, scale=1.5)
    x, _ = make_cayley_table(P)
    with torch.no_grad():
        err = (m(x)[:, -1] - m(x, last_only=True)).abs().max().item()
    assert err < TOL


def test_hooks_available():
    x, _ = make_cayley_table(P)
    _, cache = to_hooked_transformer(_model(0)).run_with_cache(x[:64])
    assert cache["blocks.0.mlp.hook_post"].shape == (64, 3, 512)
    assert cache["blocks.0.attn.hook_pattern"].shape == (64, 4, 3, 3)


def _trained_checkpoints() -> list[Path]:
    root = REPO_ROOT / "checkpoints" / "01_baseline"
    return sorted(root.glob("baseline_seed*/step040000.pt"))[:2]


@pytest.mark.skipif(not _trained_checkpoints(), reason="no trained checkpoints on disk")
@pytest.mark.parametrize("path", _trained_checkpoints(), ids=lambda p: p.parent.name)
def test_port_trained_checkpoint(path):
    model, _ = load_checkpoint(path)
    _assert_port_matches(model)
