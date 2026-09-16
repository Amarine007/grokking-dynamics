"""Same seed -> byte-identical CSV; different seed -> different run."""

import copy

import torch

from src.train import (
    REPO_ROOT,
    CSV_COLUMNS,
    checkpoint_steps,
    load_checkpoint,
    load_config,
    log_checkpoint_steps,
    parse_seeds,
    train,
)

BASELINE = load_config(REPO_ROOT / "configs" / "baseline.yaml")
# Test determinism on the GPU when there is one (that is where results are produced).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _short(cfg: dict, num_steps: int) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["train"].update(num_steps=num_steps, log_every=50, num_checkpoints=4)
    # The baseline's dense schedule targets the real 40k run; tests that want dense
    # checkpoints opt in explicitly, so short runs exercise the log-spaced path alone.
    cfg["train"].pop("checkpoint_every", None)
    cfg["train"].pop("checkpoint_every_until", None)
    cfg["runtime"]["device"] = DEVICE
    return cfg


def _tiny() -> dict:
    cfg = _short(BASELINE, 300)
    cfg["data"]["p"] = 23
    cfg["model"].update(d_model=32, d_head=8, d_mlp=64)
    return cfg


def test_same_seed_byte_identical_tiny(tmp_path):
    cfg = _tiny()
    a = train(cfg, 0, tmp_path / "a.csv", tmp_path / "ck_a", verbose=False).read_bytes()
    b = train(cfg, 0, tmp_path / "b.csv", None, verbose=False).read_bytes()
    c = train(cfg, 1, tmp_path / "c.csv", None, verbose=False).read_bytes()
    assert a == b
    assert a != c


def test_same_seed_byte_identical_baseline_config(tmp_path):
    """The real model and data, truncated to 200 steps."""
    cfg = _short(BASELINE, 200)
    a = train(cfg, 3, tmp_path / "a.csv", None, verbose=False).read_bytes()
    b = train(cfg, 3, tmp_path / "b.csv", None, verbose=False).read_bytes()
    assert a == b


def test_csv_layout_and_checkpoints(tmp_path):
    cfg = _tiny()
    path = train(cfg, 0, tmp_path / "run.csv", tmp_path / "ck", verbose=False)
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(CSV_COLUMNS)
    steps = [int(line.split(",")[0]) for line in lines[1:]]
    assert steps == list(range(0, 301, 50))

    ckpts = sorted((tmp_path / "ck").glob("step*.pt"))
    assert [int(p.stem[4:]) for p in ckpts] == checkpoint_steps(300, 4)
    ck = torch.load(ckpts[-1], weights_only=False)
    assert ck["step"] == 300 and ck["seed"] == 0
    assert ck["optimizer"]["state"], "optimizer state must be checkpointed"


def test_checkpoint_schedule():
    steps = checkpoint_steps(40000, 20)
    assert steps[0] == 0 and steps[-1] == 40000
    assert 19 <= len(steps) <= 21


def test_dense_checkpoint_schedule():
    steps = checkpoint_steps(40000, 20, every=250, every_until=15000)
    assert steps[0] == 0 and steps[-1] == 40000
    dense, log = set(range(0, 15001, 250)), set(log_checkpoint_steps(40000, 20))
    assert dense <= set(steps), "every 250 steps up to 15000 must be present"
    assert log <= set(steps), "the log-spaced schedule must be kept"
    # nothing dense is added past the limit
    assert {s for s in steps if s > 15000} == {s for s in log if s > 15000}


def test_dense_checkpoints_are_weights_only(tmp_path):
    cfg = _tiny()  # 300 steps, num_checkpoints=4
    cfg["train"].update(checkpoint_every=50, checkpoint_every_until=200)
    train(cfg, 0, tmp_path / "run.csv", tmp_path / "ck", verbose=False)
    full = set(log_checkpoint_steps(300, 4))
    saved = sorted(int(p.stem[4:]) for p in (tmp_path / "ck").glob("step*.pt"))
    assert saved == checkpoint_steps(300, 4, 50, 200)
    for path in sorted((tmp_path / "ck").glob("step*.pt")):
        step = int(path.stem[4:])
        ck = torch.load(path, weights_only=False)
        if step in full:
            assert "optimizer" in ck, f"step {step} is log-spaced and must be resumable"
            if step > 0:
                assert ck["optimizer"]["state"]
        else:
            assert "optimizer" not in ck, f"step {step} is analysis-only"
        load_checkpoint(path)  # every checkpoint must load for analysis


def test_dense_checkpointing_does_not_change_metrics(tmp_path):
    """Extra checkpoints must not perturb training -- the Colab rerun depends on this."""
    dense = _tiny()
    dense["train"].update(checkpoint_every=50, checkpoint_every_until=200)
    a = train(_tiny(), 0, tmp_path / "a.csv", tmp_path / "ck_a", verbose=False).read_bytes()
    b = train(dense, 0, tmp_path / "b.csv", tmp_path / "ck_b", verbose=False).read_bytes()
    assert a == b


def test_parse_seeds():
    assert parse_seeds("0-9") == list(range(10))
    assert parse_seeds("4") == [4]
    assert parse_seeds("0,2,5-7") == [0, 2, 5, 6, 7]
