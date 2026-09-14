"""Same seed -> byte-identical CSV; different seed -> different run."""

import copy

import torch

from src.train import REPO_ROOT, CSV_COLUMNS, checkpoint_steps, load_config, parse_seeds, train

BASELINE = load_config(REPO_ROOT / "configs" / "baseline.yaml")
# Test determinism on the GPU when there is one (that is where results are produced).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _short(cfg: dict, num_steps: int) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["train"].update(num_steps=num_steps, log_every=50, num_checkpoints=4)
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


def test_parse_seeds():
    assert parse_seeds("0-9") == list(range(10))
    assert parse_seeds("4") == [4]
    assert parse_seeds("0,2,5-7") == [0, 2, 5, 6, 7]
