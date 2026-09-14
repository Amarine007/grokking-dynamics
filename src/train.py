"""Full-batch training loop with incremental CSV logging and log-spaced checkpoints.

Every hyperparameter comes from the YAML config; nothing here has a default that
silently changes behaviour. Metrics are written one row at a time (flushed) so a
killed run keeps the progress it made.
"""

from __future__ import annotations

import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.data import make_data
from src.model import ModelConfig, Transformer, model_config_for_p

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_COLUMNS = ["step", "train_loss", "test_loss", "train_acc", "test_acc", "param_norm"]

# Deterministic cuBLAS matmuls on GPU require this before cuBLAS is first used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def load_config(path: str | os.PathLike) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_name(path: str | os.PathLike) -> str:
    return Path(path).stem


def results_path(experiment: str, cfg_name: str, seed: int, root: Path = REPO_ROOT) -> Path:
    return root / "results" / experiment / f"{cfg_name}_seed{seed}.csv"


def checkpoint_dir(experiment: str, cfg_name: str, seed: int, root: Path = REPO_ROOT) -> Path:
    return root / "checkpoints" / experiment / f"{cfg_name}_seed{seed}"


def parse_seeds(spec: str) -> list[int]:
    """'3' -> [3]; '0-9' -> [0..9]; '0,2,5-7' -> [0, 2, 5, 6, 7]."""
    seeds: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def checkpoint_steps(num_steps: int, n: int) -> list[int]:
    """Step 0 plus ~n log-spaced steps in [1, num_steps] (always including num_steps)."""
    steps = np.unique(np.round(np.geomspace(1, num_steps, n)).astype(int))
    return sorted({0, *steps.tolist()})


def cross_entropy_high_precision(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with log-softmax in float64, as in the paper's reference code.

    In float32, per-example losses underflow once the model is very confident,
    which distorts late-training gradients under strong weight decay.
    """
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    return -logprobs.gather(-1, labels[:, None]).mean()


@torch.no_grad()
def evaluate(model: Transformer, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    logits = model(x, last_only=True)
    loss = cross_entropy_high_precision(logits, y).item()
    acc = (logits.argmax(-1) == y).double().mean().item()
    return loss, acc


@torch.no_grad()
def param_norm(model: torch.nn.Module) -> float:
    return torch.sqrt(sum(p.double().pow(2).sum() for p in model.parameters())).item()


def resolve_device(cfg: dict) -> torch.device:
    """The config's runtime.device. Never silently falls back: CPU and GPU runs are
    not bit-identical, so a run must happen on the device its config names."""
    name = cfg["runtime"]["device"]
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config sets runtime.device: cuda but no GPU is available")
    return torch.device(name)


def build_model(cfg: dict) -> Transformer:
    return Transformer(model_config_for_p(cfg["data"]["p"], cfg["model"]))


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    o = cfg["optim"]
    if o["name"] != "adamw":
        raise ValueError(f"unsupported optimizer {o['name']!r}")
    return torch.optim.AdamW(
        model.parameters(), lr=o["lr"], betas=tuple(o["betas"]), weight_decay=o["weight_decay"]
    )


def save_checkpoint(path: Path, *, step: int, seed: int, cfg: dict, model: Transformer, opt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "seed": seed,
            "config": cfg,
            "model_config": model.cfg.to_dict(),
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(path: str | os.PathLike) -> tuple[Transformer, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Transformer(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def read_metrics(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Read a metrics CSV written by `train` into {column: array}. Tolerates a run
    that is still in progress (a truncated final line is ignored)."""
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if all(r.get(c) not in (None, "") for c in CSV_COLUMNS)]
    out = {c: np.array([float(r[c]) for r in rows]) for c in CSV_COLUMNS}
    out["step"] = out["step"].astype(int)
    return out


def _fmt(v: float | int) -> str:
    return str(v) if isinstance(v, int) else repr(float(v))


def train(cfg: dict, seed: int, csv_path: Path, ckpt_dir: Path | None, verbose: bool = True) -> Path:
    """Train one seed. Writes metrics to `csv_path`; checkpoints to `ckpt_dir` if given."""
    device = resolve_device(cfg)
    torch.set_num_threads(cfg["runtime"]["num_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False  # keep true float32 matmuls on GPU
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(seed)

    d = cfg["data"]
    data = make_data(d["p"], d["train_frac"], d["split_seed"])
    train_x, train_y, test_x, test_y = (
        t.to(device) for t in (data.train_x, data.train_y, data.test_x, data.test_y)
    )

    model = build_model(cfg).to(device)  # initialised on CPU, so the init is device-independent
    opt = build_optimizer(model, cfg)

    t = cfg["train"]
    num_steps, log_every = t["num_steps"], t["log_every"]
    ckpt_steps = set(checkpoint_steps(num_steps, t["num_checkpoints"]))

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        f.flush()
        for step in range(num_steps + 1):
            if step % log_every == 0 or step == num_steps:
                model.eval()
                train_loss, train_acc = evaluate(model, train_x, train_y)
                test_loss, test_acc = evaluate(model, test_x, test_y)
                row = [step, train_loss, test_loss, train_acc, test_acc, param_norm(model)]
                writer.writerow([_fmt(v) for v in row])
                f.flush()
                if verbose and step % (10 * log_every) == 0:
                    print(
                        f"[{csv_path.stem}] step {step:6d}  train {train_loss:.4f}/{train_acc:.3f}  "
                        f"test {test_loss:.4f}/{test_acc:.3f}  |w| {row[-1]:.2f}  ({time.time() - t0:.0f}s)",
                        flush=True,
                    )
            if ckpt_dir is not None and step in ckpt_steps:
                save_checkpoint(Path(ckpt_dir) / f"step{step:06d}.pt", step=step, seed=seed, cfg=cfg, model=model, opt=opt)
            if step == num_steps:
                break

            model.train()
            loss = cross_entropy_high_precision(model(train_x, last_only=True), train_y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return csv_path
