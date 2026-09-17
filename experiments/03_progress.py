"""Experiment 03: restricted and excluded loss through training (Stage 3).

    python experiments/03_progress.py --config configs/progress.yaml
    python experiments/03_progress.py --config configs/progress.yaml --seeds 0

Analysis only: runs every saved checkpoint of each baseline seed forward on CPU and
never trains. Before any measure is trusted, the clean losses and accuracies it
recomputes are checked against the committed Stage 1 CSV rows at the same steps, and
the parameter norm must agree too; a mismatch aborts the run.

Outputs (all in results/03_progress/):
    measures.csv   per seed, per checkpoint: clean metrics, restricted and excluded
                   loss/accuracy (both variants), parameter norm
    phases.csv     per seed: the pre-registered phase boundaries and the restricted-loss lead
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import make_data  # noqa: E402
from src.measures import full_logit_grid, phase_boundaries, progress_measures  # noqa: E402
from src.train import (  # noqa: E402
    REPO_ROOT,
    checkpoint_dir,
    load_checkpoint,
    load_config,
    param_norm,
    parse_seeds,
    read_metrics,
    results_path,
)

EXPERIMENT = "03_progress"
OUT_DIR = REPO_ROOT / "results" / EXPERIMENT


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write `rows` with the keys of the first row as the header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({len(rows)} rows)")


def load_key_frequencies(path: Path) -> dict[int, list[int]]:
    with open(path, newline="") as f:
        return {int(r["seed"]): [int(k) for k in r["frequencies"].split()] for r in csv.DictReader(f)}


def check_against_stage1(seed: int, rows: list[dict], stage1: dict, tol: dict) -> tuple[int, float, float]:
    """Compare recomputed clean metrics with the Stage 1 CSV. Returns (n rows, max rel loss, max acc diff)."""
    by_step = {int(s): i for i, s in enumerate(stage1["step"])}
    worst_loss = worst_acc = 0.0
    n = 0
    for r in rows:
        i = by_step.get(r["step"])
        if i is None:
            continue
        n += 1
        for split in ("train", "test"):
            ref = stage1[f"{split}_loss"][i]
            worst_loss = max(worst_loss, abs(r[f"{split}_loss"] - ref) / abs(ref))
            worst_acc = max(worst_acc, abs(r[f"{split}_acc"] - stage1[f"{split}_acc"][i]))
        norm_rel = abs(r["param_norm"] - stage1["param_norm"][i]) / stage1["param_norm"][i]
        if norm_rel > 1e-12:
            raise RuntimeError(f"seed {seed} step {r['step']}: param_norm differs from Stage 1 by {norm_rel:.2e} "
                               "(the checkpoint does not hold the weights that produced the CSV)")
    if n == 0:
        raise RuntimeError(f"seed {seed}: no checkpoint step appears in the Stage 1 CSV")
    if worst_loss > tol["check_loss_rel_tol"] or worst_acc > tol["check_acc_abs_tol"]:
        raise RuntimeError(f"seed {seed}: clean metrics disagree with Stage 1 "
                           f"(max rel loss diff {worst_loss:.2e}, max acc diff {worst_acc:.2e})")
    return n, worst_loss, worst_acc


def analyse_seed(cfg: dict, seed: int, data, key: list[int]) -> list[dict]:
    src = cfg["source"]
    ckpt_dir = checkpoint_dir(src["experiment"], src["config"], seed)
    paths = sorted(ckpt_dir.glob("step*.pt"))
    if not paths:
        raise RuntimeError(f"no checkpoints in {ckpt_dir}")
    rows = []
    for path in paths:
        model, ck = load_checkpoint(path)
        if ck["seed"] != seed:
            raise RuntimeError(f"{path} holds seed {ck['seed']}, expected {seed}")
        grid = full_logit_grid(model, data, cfg["runtime"]["batch_size"])
        rows.append({"step": int(ck["step"]), "param_norm": param_norm(model),
                     **progress_measures(grid, data, key)})
    rows.sort(key=lambda r: r["step"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=str, help="override the config's seeds, e.g. 0 or 0-4")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(cfg["runtime"]["num_threads"])
    torch.use_deterministic_algorithms(True)
    if cfg["runtime"]["device"] != "cpu":
        raise RuntimeError("Stage 3 analysis is CPU-only; set runtime.device: cpu")

    src = cfg["source"]
    base = load_config(REPO_ROOT / "configs" / f"{src['config']}.yaml")
    p = base["data"]["p"]
    data = make_data(p, base["data"]["train_frac"], base["data"]["split_seed"])
    keys = load_key_frequencies(REPO_ROOT / src["key_frequencies_csv"])
    seeds = parse_seeds(args.seeds if args.seeds else str(src["seeds"]))

    measure_rows, phase_rows = [], []
    for seed in seeds:
        key = sorted(keys[seed])
        print(f"seed {seed}: key frequencies {key}", flush=True)
        rows = analyse_seed(cfg, seed, data, key)
        stage1 = read_metrics(results_path(src["experiment"], src["config"], seed))
        n, dl, da = check_against_stage1(seed, rows, stage1, cfg["measures"])
        print(f"  {len(rows)} checkpoints; {n} match Stage 1 (max rel loss diff {dl:.1e}, "
              f"max acc diff {da:.1e}, param_norm exact)")

        b = phase_boundaries(rows, p, cfg["phases"])
        lead = lambda x: (b["test_acc_jump"] - x) if None not in (x, b["test_acc_jump"]) else None  # noqa: E731
        phase_rows.append({
            "seed": seed, "n_key": len(key), **b,
            "lead_restricted_beats_uniform": lead(b["restricted_beats_uniform"]),
            "lead_restricted_peak": lead(b["restricted_peak"]),
        })
        print("  " + "  ".join(f"{k}={v}" for k, v in phase_rows[-1].items() if k not in ("seed", "n_key")))
        for r in rows:
            measure_rows.append({"seed": seed, "n_key": len(key),
                                 **{k: (v if isinstance(v, int) else f"{v:.10g}") for k, v in r.items()}})

    write_csv(OUT_DIR / "measures.csv", measure_rows)
    write_csv(OUT_DIR / "phases.csv", phase_rows)


if __name__ == "__main__":
    main()
