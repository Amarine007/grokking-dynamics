"""Experiment 02 (add-on): is attention part of the algorithm, or just a fixed router?

    python experiments/02_attention.py --config configs/attention.yaml

Analysis only, CPU, a few minutes. Three measurements per seed:

  1. Structure -- the final-position query must be constant, and the a-score must not
     depend on b (nor the b-score on a). Checks the argument in src/attention.py.
  2. Frozen attention -- replace the pattern with its mean over all inputs. If the model
     survives, attention carries no information about a or b.
  3. Per-head mean-ablation -- which heads matter.

Outputs land in results/02_attention/, one row per (seed, head) or (seed, condition).
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attention import (  # noqa: E402
    ablate_head_hook,
    factorisation_error,
    final_patterns,
    freeze_pattern_hook,
    head_mean_z,
    hooked_logit_grid,
    mean_pattern,
    query_spread,
    variance_retained,
)
from src.data import make_data  # noqa: E402
from src.fourier import evaluate_logit_grid  # noqa: E402
from src.model import to_hooked_transformer  # noqa: E402
from src.train import REPO_ROOT, load_checkpoint, load_config, parse_seeds  # noqa: E402

EXPERIMENT = "02_attention"
OUT_DIR = REPO_ROOT / "results" / EXPERIMENT


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({len(rows)} rows)")


def analyse_seed(cfg: dict, seed: int, p: int, data) -> dict:
    s, a = cfg["source"], cfg["analysis"]
    batch = cfg["runtime"]["batch_size"]
    path = REPO_ROOT / "checkpoints" / s["experiment"] / f"{s['config']}_seed{seed}" / f"step{s['step']:06d}.pt"
    model, ck = load_checkpoint(path)
    if ck["step"] != s["step"] or ck["seed"] != seed:
        raise RuntimeError(f"{path} holds step {ck['step']} seed {ck['seed']}")
    hooked = to_hooked_transformer(model)

    # --- 1. structure -----------------------------------------------------------
    spread = query_spread(hooked, p, batch)
    patterns = final_patterns(hooked, p, batch)
    err_a, err_b = factorisation_error(patterns)
    mean = mean_pattern(patterns)
    structure_row = {
        "seed": seed,
        "query_spread": f"{spread:.6g}",
        "query_constant": int(spread <= a["max_query_spread"]),
        "factorisation_error_a": f"{err_a:.6g}",
        "factorisation_error_b": f"{err_b:.6g}",
        "factorises": int(max(err_a, err_b) <= a["max_factorisation_error"]),
    }
    pattern_rows = [
        {
            "seed": seed,
            "head": h,
            "mean_attn_a": f"{mean[h, 0].item():.6g}",
            "mean_attn_b": f"{mean[h, 1].item():.6g}",
            "mean_attn_eq": f"{mean[h, 2].item():.6g}",
            "std_attn_a": f"{patterns[:, :, h, 0].std().item():.6g}",
            "std_attn_b": f"{patterns[:, :, h, 1].std().item():.6g}",
        }
        for h in range(mean.shape[0])
    ]

    # --- 2 and 3. frozen attention, then each head mean-ablated -----------------
    clean = hooked_logit_grid(hooked, p, [], batch)
    z_mean = head_mean_z(hooked, p, batch)
    conditions = {"frozen_attention": freeze_pattern_hook(mean)}
    for h in range(mean.shape[0]):
        conditions[f"ablate_head{h}"] = ablate_head_hook(h, z_mean)

    ablation_rows = [{"seed": seed, "condition": "clean", "variance_retained": 1.0,
                      **evaluate_logit_grid(clean, data)}]
    for name, hooks in conditions.items():
        grid = hooked_logit_grid(hooked, p, hooks, batch)
        ablation_rows.append({
            "seed": seed,
            "condition": name,
            "variance_retained": f"{variance_retained(clean, grid):.6g}",
            **evaluate_logit_grid(grid, data),
        })
    return {"structure": [structure_row], "patterns": pattern_rows, "ablation": ablation_rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=str)
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(cfg["runtime"]["num_threads"])
    torch.use_deterministic_algorithms(True)
    if cfg["runtime"]["device"] != "cpu":
        raise RuntimeError("this analysis is CPU-only; set runtime.device: cpu")

    base = load_config(REPO_ROOT / "configs" / f"{cfg['source']['config']}.yaml")
    p = base["data"]["p"]
    data = make_data(p, base["data"]["train_frac"], base["data"]["split_seed"])
    seeds = parse_seeds(args.seeds if args.seeds else str(cfg["source"]["seeds"]))

    collected: dict[str, list[dict]] = {"structure": [], "patterns": [], "ablation": []}
    for seed in seeds:
        print(f"seed {seed}:", flush=True)
        result = analyse_seed(cfg, seed, p, data)
        for name in collected:
            collected[name] += result[name]
        st = result["structure"][0]
        frozen = next(r for r in result["ablation"] if r["condition"] == "frozen_attention")
        print(f"  query spread {float(st['query_spread']):.2e} (constant: {bool(st['query_constant'])}), "
              f"factorisation err {float(st['factorisation_error_a']):.2e} / "
              f"{float(st['factorisation_error_b']):.2e}")
        print(f"  frozen attention: test acc {frozen['test_acc']:.4f}, "
              f"variance retained {float(frozen['variance_retained']):.4f}")

    write_csv(OUT_DIR / "structure.csv", collected["structure"])
    write_csv(OUT_DIR / "patterns.csv", collected["patterns"])
    write_csv(OUT_DIR / "ablation.csv", collected["ablation"])


if __name__ == "__main__":
    main()
