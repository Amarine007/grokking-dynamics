"""Experiment 04: when does each seed commit to its key frequencies? (Stage 4)

    python experiments/04_frequency_timing.py --config configs/frequency_timing.yaml

Analysis only: reads every checkpoint of each baseline seed on CPU and never trains.
Definitions and tests are pre-registered in the config.

Outputs (all in results/04_frequency_timing/):
    spectra.csv        per seed, checkpoint, matrix: share of squared norm per frequency
    timing.csv         per seed, checkpoint: final-set share/overlap/rank, own top-n set,
                       restricted loss with the own set
    init_test.csv      per matrix: step-0 permutation test (the pre-registered test)
    rank_by_step.csv   per matrix, checkpoint: the same statistic, descriptive only
    decisions.csv      per seed: set-decided step per matrix, and its interpretation band
    lock_in.csv        per seed, matrix, final key frequency: rank at init, lock-in step
    competitors.csv    every non-final frequency that ever entered a seed's top n
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import make_data  # noqa: E402
from src.frequency_timing import (  # noqa: E402
    frequency_ranks,
    rank_permutation_test,
    settled_step,
    spectra,
    top_frequencies,
)
from src.measures import full_logit_grid, restricted_logits, score  # noqa: E402
from src.train import REPO_ROOT, checkpoint_dir, load_checkpoint, load_config, parse_seeds  # noqa: E402

EXPERIMENT = "04_frequency_timing"
OUT_DIR = REPO_ROOT / "results" / EXPERIMENT


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({len(rows)} rows)")


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def band(step: int | None, memorization_end: int) -> str:
    if step is None:
        return "not settled by the last checkpoint"
    if step == 0:
        return "latent at initialization"
    if step <= memorization_end:
        return "decided during memorization"
    return "decided during circuit formation or later"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=str, help="override the config's seeds, e.g. 0 or 0-4")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(cfg["runtime"]["num_threads"])
    torch.use_deterministic_algorithms(True)
    if cfg["runtime"]["device"] != "cpu":
        raise RuntimeError("Stage 4 analysis is CPU-only; set runtime.device: cpu")

    src, a = cfg["source"], cfg["analysis"]
    matrices = a["matrices"]
    base = load_config(REPO_ROOT / "configs" / f"{src['config']}.yaml")
    p = base["data"]["p"]
    data = make_data(p, base["data"]["train_frac"], base["data"]["split_seed"])
    finals = {int(r["seed"]): sorted(int(k) for k in r["frequencies"].split())
              for r in read_csv(REPO_ROOT / src["key_frequencies_csv"])}
    mem_end = {int(r["seed"]): int(r["memorization_end"]) for r in read_csv(REPO_ROOT / src["phases_csv"])}
    seeds = parse_seeds(args.seeds if args.seeds else str(src["seeds"]))

    out = {k: [] for k in ("spectra", "timing", "decisions", "lock_in", "competitors")}
    ranks_by_step: dict[str, dict[int, list[dict]]] = {m: {} for m in matrices}
    for seed in seeds:
        final, n = finals[seed], len(finals[seed])
        paths = sorted(checkpoint_dir(src["experiment"], src["config"], seed).glob("step*.pt"))
        print(f"seed {seed}: final set {final}, {len(paths)} checkpoints", flush=True)
        steps, tops, ranks = [], {m: [] for m in matrices}, {m: [] for m in matrices}
        for path in paths:
            model, ck = load_checkpoint(path)
            if ck["seed"] != seed:
                raise RuntimeError(f"{path} holds seed {ck['seed']}, expected {seed}")
            step = int(ck["step"])
            steps.append(step)
            spec = spectra(model, p)
            row = {"seed": seed, "step": step, "n_key": n}
            for m in matrices:
                s = spec[m].numpy()
                r = frequency_ranks(s)
                top = top_frequencies(s, n)
                tops[m].append(top)
                ranks[m].append(r)
                ranks_by_step[m].setdefault(step, []).append(r)
                out["spectra"].append({"seed": seed, "step": step, "matrix": m,
                                       **{("const" if k == 0 else f"f{k}"): f"{v:.6g}" for k, v in enumerate(s)}})
                row[f"key_share_{m}"] = f"{sum(s[k] for k in final):.10g}"
                row[f"overlap_{m}"] = len(set(top) & set(final))
                row[f"mean_key_rank_{m}"] = f"{np.mean([r[k] for k in final]):.4g}"
                row[f"top_{m}"] = " ".join(map(str, top))
            own = tops["W_E"][-1]
            grid = full_logit_grid(model, data, cfg["runtime"]["batch_size"])
            row["restricted_loss_test_own"], row["restricted_acc_test_own"] = (
                f"{v:.10g}" for v in score(restricted_logits(grid, own), data, "test"))
            out["timing"].append(row)

        if tops["W_E"][-1] != final:
            raise RuntimeError(f"seed {seed}: top-{n} of W_E at the last checkpoint is {tops['W_E'][-1]}, "
                               f"not the Stage 2 set {final}")
        decision = {"seed": seed, "n_key": n, "final_set": " ".join(map(str, final)),
                    "memorization_end": mem_end[seed]}
        for m in matrices:
            decided = settled_step(steps, [t == final for t in tops[m]])
            decision[f"set_decided_{m}"] = decided
            decision[f"band_{m}"] = band(decided, mem_end[seed])
            for k in final:
                out["lock_in"].append({
                    "seed": seed, "matrix": m, "frequency": k, "rank_at_init": ranks[m][0][k],
                    "lock_in_step": settled_step(steps, [r[k] <= n for r in ranks[m]]),
                })
            for k in sorted({k for t in tops[m] for k in t} - set(final)):
                present = [s for s, t in zip(steps, tops[m]) if k in t]
                out["competitors"].append({
                    "seed": seed, "matrix": m, "frequency": k, "first_step": present[0],
                    "last_step": present[-1], "n_checkpoints": len(present),
                    "best_rank": min(r[k] for r in ranks[m]),
                })
        out["decisions"].append(decision)
        print("  " + "  ".join(f"{m}: decided {decision[f'set_decided_{m}']} ({decision[f'band_{m}']})"
                               for m in matrices))

    init_rows, by_step_rows = [], []
    key_sets = [finals[s] for s in seeds]
    for m in matrices:
        res = rank_permutation_test(ranks_by_step[m][0], key_sets, a["init_test_draws"], a["init_test_rng_seed"])
        init_rows.append({"matrix": m, "step": 0, "primary": int(m == matrices[0]), **res,
                          "alpha": a["alpha"], "significant": int(res["p_value"] < a["alpha"])})
        print(f"  init test {m}: mean rank {res['observed_mean_rank']:.2f} vs null {res['null_mean_rank']:.2f} "
              f"(sd {res['null_sd']:.2f}), p = {res['p_value']:.4g}")
        for step in sorted(ranks_by_step[m]):
            res = rank_permutation_test(ranks_by_step[m][step], key_sets, a["per_checkpoint_draws"],
                                        a["init_test_rng_seed"])
            by_step_rows.append({"matrix": m, "step": step, **{k: (f"{v:.6g}" if isinstance(v, float) else v)
                                                               for k, v in res.items()}})

    write_csv(OUT_DIR / "spectra.csv", out["spectra"])
    write_csv(OUT_DIR / "timing.csv", out["timing"])
    write_csv(OUT_DIR / "init_test.csv", init_rows)
    write_csv(OUT_DIR / "rank_by_step.csv", by_step_rows)
    write_csv(OUT_DIR / "decisions.csv", out["decisions"])
    write_csv(OUT_DIR / "lock_in.csv", out["lock_in"])
    if out["competitors"]:
        write_csv(OUT_DIR / "competitors.csv", out["competitors"])


if __name__ == "__main__":
    main()
