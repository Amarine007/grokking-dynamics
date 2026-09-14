"""Experiment 01: baseline grokking run (and its weight-decay control).

    python experiments/01_baseline.py --config configs/baseline.yaml --seed 0
    python experiments/01_baseline.py --config configs/baseline.yaml --seeds 0-9 --jobs 4

Every hyperparameter comes from the YAML. Metrics go to
results/01_baseline/{config}_seed{k}.csv, checkpoints to
checkpoints/01_baseline/{config}_seed{k}/. `--jobs` only controls how many seeds
run in parallel processes; each process uses the config's runtime.num_threads.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import checkpoint_dir, config_name, load_config, parse_seeds, results_path, train  # noqa: E402

EXPERIMENT = "01_baseline"


def run_seed(config_path: str, seed: int) -> str:
    cfg = load_config(config_path)
    name = config_name(config_path)
    out = train(cfg, seed, results_path(EXPERIMENT, name, seed), checkpoint_dir(EXPERIMENT, name, seed))
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", type=int)
    g.add_argument("--seeds", type=str, help="e.g. 0-9 or 0,2,5-7")
    ap.add_argument("--jobs", type=int, default=1, help="seeds to run in parallel processes")
    args = ap.parse_args()

    seeds = [args.seed] if args.seed is not None else parse_seeds(args.seeds)
    if args.jobs == 1 or len(seeds) == 1:
        for s in seeds:
            print("wrote", run_seed(args.config, s), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
            for out in pool.map(run_seed, [args.config] * len(seeds), seeds):
                print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
