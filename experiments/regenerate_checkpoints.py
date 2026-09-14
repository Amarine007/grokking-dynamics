"""Regenerate checkpoints for committed runs and check the metrics reproduce exactly.

Checkpoints are not committed (they are ~3 MB each, ~20 per run). This re-trains each
requested (config, seed), writing checkpoints to checkpoints/ and the metrics to a
temporary CSV that is compared byte-for-byte with the committed one in results/.

    python experiments/regenerate_checkpoints.py --all --jobs 4
    python experiments/regenerate_checkpoints.py --config configs/baseline.yaml --seeds 0-4 --jobs 4

Exit status is 1 if any regenerated CSV differs from the committed CSV.
"""

import argparse
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import (  # noqa: E402
    REPO_ROOT,
    checkpoint_dir,
    config_name,
    load_config,
    parse_seeds,
    results_path,
    train,
)


def regenerate(experiment: str, config_path: str, seed: int) -> tuple[str, str]:
    cfg = load_config(config_path)
    name = config_name(config_path)
    committed = results_path(experiment, name, seed)
    with tempfile.TemporaryDirectory() as td:
        fresh = train(cfg, seed, Path(td) / "metrics.csv", checkpoint_dir(experiment, name, seed), verbose=False)
        if not committed.exists():
            status = "NO COMMITTED CSV"
        elif fresh.read_bytes() == committed.read_bytes():
            status = "MATCH"
        else:
            status = "MISMATCH"
    return f"{experiment}/{name}_seed{seed}", status


def committed_runs(experiment: str) -> list[tuple[str, int]]:
    runs = []
    for csv in sorted((REPO_ROOT / "results" / experiment).glob("*_seed*.csv")):
        m = re.fullmatch(r"(.+)_seed(\d+)", csv.stem)
        runs.append((str(REPO_ROOT / "configs" / f"{m.group(1)}.yaml"), int(m.group(2))))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="01_baseline")
    ap.add_argument("--all", action="store_true", help="every committed CSV of the experiment")
    ap.add_argument("--config")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seeds", type=str)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    if args.all:
        runs = committed_runs(args.experiment)
    else:
        if not args.config or (args.seed is None and args.seeds is None):
            ap.error("give --all, or --config with --seed/--seeds")
        seeds = [args.seed] if args.seed is not None else parse_seeds(args.seeds)
        runs = [(args.config, s) for s in seeds]

    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(regenerate, args.experiment, c, s) for c, s in runs]
        results = [f.result() for f in futures]
    for run, status in results:
        print(f"{status:17s} {run}")
    sys.exit(0 if all(s == "MATCH" for _, s in results) else 1)


if __name__ == "__main__":
    main()
