"""Regenerate every figure from the committed CSVs in results/ -- no retraining.

    python figures/make_all.py

Also prints the Stage 1 milestone table (first step with train acc > 99% and test
acc > 90%) for every run found.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train import load_config, read_metrics  # noqa: E402

FIG_DIR = ROOT / "figures"
RESULTS = ROOT / "results"
EXP01 = "01_baseline"

# Reference data-viz palette: categorical slots 1-2 plus chart chrome and ink.
TRAIN, TEST = "#2a78d6", "#eb6834"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "font.size": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def load_runs(experiment: str, config: str) -> dict[int, dict[str, np.ndarray]]:
    runs = {}
    for path in (RESULTS / experiment).glob(f"{config}_seed*.csv"):
        m = re.fullmatch(rf"{re.escape(config)}_seed(\d+)", path.stem)
        if m:
            runs[int(m.group(1))] = read_metrics(path)
    return dict(sorted(runs.items()))


def stack(runs: dict[int, dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """[n_seeds, n_steps] array over the steps every run has reached."""
    n = min(len(r["step"]) for r in runs.values())
    steps = next(iter(runs.values()))["step"][:n]
    return steps, np.stack([r[key][:n] for r in runs.values()])


def first_step_above(metrics: dict, key: str, threshold: float) -> int | None:
    hit = np.nonzero(metrics[key] > threshold)[0]
    return int(metrics["step"][hit[0]]) if len(hit) else None


def _log_x(ax, steps: np.ndarray) -> None:
    ax.set_xscale("log")
    ax.set_xlim(steps[steps > 0][0], steps[-1])
    ax.set_xlabel("Training step (log scale)")


def fig_accuracy_across_seeds(runs: dict, p: int, out: Path) -> None:
    steps, train = stack(runs, "train_acc")
    _, test = stack(runs, "test_acc")
    keep = steps > 0  # log axis
    fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    for arr, color, name in [(train, TRAIN, "Train"), (test, TEST, "Test")]:
        mean = arr.mean(0)
        sd = arr.std(0, ddof=1) if len(arr) > 1 else np.zeros_like(mean)
        lo, hi = np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1)
        ax.fill_between(steps[keep], lo[keep], hi[keep], color=color, alpha=0.12, linewidth=0)
        ax.plot(steps[keep], mean[keep], color=color, label=f"{name} accuracy (mean ± 1 SD)")
    ax.axhline(1 / p, color=MUTED, linewidth=0.8)
    ax.annotate(f"chance = 1/{p}", xy=(steps[-1], 1 / p), xytext=(-4, 4), textcoords="offset points",
                ha="right", color=MUTED, fontsize=8)
    _log_x(ax, steps)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Grokking on (a + b) mod {p}: {len(runs)} seeds, weight decay 1.0")
    ax.legend(loc="center left")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_weight_decay_control(base: dict, ctrl: dict, labels: tuple[str, str], p: int, out: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 8.0), sharex=True, sharey="row", layout="constrained")
    for col, (m, title) in enumerate([(base, labels[0]), (ctrl, labels[1])]):
        steps, keep = m["step"], m["step"] > 0
        a_acc, a_loss, a_norm = axes[0, col], axes[1, col], axes[2, col]
        a_acc.set_title(title)
        for split, color in [("train", TRAIN), ("test", TEST)]:
            a_acc.plot(steps[keep], m[f"{split}_acc"][keep], color=color, label=split.capitalize())
            a_loss.plot(steps[keep], m[f"{split}_loss"][keep], color=color, label=split.capitalize())
        a_acc.axhline(1 / p, color=MUTED, linewidth=0.8)
        a_norm.plot(steps[keep], m["param_norm"][keep], color=INK_2)
        a_loss.set_yscale("log")
        a_acc.yaxis.set_major_formatter(PercentFormatter(1.0))
        _log_x(a_norm, steps)
    axes[0, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylabel("Cross-entropy loss (log scale)")
    axes[2, 0].set_ylabel("L2 norm of all parameters")
    axes[0, 0].legend(loc="center left")
    fig.suptitle("Weight-decay control, same seed and split", x=0.01, ha="left", color=INK, fontweight="semibold")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def print_milestones(title: str, runs: dict) -> None:
    print(f"\n{title}")
    print(f"{'seed':>4}  {'last step':>9}  {'train>99%':>9}  {'test>90%':>9}  {'final train':>11}  {'final test':>10}")
    for seed, m in runs.items():
        tr, te = first_step_above(m, "train_acc", 0.99), first_step_above(m, "test_acc", 0.90)
        print(f"{seed:>4}  {m['step'][-1]:>9}  {str(tr):>9}  {str(te):>9}  "
              f"{m['train_acc'][-1]:>11.4f}  {m['test_acc'][-1]:>10.4f}")


def main() -> None:
    set_style()
    p = load_config(ROOT / "configs" / "baseline.yaml")["data"]["p"]
    base = load_runs(EXP01, "baseline")
    ctrl = load_runs(EXP01, "baseline_wd0")

    if base:
        fig_accuracy_across_seeds(base, p, FIG_DIR / "fig01_accuracy_across_seeds.png")
        print_milestones("baseline (weight decay 1.0)", base)
    if base and ctrl:
        seed = min(set(base) & set(ctrl))
        wd_base = load_config(ROOT / "configs" / "baseline.yaml")["optim"]["weight_decay"]
        wd_ctrl = load_config(ROOT / "configs" / "baseline_wd0.yaml")["optim"]["weight_decay"]
        fig_weight_decay_control(
            base[seed], ctrl[seed],
            (f"weight decay = {wd_base:g}  (baseline, seed {seed})", f"weight decay = {wd_ctrl:g}  (control, seed {seed})"),
            p, FIG_DIR / "fig02_weight_decay_control.png",
        )
        print_milestones("control (weight decay 0)", ctrl)


if __name__ == "__main__":
    main()
