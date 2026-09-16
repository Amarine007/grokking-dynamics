"""Regenerate every figure from the committed CSVs in results/ -- no retraining.

    python figures/make_all.py

Also prints the Stage 1 milestone table (first step with train acc > 99% and test
acc > 90%) for every run found.
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train import load_config, read_metrics  # noqa: E402

FIG_DIR = ROOT / "figures"
RESULTS = ROOT / "results"
EXP01 = "01_baseline"
EXP02 = "02_fourier"

# Reference data-viz palette: categorical slots 1-2 plus chart chrome and ink.
TRAIN, TEST = "#2a78d6", "#eb6834"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
# Sequential ramp for continuous magnitude: one hue, light to dark (never a rainbow).
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_DARK = SEQ_BLUE[-1]


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


def read_rows(path: Path) -> list[dict]:
    """A results CSV as a list of dicts, values left as strings."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fig_embedding_spectrum(rows: list[dict], key_rows: list[dict], seed: int, out: Path) -> None:
    """How the embedding's squared norm is spread over frequencies, for one seed.

    Two panels of the same data: linear, where the sparsity is the whole story, and
    log, where the gap between the last key frequency and the first excluded one is
    visible and the threshold can be judged rather than taken on trust.
    """
    rows = [r for r in rows if int(r["seed"]) == seed]
    freq = np.array([int(r["frequency"]) for r in rows])
    frac = np.array([float(r["fraction"]) for r in rows])
    is_key = np.array([bool(int(r["is_key"])) for r in rows])
    info = next(r for r in key_rows if int(r["seed"]) == seed)
    threshold = float(info["threshold"])
    keep = freq > 0  # index 0 is the constant term, not a frequency

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), layout="constrained")
    for ax, log in zip(axes, (False, True)):
        ax.bar(freq[keep & ~is_key], frac[keep & ~is_key], width=0.8, color=MUTED,
               label="Below threshold")
        ax.bar(freq[keep & is_key], frac[keep & is_key], width=0.8, color=TRAIN,
               label="Key frequency")
        ax.axhline(threshold, color=AXIS, linewidth=0.8)
        ax.set_xlabel("Frequency k")
        ax.set_xlim(0, freq.max() + 1)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-7, 1.0)
            ax.set_title("Log scale: the gap either side of the threshold")
            ax.annotate(f"threshold = {threshold:.0%}", xy=(freq.max(), threshold),
                        xytext=(-4, 9), textcoords="offset points", ha="right",
                        color=MUTED, fontsize=8)
        else:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_ylabel("Fraction of squared norm")
            ax.set_title(f"Linear scale: {info['n_key']} frequencies carry the embedding")
            for k, f in zip(freq[keep & is_key], frac[keep & is_key]):
                ax.annotate(str(k), xy=(k, f), xytext=(0, 3), textcoords="offset points",
                            ha="center", color=INK_2, fontsize=8)
    # Centre-right: the tall key-frequency bars and their labels own the top of the panel.
    axes[0].legend(loc="center right")
    fig.suptitle(
        f"Embedding DFT, seed {seed}: {info['n_key']} key frequencies carry "
        f"{float(info['fraction_of_norm']):.1%} of the squared norm",
        x=0.01, ha="left", color=INK, fontweight="semibold")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_neuron_heatmap(heat_rows: list[dict], neuron_rows: list[dict], p: int, out: Path,
                       selection: str = "single_frequency") -> None:
    """One neuron's activation over the whole (a, b) grid, and against a + b.

    The left panel shows the periodic structure the neuron imposes on (a, b); the right
    panel shows how much of it is actually a function of a + b, since the vertical
    spread at each a + b is exactly the part that is not.
    """
    heat_rows = [r for r in heat_rows if r["selection"] == selection]
    seed, neuron = int(heat_rows[0]["seed"]), int(heat_rows[0]["neuron"])
    grid = np.zeros((p, p))
    for r in heat_rows:
        grid[int(r["a"]), int(r["b"])] = float(r["activation"])
    info = next(r for r in neuron_rows if int(r["neuron"]) == neuron)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3), layout="constrained")
    ax = axes[0]
    ax.grid(False)
    im = ax.imshow(grid, origin="lower", cmap=cmap, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Activation (post-ReLU)", color=INK_2)
    cbar.outline.set_visible(False)
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_title(f"Neuron {neuron} over every (a, b)")

    s = ((np.arange(p)[:, None] + np.arange(p)[None, :]) % p).ravel()
    flat = grid.ravel()
    ax = axes[1]
    ax.scatter(s, flat, s=2, color=TRAIN, alpha=0.10, linewidths=0, label="Every (a, b) pair")
    ax.plot(np.arange(p), np.array([flat[s == v].mean() for v in range(p)]),
            color=SEQ_DARK, linewidth=1.6, label="Mean over pairs with that a + b")
    ax.set_xlabel("(a + b) mod p")
    ax.set_ylabel("Activation")
    ax.set_xlim(0, p - 1)
    ax.set_ylim(min(0.0, flat.min()), flat.max() * 1.22)  # headroom so the legend clears the data
    ax.set_title("The same neuron against a + b")
    ax.legend(loc="upper right")

    fig.suptitle(
        f"Seed {seed}, neuron {neuron}: frequency {info['frequency']}, with "
        f"{float(info['single_fve']):.0%} of its variance at that one frequency but only "
        f"{float(info['sum_fve']):.0%} of it a function of a + b",
        x=0.01, ha="left", color=INK, fontweight="semibold")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_ablation(rows: list[dict], out: Path) -> None:
    """What survives when the logits are rebuilt from a subset of Fourier components.

    Horizontal, so each bar's value sits at its own tip and the condition names read as
    ordinary left-aligned labels instead of wrapped x-tick text.
    """
    conds = ["key_diag", "key_block", "cross_terms", "ablate_key"]
    labels = ["Key frequencies,\nmatched on both axes", "Key frequencies,\nany pairing",
              "Cross terms only", "Key frequencies\nremoved"]

    def stat(cond: str, col: str) -> tuple[float, float]:
        v = np.array([float(r[col]) for r in rows if r["condition"] == cond])
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)

    var = [stat(c, "variance_retained") for c in conds]
    acc = [stat(c, "test_acc") for c in conds]
    n_seeds = len({r["seed"] for r in rows})
    y = np.arange(len(conds))
    h = 0.13  # keeps the drawn bar within the 24px mark cap at this figure size
    err = {"ecolor": INK_2, "elinewidth": 0.9, "capsize": 0}

    fig, ax = plt.subplots(figsize=(8.4, 4.2), layout="constrained")
    for offset, series, color, name in [
        (h / 2 + 0.006, var, TRAIN, "Logit variance retained"),
        (-h / 2 - 0.006, acc, TEST, "Test accuracy"),
    ]:
        ax.barh(y + offset, [m for m, _ in series], h, xerr=[s for _, s in series],
                color=color, label=name, error_kw=err)
        for yi, (m, _) in zip(y + offset, series):
            ax.annotate(f"{m:.1%}", xy=(m, yi), xytext=(6, 0), textcoords="offset points",
                        va="center", ha="left", color=INK_2, fontsize=8)
    # The 100% reference. Deliberately unlabelled: the top bars reach 100%, so any
    # annotation here lands on a mark, and the x-axis title already says what 100% means.
    ax.axvline(1.0, color=AXIS, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # first condition at the top
    ax.set_xlim(0, 1.16)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Fraction of the clean model")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Fourier-space ablation of the logits (mean ± 1 SD over {n_seeds} seeds)")
    ax.legend(loc="lower right")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def print_stage2(key_rows: list[dict], ablation: list[dict], neurons: list[dict]) -> None:
    print("\nStage 2: key frequencies and what they carry")
    print(f"{'seed':>4}  {'n':>2}  {'frequencies':>22}  {'of norm':>8}  "
          f"{'retained':>8}  {'test acc':>8}  {'1-freq neurons':>14}")
    for row in key_rows:
        seed = row["seed"]
        diag = next(r for r in ablation if r["seed"] == seed and r["condition"] == "key_diag")
        neu = next(r for r in neurons if r["seed"] == seed)
        print(f"{seed:>4}  {row['n_key']:>2}  {row['frequencies']:>22}  "
              f"{float(row['fraction_of_norm']):>7.1%}  "
              f"{float(diag['variance_retained']):>7.2%}  {float(diag['test_acc']):>7.2%}  "
              f"{neu['n_single_explained']:>6} / {neu['n_neurons']:<5}")


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

    spectrum_path = RESULTS / EXP02 / "embedding_spectrum.csv"
    if spectrum_path.exists():
        ref = int(load_config(ROOT / "configs" / "fourier.yaml")["analysis"]["reference_seed"])
        keys = read_rows(RESULTS / EXP02 / "key_frequencies.csv")
        ablation = read_rows(RESULTS / EXP02 / "ablation.csv")
        fig_embedding_spectrum(read_rows(spectrum_path), keys, ref,
                               FIG_DIR / "fig03_embedding_spectrum.png")
        heatmap = RESULTS / EXP02 / f"neuron_heatmap_seed{ref}.csv"
        if heatmap.exists():
            fig_neuron_heatmap(read_rows(heatmap), read_rows(RESULTS / EXP02 / f"neurons_seed{ref}.csv"),
                               p, FIG_DIR / "fig04_neuron_heatmap.png")
        fig_ablation(ablation, FIG_DIR / "fig05_ablation.png")
        print_stage2(keys, ablation, read_rows(RESULTS / EXP02 / "neuron_summary.csv"))


if __name__ == "__main__":
    main()
