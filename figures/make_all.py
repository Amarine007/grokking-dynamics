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
EXP03 = "03_progress"
EXP04 = "04_frequency_timing"

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


# Stage 3 series take categorical slots 3-4 after train/test (validated as a 4-slot set;
# aqua and yellow are under 3:1 on the surface, so every line is also direct-labelled
# and the two measures are dashed).
RESTRICTED, EXCLUDED = "#1baf7a", "#eda100"
PHASES = [("memorization_end", "Memorization"), ("cleanup_start", "Circuit formation"),
          ("cleanup_end", "Cleanup")]


def seed_series(rows: list[dict], seed: int, cols: list[str]) -> dict[str, np.ndarray]:
    rows = [r for r in rows if int(r["seed"]) == seed]
    out = {c: np.array([float(r[c]) for r in rows]) for c in cols}
    out["step"] = np.array([int(r["step"]) for r in rows])
    return out


def _boundaries(ax, phase: dict, label: bool) -> None:
    """Vertical lines at the pre-registered phase boundaries; phase names above the plot."""
    edges = [1] + [int(phase[k]) for k, _ in PHASES if phase[k] not in ("", None)]
    for x in edges[1:]:
        ax.axvline(x, color=AXIS, linewidth=0.9, linestyle=(0, (3, 2)), zorder=0)
    if not label:
        return
    names = [n for _, n in PHASES][: len(edges)] + ["Stable"]
    right = [*edges[1:], None]
    for i, (left, r, name) in enumerate(zip(edges, right, names)):
        mid = np.sqrt(left * r) if r else left * 2.2
        # Alternate two rows: cleanup is narrow on a log axis, so neighbours would collide.
        ax.annotate(name, xy=(mid, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3 + 11 * (i % 2)),
                    textcoords="offset points", ha="center" if r else "left", va="bottom",
                    color=MUTED, fontsize=8)


def _end_label(ax, x: np.ndarray, y: np.ndarray, text: str, dy: float = 0) -> None:
    ax.annotate(text, xy=(x[-1], y[-1]), xytext=(5, dy), textcoords="offset points",
                va="center", ha="left", color=INK_2, fontsize=8)


def fig_progress_measures(rows: list[dict], phases: list[dict], seed: int, p: int, out: Path) -> None:
    """Both measures against train/test for one seed: losses above, accuracies below."""
    cols = ["train_loss", "test_loss", "restricted_loss_test", "excluded_loss_train",
            "train_acc", "test_acc", "restricted_acc_test", "excluded_acc_train"]
    s = seed_series(rows, seed, cols)
    ph = next(r for r in phases if int(r["seed"]) == seed)
    keep = s["step"] > 0
    x = s["step"][keep]
    series = [("train", TRAIN, "-", "Train"), ("test", TEST, "-", "Test"),
              ("restricted_{}_test", RESTRICTED, (0, (5, 2)), "Restricted (test)"),
              ("excluded_{}_train", EXCLUDED, (0, (5, 2)), "Excluded (train)")]

    fig, (a_loss, a_acc) = plt.subplots(2, 1, figsize=(8.6, 7.0), sharex=True, layout="constrained")
    floor = 1e-9  # log axis; the smallest real loss here is ~2e-8
    ends = {"Train": 0, "Test": 9, "Restricted (test)": -10, "Excluded (train)": 0}
    for key, color, ls, name in series:
        col = f"{key}_loss" if "{}" not in key else key.format("loss")
        y = np.maximum(s[col][keep], floor)
        a_loss.plot(x, y, color=color, linestyle=ls, label=name)
        _end_label(a_loss, x, y, name, ends[name])
        col = f"{key}_acc" if "{}" not in key else key.format("acc")
        a_acc.plot(x, s[col][keep], color=color, linestyle=ls, label=name)
    a_loss.axhline(np.log(p), color=MUTED, linewidth=0.8)
    a_loss.annotate(f"uniform guess, ln {p}", xy=(x[0], np.log(p)), xytext=(4, 4),
                    textcoords="offset points", color=MUTED, fontsize=8)
    a_loss.set_yscale("log")
    a_loss.set_ylim(floor, 100)
    a_loss.set_ylabel("Cross-entropy loss (log scale)")
    a_acc.axhline(1 / p, color=MUTED, linewidth=0.8)
    a_acc.set_ylim(0, 1.02)
    a_acc.yaxis.set_major_formatter(PercentFormatter(1.0))
    a_acc.set_ylabel("Accuracy")
    a_acc.legend(loc="center left")
    for ax, label in ((a_loss, True), (a_acc, False)):
        _boundaries(ax, ph, label)
        ax.set_xscale("log")
        ax.set_xlim(1, s["step"][-1] * 4)  # right margin for the end labels
    a_acc.set_xlabel("Training step (log scale)")
    fig.suptitle(
        f"Progress measures, seed {seed}: the key frequencies alone beat chance on test pairs by step "
        f"{ph['restricted_beats_uniform']},\n{ph['lead_restricted_beats_uniform']} steps before the full "
        f"model's test accuracy reaches 50%", x=0.01, ha="left", color=INK, fontweight="semibold", fontsize=10)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_progress_all_seeds(rows: list[dict], phases: list[dict], p: int, out: Path) -> None:
    """Small multiples: test loss and both measures for every seed, same axes."""
    seeds = sorted({int(r["seed"]) for r in rows})
    ncol = 5
    nrow = int(np.ceil(len(seeds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.0, 2.6 * nrow + 0.6), sharex=True, sharey=True,
                             layout="constrained", squeeze=False)
    cols = ["test_loss", "restricted_loss_test", "excluded_loss_train"]
    floor = 1e-9
    for ax, seed in zip(axes.flat, seeds):
        s = seed_series(rows, seed, cols)
        keep = s["step"] > 0
        x = s["step"][keep]
        for col, color, ls, name in [("test_loss", TEST, "-", "Test loss"),
                                     ("restricted_loss_test", RESTRICTED, (0, (5, 2)), "Restricted loss (test)"),
                                     ("excluded_loss_train", EXCLUDED, (0, (5, 2)), "Excluded loss (train)")]:
            ax.plot(x, np.maximum(s[col][keep], floor), color=color, linestyle=ls, linewidth=1.3, label=name)
        ax.axhline(np.log(p), color=MUTED, linewidth=0.7)
        _boundaries(ax, next(r for r in phases if int(r["seed"]) == seed), label=False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, s["step"][-1])
        ax.set_ylim(floor, 100)
        ax.set_title(f"Seed {seed}")
    for ax in axes.flat[len(seeds):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Step (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Loss (log)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("Every seed: restricted loss sits below ln p while test loss is still far above it, and "
                 "excluded loss climbs before the jump (dashed grey: pre-registered phase boundaries; solid grey: ln p)",
                 x=0.01, ha="left", color=INK, fontweight="semibold", fontsize=10)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_phase_timeline(phases: list[dict], last_step: int, out: Path) -> None:
    """Each seed's phases as one horizontal bar on a log step axis."""
    seeds = [int(r["seed"]) for r in phases]
    # Ordinal: one hue, light to dark, in phase order.
    shades = [SEQ_BLUE[2], SEQ_BLUE[6], SEQ_BLUE[10], GRID]
    names = ["Memorization", "Circuit formation", "Cleanup", "Stable"]
    fig, ax = plt.subplots(figsize=(8.6, 0.34 * len(seeds) + 1.4), layout="constrained")
    for i, r in enumerate(phases):
        edges = [1, int(r["memorization_end"]), int(r["cleanup_start"]),
                 int(r["cleanup_end"]) if r["cleanup_end"] else last_step, last_step]
        for j in range(4):
            if edges[j + 1] > edges[j]:
                ax.barh(i, edges[j + 1] - edges[j], left=edges[j], height=0.55, color=shades[j],
                        edgecolor=SURFACE, linewidth=2, label=names[j] if i == 0 else None)
        ax.plot(int(r["restricted_beats_uniform"]), i, marker="o", markersize=5, color=INK,
                linestyle="none", label="Restricted loss beats ln p (test)" if i == 0 else None)
    ax.set_xscale("log")
    ax.set_xlim(1, last_step)
    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels([f"Seed {s}" for s in seeds])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Training step (log scale)")
    ax.set_title("Pre-registered phase boundaries per seed")
    fig.legend(loc="outside lower center", ncol=5, fontsize=8)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def print_stage3_post_hoc(measures: list[dict], phases: list[dict]) -> None:
    """Descriptive numbers chosen AFTER seeing the curves, kept apart from the pre-registered ones.

    The pre-registered restricted-loss rule (beats ln p) fires almost immediately, so it
    says little beyond "right after memorization". Restricted test accuracy reaching 50%
    is a more demanding marker; it is reported here, labelled post hoc.
    """
    print("\nStage 3, POST HOC (not pre-registered): restricted test accuracy >= 50%")
    print(f"{'seed':>4}  {'step':>6}  {'full test acc then':>18}  {'lead to jump':>12}  {'excl. loss rise':>15}")
    leads = []
    for ph in phases:
        seed = int(ph["seed"])
        rows = [r for r in measures if int(r["seed"]) == seed]
        hit = next(r for r in rows if float(r["restricted_acc_test"]) >= 0.5)
        excl = {int(r["step"]): float(r["excluded_loss_train"]) for r in rows}
        rise = excl[int(ph["cleanup_start"])] / excl[int(ph["memorization_end"])]
        lead = int(ph["test_acc_jump"]) - int(hit["step"])
        leads.append(lead)
        print(f"{seed:>4}  {hit['step']:>6}  {float(hit['test_acc']):>17.1%}  {lead:>12}  {rise:>14.0f}x")
    print(f"  lead median {np.median(leads):.0f}, min {min(leads)}, max {max(leads)}")


def print_stage3(phases: list[dict]) -> None:
    keys = ["memorization_end", "cleanup_start", "cleanup_end", "restricted_beats_uniform",
            "restricted_peak", "test_acc_jump", "lead_restricted_beats_uniform"]
    print("\nStage 3: pre-registered phase boundaries (checkpoint steps)")
    print(f"{'seed':>4}  " + "  ".join(f"{k[:14]:>14}" for k in keys))
    for r in phases:
        print(f"{r['seed']:>4}  " + "  ".join(f"{r[k]:>14}" for k in keys))
    for k in keys:
        v = np.array([float(r[k]) for r in phases if r[k] not in ("", None)])
        print(f"  {k:<30} median {np.median(v):>7.0f}  min {v.min():>6.0f}  max {v.max():>6.0f}  "
              f"(n={len(v)})")


# Stage 4: up to five final key frequencies per seed take categorical slots 1-5 in
# ascending frequency order (validated as a 5-slot set; slots 3-5 are under 3:1 on the
# surface, so every line is direct-labelled).
CATEGORICAL = [TRAIN, TEST, RESTRICTED, EXCLUDED, "#e87ba4"]


def fig_frequency_ranks(timing_spectra: list[dict], decisions: list[dict], phases: list[dict],
                        seed: int, out: Path) -> None:
    """Rank of each final key frequency in W_E and W_L over training, for one seed."""
    dec = next(r for r in decisions if int(r["seed"]) == seed)
    final = [int(k) for k in dec["final_set"].split()]
    n = len(final)
    ph = next(r for r in phases if int(r["seed"]) == seed)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True, layout="constrained")
    for ax, m, title in [(axes[0], "W_E", "Embedding W_E"), (axes[1], "W_L", "Neuron-logit map W_L")]:
        rows = sorted((r for r in timing_spectra if int(r["seed"]) == seed and r["matrix"] == m),
                      key=lambda r: int(r["step"]))
        steps = np.array([int(r["step"]) for r in rows])
        keep = steps > 0
        rank = {k: [] for k in final}
        for r in rows:
            vals = np.array([float(r[f"f{k}"]) for k in range(1, 57)])
            order = sorted(range(56), key=lambda i: (-vals[i], i))
            pos = {i + 1: j + 1 for j, i in enumerate(order)}
            for k in final:
                rank[k].append(pos[k])
        ax.axhspan(0.5, n + 0.5, color=GRID, alpha=0.6, linewidth=0, zorder=0, label=f"Top {n}")
        # Final ranks sit one apart, too close for end labels, so identity goes in a legend.
        for color, k in zip(CATEGORICAL, final):
            ax.plot(steps[keep], np.array(rank[k])[keep], color=color, linewidth=1.5, label=f"k = {k}")
        for key, label in (("memorization_end", "memorization ends"), ("test_acc_jump", "test acc 50%")):
            x = int(ph[key])
            ax.axvline(x, color=AXIS, linewidth=0.9, linestyle=(0, (3, 2)), zorder=0)
            ax.annotate(label, xy=(x, 56), xytext=(3, 0), textcoords="offset points", va="bottom",
                        ha="left", color=MUTED, fontsize=8, rotation=90)
        ax.set_xscale("log")
        ax.set_xlim(1, steps[-1] * 1.5)
        ax.set_ylim(57, 0)
        ax.set_xlabel("Training step (log scale)")
        decided = dec[f"set_decided_{m}"]
        ax.set_title(f"{title}: set decided at step {decided}" if decided else f"{title}: exact set never matches")
    axes[0].set_ylabel("Rank among 56 frequencies (1 = most norm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=8)
    fig.suptitle(f"Seed {seed}: where the final key frequencies rank through training",
                 x=0.01, ha="left", color=INK, fontweight="semibold")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_lock_in(lock_in: list[dict], phases: list[dict], out: Path) -> None:
    """Per seed, the step each final key frequency enters W_E's top n for good."""
    rows = [r for r in lock_in if r["matrix"] == "W_E"]
    seeds = sorted({int(r["seed"]) for r in rows})
    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(seeds) + 1.5), layout="constrained")
    for i, seed in enumerate(seeds):
        ph = next(r for r in phases if int(r["seed"]) == seed)
        ax.plot([int(ph["memorization_end"])], [i], marker="|", markersize=14, color=INK_2, linestyle="none",
                label="Memorization ends (Stage 3)" if i == 0 else None)
        ax.plot([int(ph["test_acc_jump"])], [i], marker="|", markersize=14, color=TEST, linestyle="none",
                label="Test accuracy reaches 50%" if i == 0 else None)
        steps = [max(int(r["lock_in_step"]), 1) for r in rows if int(r["seed"]) == seed]
        # Frequencies locking in at the same checkpoint are offset vertically, not stacked.
        seen: dict[int, int] = {}
        ys = []
        for x in steps:
            ys.append(i + 0.18 * seen.get(x, 0))
            seen[x] = seen.get(x, 0) + 1
        ax.plot(steps, ys, marker="o", markersize=6, color=TRAIN, linestyle="none",
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label="A final key frequency locks into W_E's top n" if i == 0 else None)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 60000)
    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels([f"Seed {s}" for s in seeds])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Training step (log scale; step 0 drawn at 1)")
    ax.set_title("When each final key frequency locks in")
    fig.legend(loc="outside lower center", ncol=3, fontsize=8)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_rank_by_step(rows: list[dict], init: list[dict], out: Path) -> None:
    """Mean rank of the final key frequencies over all seeds, against random subsets."""
    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    for m, color, name in (("W_E", TRAIN, "Embedding W_E"), ("W_L", TEST, "Neuron-logit map W_L")):
        rr = sorted((r for r in rows if r["matrix"] == m), key=lambda r: int(r["step"]))
        steps = np.array([int(r["step"]) for r in rr])
        keep = steps > 0
        obs = np.array([float(r["observed_mean_rank"]) for r in rr])
        ax.plot(steps[keep], obs[keep], color=color, label=name)
        ax.annotate(name, xy=(steps[-1], obs[-1]), xytext=(5, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=8)
    rr = sorted((r for r in rows if r["matrix"] == "W_E"), key=lambda r: int(r["step"]))
    steps = np.array([int(r["step"]) for r in rr])
    null = np.array([float(r["null_mean_rank"]) for r in rr])
    sd = np.array([float(r["null_sd"]) for r in rr])
    keep = steps > 0
    ax.fill_between(steps[keep], (null - 2 * sd)[keep], (null + 2 * sd)[keep], color=MUTED, alpha=0.15,
                    linewidth=0, label="Random frequencies, ±2 SD")
    ax.set_xscale("log")
    ax.set_xlim(1, steps[-1] * 4)
    ax.set_ylim(57, 0)
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel("Mean rank of final key frequencies")
    e = next(r for r in init if r["matrix"] == "W_E")
    ax.set_title(f"Pooled over seeds: at initialization the final frequencies rank {float(e['observed_mean_rank']):.1f} "
                 f"on average (random: {float(e['null_mean_rank']):.1f}, p = {float(e['p_value']):.2g})")
    ax.legend(loc="lower left")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def print_stage4(decisions: list[dict], init: list[dict], lock_in: list[dict], timing: list[dict],
                 measures: list[dict], spectrum: list[dict]) -> None:
    print("\nStage 4, PRE-REGISTERED: initialization test and set-decided step")
    for r in init:
        print(f"  {r['matrix']} (primary={r['primary']}): mean rank {float(r['observed_mean_rank']):.2f} "
              f"vs random {float(r['null_mean_rank']):.2f} (sd {float(r['null_sd']):.2f}), "
              f"p = {float(r['p_value']):.3g}, n = {r['n_frequencies']}")
    for r in decisions:
        print(f"  seed {r['seed']}: W_E {r['set_decided_W_E'] or '-':>6} ({r['band_W_E']});  "
              f"W_L {r['set_decided_W_L'] or '-':>6} ({r['band_W_L']})")

    print("\nStage 4, POST HOC: early vs late lock-in (W_E), split at Stage 3 memorization end")
    mem = {r["seed"]: int(r["memorization_end"]) for r in decisions}
    share = {(r["seed"], r["frequency"]): float(r["fraction"]) for r in spectrum}
    wl = {(r["seed"], r["frequency"]): r["lock_in_step"] for r in lock_in if r["matrix"] == "W_L"}
    we = [r for r in lock_in if r["matrix"] == "W_E"]
    for name, group in (("early", [r for r in we if int(r["lock_in_step"]) <= mem[r["seed"]]]),
                        ("late", [r for r in we if int(r["lock_in_step"]) > mem[r["seed"]]])):
        steps = [int(r["lock_in_step"]) for r in group]
        shares = [share[(r["seed"], r["frequency"])] for r in group]
        never = sum(wl[(r["seed"], r["frequency"])] == "" for r in group)
        print(f"  {name:>5}: {len(group)} of {len(we)} frequencies, lock-in steps {min(steps)}-{max(steps)}, "
              f"final W_E share median {np.median(shares):.1%} ({min(shares):.1%}-{max(shares):.1%}), "
              f"mean init rank {np.mean([int(r['rank_at_init']) for r in group]):.1f}, never lock in W_L: {never}")

    print("\nStage 4, POST HOC: restricted test accuracy with own top-n set minus with final set (max over steps)")
    final_acc = {(r["seed"], r["step"]): float(r["restricted_acc_test"]) for r in measures}
    for seed in sorted({r["seed"] for r in timing}, key=int):
        best = max((float(r["restricted_acc_test_own"]) - final_acc[(seed, r["step"])], int(r["step"]), r["top_W_E"])
                   for r in timing if r["seed"] == seed)
        print(f"  seed {seed}: {best[0]:+.3f} at step {best[1]} (own set {best[2]})")


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

    measures_path = RESULTS / EXP03 / "measures.csv"
    if measures_path.exists():
        ref = int(load_config(ROOT / "configs" / "progress.yaml")["phases"]["reference_seed"])
        measures = read_rows(measures_path)
        phases = read_rows(RESULTS / EXP03 / "phases.csv")
        fig_progress_measures(measures, phases, ref, p, FIG_DIR / "fig06_progress_measures.png")
        fig_progress_all_seeds(measures, phases, p, FIG_DIR / "fig07_progress_all_seeds.png")
        fig_phase_timeline(phases, max(int(r["step"]) for r in measures), FIG_DIR / "fig08_phase_timeline.png")
        print_stage3(phases)
        print_stage3_post_hoc(measures, phases)

        timing_dir = RESULTS / EXP04
        if (timing_dir / "decisions.csv").exists():
            decisions = read_rows(timing_dir / "decisions.csv")
            fig_frequency_ranks(read_rows(timing_dir / "spectra.csv"), decisions, phases, ref,
                                FIG_DIR / "fig09_frequency_ranks.png")
            fig_lock_in(read_rows(timing_dir / "lock_in.csv"), phases, FIG_DIR / "fig10_lock_in.png")
            fig_rank_by_step(read_rows(timing_dir / "rank_by_step.csv"), read_rows(timing_dir / "init_test.csv"),
                             FIG_DIR / "fig11_rank_by_step.png")
            print_stage4(decisions, read_rows(timing_dir / "init_test.csv"), read_rows(timing_dir / "lock_in.csv"),
                         read_rows(timing_dir / "timing.csv"), measures,
                         read_rows(RESULTS / EXP02 / "embedding_spectrum.csv"))


if __name__ == "__main__":
    main()
