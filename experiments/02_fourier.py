"""Experiment 02: reverse-engineer the circuit of the grokked models (Stage 2).

    python experiments/02_fourier.py --config configs/fourier.yaml
    python experiments/02_fourier.py --config configs/fourier.yaml --seeds 0

Analysis only: reads the final checkpoint of each baseline seed and never trains.
Runs on CPU in a couple of minutes. Every number a figure or the README needs is
written to a CSV in results/02_fourier/, so nothing downstream ever reads a
checkpoint. Each CSV carries a `seed` column, so the same schema holds whether one
seed or ten were analysed.

Outputs (all in results/02_fourier/):
    embedding_spectrum.csv     per seed, per frequency: fraction of squared norm
    key_frequencies.csv        per seed: how many key frequencies, which, how much norm
    ablation.csv               per seed, per condition: variance retained, loss, accuracy
    neuron_summary.csv         per seed: distribution of variance explained over neurons
    neuron_frequencies.csv     per seed, per key frequency: neurons owned, phase spread
    neurons_seed{r}.csv        reference seed only: one row per neuron
    neuron_detail_seed{r}.csv  reference seed only: one row per (neuron, key frequency)
    neuron_heatmap_seed{r}.csv reference seed only: the best-fit neuron over every (a, b)
    misclassified.csv          every test pair any seed gets wrong
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import make_cayley_table, make_data  # noqa: E402
from src.fourier import (  # noqa: E402
    center_logits,
    embedding_spectrum,
    evaluate_logit_grid,
    fourier_2d,
    frequency_masks,
    inverse_fourier_2d,
    key_frequencies,
    logit_grid,
    misclassified_pairs,
    mlp_activations,
    neuron_frequency_fit,
    neuron_sum_fit,
    variance_fraction,
)
from src.model import to_hooked_transformer  # noqa: E402
from src.train import REPO_ROOT, load_checkpoint, load_config, parse_seeds  # noqa: E402

EXPERIMENT = "02_fourier"
OUT_DIR = REPO_ROOT / "results" / EXPERIMENT


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write `rows` with the keys of the first row as the header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({len(rows)} rows)")


def checkpoint_path(cfg: dict, seed: int) -> Path:
    s = cfg["source"]
    return (
        REPO_ROOT / "checkpoints" / s["experiment"] / f"{s['config']}_seed{seed}" / f"step{s['step']:06d}.pt"
    )


def circular_resultant(phases: torch.Tensor) -> float:
    """Concentration of a set of angles: 1.0 if identical, ~0 if spread around the circle.

    The clock algorithm predicts neurons sharing a frequency have phases spread around
    the circle (the a - b terms cancel only across the population), so a value near 0
    supports the hypothesis and a value near 1 would contradict it.
    """
    if phases.numel() == 0:
        return float("nan")
    return torch.complex(phases.cos(), phases.sin()).mean().abs().item()


def analyse_seed(cfg: dict, seed: int, p: int, data) -> dict:
    """Every Stage 2 measurement for one seed. Returns dict-of-lists-of-rows."""
    a = cfg["analysis"]
    batch = cfg["runtime"]["batch_size"]
    path = checkpoint_path(cfg, seed)
    model, ck = load_checkpoint(path)
    if ck["step"] != cfg["source"]["step"] or ck["seed"] != seed:
        raise RuntimeError(f"{path} holds step {ck['step']} seed {ck['seed']}, expected {cfg['source']['step']} / {seed}")

    # --- 1. embedding spectrum and key frequencies ------------------------------
    spectrum = embedding_spectrum(model.W_E, p)
    key = key_frequencies(spectrum, a["key_frequency_threshold"])
    key_sorted = sorted(key)
    spectrum_rows = [
        {"seed": seed, "frequency": k, "fraction": f"{spectrum[k].item():.10g}", "is_key": int(k in key)}
        for k in range(len(spectrum))
    ]
    key_row = {
        "seed": seed,
        "threshold": a["key_frequency_threshold"],
        "n_key": len(key),
        "frequencies": " ".join(str(k) for k in key_sorted),
        "fraction_of_norm": f"{sum(spectrum[k].item() for k in key):.10g}",
        "constant_fraction": f"{spectrum[0].item():.10g}",
        "largest_excluded": f"{max((spectrum[k].item() for k in range(1, len(spectrum)) if k not in key), default=0.0):.10g}",
    }

    # --- 2. Fourier-space ablation of the logits --------------------------------
    # Dropping the "=" output column must not change any prediction; check, don't assume.
    x, _ = make_cayley_table(p)
    with torch.no_grad():
        eq_argmax = sum(
            (model(x[i : i + batch], last_only=True).argmax(-1) == p).sum().item()
            for i in range(0, x.shape[0], batch)
        )
    if eq_argmax:
        raise RuntimeError(f"seed {seed}: '=' is the argmax for {eq_argmax} inputs; slicing to :p would change predictions")

    clean = center_logits(logit_grid(model, p, batch))
    coef = fourier_2d(clean, p)
    masks = frequency_masks(p, key)
    masks["cross_terms"] = ~(masks["key_block"] | masks["ablate_key"])

    ablation_rows = [{"seed": seed, "condition": "clean", "n_key": len(key),
                      "variance_retained": 1.0, **evaluate_logit_grid(clean, data)}]
    for name in ("key_diag", "key_block", "ablate_key", "cross_terms"):
        grid = inverse_fourier_2d(coef * masks[name][:, :, None].to(coef.dtype), p)
        ablation_rows.append(
            {
                "seed": seed,
                "condition": name,
                "n_key": len(key),
                "variance_retained": f"{variance_fraction(coef, masks[name]):.10g}",
                **evaluate_logit_grid(grid, data),
            }
        )
    # The three partition the spectrum except that the constant sits in two of them.
    accounting = (
        variance_fraction(coef, masks["key_block"])
        + variance_fraction(coef, masks["ablate_key"])
        + variance_fraction(coef, masks["cross_terms"])
        - variance_fraction(coef, torch.eye(p, dtype=torch.bool) & (torch.arange(p)[:, None] == 0))
    )
    if abs(accounting - 1.0) > 1e-8:
        raise RuntimeError(f"seed {seed}: variance accounting is {accounting}, expected 1")

    # --- 3. MLP neurons as functions of (a + b) ---------------------------------
    acts = mlp_activations(to_hooked_transformer(model), p, batch)
    fit = neuron_sum_fit(acts, p, key_sorted)
    sf = neuron_frequency_fit(acts, p, key_sorted)
    fve, dead = fit["fve"], fit["dead"]
    sf_fve = sf["best_fve"]
    alive = ~dead
    # A neuron belongs to the frequency explaining most of its variance overall, not to
    # the one with the largest a + b component: the single-frequency fit is the weaker,
    # better-posed criterion, and it is the one the paper's claim is about.
    dominant = sf["best"]
    q = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    empty = torch.full((3,), float("nan"), dtype=torch.float64)
    quart = torch.quantile(fve[alive], q) if alive.any() else empty
    sf_quart = torch.quantile(sf_fve[alive], q) if alive.any() else empty
    threshold = a["neuron_explained_threshold"]
    summary_row = {
        "seed": seed,
        "n_key": len(key),
        "n_neurons": int(fve.numel()),
        "n_dead": int(dead.sum()),
        # sum_*: variance explained by cos/sin(w_k(a + b)) alone -- the strong claim.
        "sum_fve_mean": f"{fve[alive].mean().item():.10g}",
        "sum_fve_q25": f"{quart[0].item():.10g}",
        "sum_fve_median": f"{quart[1].item():.10g}",
        "sum_fve_q75": f"{quart[2].item():.10g}",
        "sum_fve_min": f"{fve[alive].min().item():.10g}",
        "sum_fve_max": f"{fve[alive].max().item():.10g}",
        # single_*: variance explained by everything at one frequency -- the weak claim.
        "single_fve_mean": f"{sf_fve[alive].mean().item():.10g}",
        "single_fve_q25": f"{sf_quart[0].item():.10g}",
        "single_fve_median": f"{sf_quart[1].item():.10g}",
        "single_fve_q75": f"{sf_quart[2].item():.10g}",
        "single_fve_min": f"{sf_fve[alive].min().item():.10g}",
        "single_fve_max": f"{sf_fve[alive].max().item():.10g}",
        "explained_threshold": threshold,
        "n_sum_explained": int((fve >= threshold).sum()),
        "n_single_explained": int((sf_fve >= threshold).sum()),
    }
    frequency_rows = []
    for i, k in enumerate(key_sorted):
        owned = alive & (dominant == i)
        frequency_rows.append(
            {
                "seed": seed,
                "frequency": k,
                "n_neurons": int(owned.sum()),
                "single_fve_median": f"{sf['fve'][i][owned].median().item():.10g}" if owned.any() else "",
                "sum_fve_median": f"{fve[owned].median().item():.10g}" if owned.any() else "",
                # Near 0 means the phases of this frequency's neurons are spread around
                # the circle, as the clock algorithm predicts; near 1 would contradict it.
                "phase_resultant": f"{circular_resultant(fit['phase'][i][owned]):.10g}",
            }
        )

    return {
        "spectrum": spectrum_rows,
        "key": [key_row],
        "ablation": ablation_rows,
        "neuron_summary": [summary_row],
        "neuron_frequencies": frequency_rows,
        "misclassified": [{"seed": seed, **r} for r in misclassified_pairs(clean, data)],
        "_key_sorted": key_sorted,
        "_fit": fit,
        "_sf": sf,
        "_acts": acts,
    }


def write_reference_detail(cfg: dict, seed: int, p: int, result: dict) -> None:
    """Per-neuron CSVs for the reference seed, including the heatmap figure's data."""
    key_sorted, fit, acts = result["_key_sorted"], result["_fit"], result["_acts"]
    sf = result["_sf"]
    fve, dead = fit["fve"], fit["dead"]
    dominant = sf["best"]

    write_csv(
        OUT_DIR / f"neurons_seed{seed}.csv",
        [
            {
                "seed": seed,
                "neuron": n,
                "dead": int(dead[n]),
                "frequency": key_sorted[dominant[n]],
                "single_fve": f"{sf['best_fve'][n].item():.10g}",
                "sum_fve": f"{fve[n].item():.10g}",
                "sum_fve_at_frequency": f"{fit['fve_per_freq'][dominant[n], n].item():.10g}",
                "phase_at_frequency": f"{fit['phase'][dominant[n], n].item():.10g}",
            }
            for n in range(fve.numel())
        ],
    )
    write_csv(
        OUT_DIR / f"neuron_detail_seed{seed}.csv",
        [
            {
                "seed": seed,
                "neuron": n,
                "frequency": k,
                "single_fve": f"{sf['fve'][i, n].item():.10g}",
                "sum_fve": f"{fit['fve_per_freq'][i, n].item():.10g}",
                "phase": f"{fit['phase'][i, n].item():.10g}",
                "amplitude": f"{fit['amplitude'][i, n].item():.10g}",
            }
            for n in range(fve.numel())
            for i, k in enumerate(key_sorted)
        ],
    )
    # The figure plots one neuron over the whole (a, b) grid, so its activations have to
    # live in a CSV -- figures never load checkpoints.
    best = int(torch.argmax(fve).item())
    print(f"  heatmap neuron: {best} (fve {fve[best].item():.4f}, "
          f"frequency {key_sorted[dominant[best]]})")
    write_csv(
        OUT_DIR / f"neuron_heatmap_seed{seed}.csv",
        [
            {"seed": seed, "neuron": best, "a": a, "b": b, "activation": f"{acts[a, b, best].item():.6g}"}
            for a in range(p)
            for b in range(p)
        ],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=str, help="override the config's seeds, e.g. 0 or 0-4")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(cfg["runtime"]["num_threads"])
    torch.use_deterministic_algorithms(True)
    if cfg["runtime"]["device"] != "cpu":
        raise RuntimeError("Stage 2 analysis is CPU-only; set runtime.device: cpu")

    base = load_config(REPO_ROOT / "configs" / f"{cfg['source']['config']}.yaml")
    p = base["data"]["p"]
    data = make_data(p, base["data"]["train_frac"], base["data"]["split_seed"])
    seeds = parse_seeds(args.seeds if args.seeds else str(cfg["source"]["seeds"]))
    reference = cfg["analysis"]["reference_seed"]

    collected: dict[str, list[dict]] = {k: [] for k in
                                        ("spectrum", "key", "ablation", "neuron_summary",
                                         "neuron_frequencies", "misclassified")}
    for seed in seeds:
        print(f"seed {seed}:", flush=True)
        result = analyse_seed(cfg, seed, p, data)
        for name in collected:
            collected[name] += result[name]
        row = result["key"][0]
        print(f"  key frequencies: {row['frequencies']}  "
              f"({row['n_key']}, carrying {float(row['fraction_of_norm']):.2%} of squared norm)")
        diag = next(r for r in result["ablation"] if r["condition"] == "key_diag")
        print(f"  key_diag retains {float(diag['variance_retained']):.2%} of logit variance, "
              f"test acc {diag['test_acc']:.4f}")
        if seed == reference:
            write_reference_detail(cfg, seed, p, result)

    write_csv(OUT_DIR / "embedding_spectrum.csv", collected["spectrum"])
    write_csv(OUT_DIR / "key_frequencies.csv", collected["key"])
    write_csv(OUT_DIR / "ablation.csv", collected["ablation"])
    write_csv(OUT_DIR / "neuron_summary.csv", collected["neuron_summary"])
    write_csv(OUT_DIR / "neuron_frequencies.csv", collected["neuron_frequencies"])
    if collected["misclassified"]:
        write_csv(OUT_DIR / "misclassified.csv", collected["misclassified"])
    else:
        print("  no misclassified test pairs in any seed")


if __name__ == "__main__":
    main()
