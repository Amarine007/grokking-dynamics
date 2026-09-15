# grokking-dynamics

A small decoder-only transformer trained on modular addition mod 113, where the model reaches 100% training accuracy within a few hundred steps and then stays at low test accuracy for thousands more before generalization arrives abruptly. This repo reproduces that delayed-generalization curve across multiple seeds and then investigates what changes inside the network during the plateau — the window where every metric you'd normally watch is flat but the representation is evidently reorganizing. The task is deliberately tiny and fully specified: because the ground truth has known algebraic structure, claims about what the model encodes internally can be checked against the group structure rather than against intuition. Results are reported over multiple seeds with variance, since onset timing on this task is variable enough that single-run curves are misleading.

The implementation replicates Nanda et al. 2023, [*Progress Measures for Grokking via Mechanistic Interpretability*](https://arxiv.org/abs/2301.05217): a 1-layer transformer with no LayerNorm, biases, or dropout, trained on (a + b) mod 113.

## Status

| Stage | Goal | Status |
|---|---|---|
| 1 | Data, model, training: reproduce the grokking curve | **Done** |
| 2 | Reverse-engineer the learned algorithm (Fourier analysis of weights and activations) | Not started |
| 3 | Progress measures (restricted / excluded loss) and the three training phases | Not started |

## Stage 1 results

**Setup:** all 12,769 pairs (a, b) with p = 113; a fixed seeded 30% / 70% train/test split; full-batch
AdamW (lr 1e-3, betas (0.9, 0.98), **weight decay 1.0**); 40,000 steps; 10 seeds, which differ only in
initialization. Runs were done on a Colab T4 GPU (versions in
[`results/01_baseline/ENVIRONMENT.txt`](results/01_baseline/ENVIRONMENT.txt)). Metrics were logged every 100 steps.

![Train and test accuracy across 10 seeds](figures/fig01_accuracy_across_seeds.png)

- **Memorization:** training accuracy passes 99% by step 200 in every seed.
- **Plateau:** test accuracy stays low for thousands of steps, but not flat at chance (1/113 ≈ 0.9%). It is 2–18%
  at step 1,000 across seeds and creeps upward before the jump.
- **Grokking:** test accuracy passes 90% between steps 4,800 and 14,300 (median 6,600), and no seed falls back
  below 90% afterwards.
- **Weight norm:** it rises to about 60 during memorization and falls to about 30 as the model generalizes.

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Train acc > 99% (step) | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 |
| Test acc > 90% (step) | 5,600 | 5,200 | 6,300 | 5,400 | 8,600 | 4,800 | 8,600 | 8,000 | 6,900 | 14,300 |
| Final test acc | 100% | 100% | 99.73% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

Seed 2 settles at 99.73% (about 24 of 8,939 test pairs wrong) from step ~20,000 onward. This is stable, not a
training instability; the misclassified pairs have not been examined yet.

### Control: weight decay 0

![Weight decay 1 vs 0, same seed and split](figures/fig02_weight_decay_control.png)

With weight decay set to 0 (same seed and split), the model memorizes by step 200 but **never generalizes**.
Test accuracy peaks at 2.5% and ends at 2.4%, while the weight norm keeps growing (42 → 74). Weight decay is what
drives grokking here.

### Verification
- **Reproducibility:** re-training seed 0 from scratch produces a byte-identical metrics CSV (on the same GPU and
  software versions).
- **TransformerLens port:** the ported model's logits match the from-scratch model to within 2.9e-5 on the trained
  model, over all inputs (the tolerance is 1e-4).
- **Tests:** the full suite (split determinism, seed reproducibility, weight port) passes on the GPU after training.

## Next stages

- **Stage 2:** find the key frequencies in the embedding with a discrete Fourier transform; test whether MLP
  neurons behave like cos/sin of w(a + b); and ablate the logits in Fourier space (keep only the key frequencies,
  then remove them) to test whether those frequencies carry the computation.
- **Stage 3:** track restricted and excluded loss across training checkpoints, locate the memorization →
  circuit-formation → cleanup phases for each seed, and test whether the circuit forms before the visible
  test-accuracy jump.

## Repository layout

```
configs/           one YAML per experiment (every hyperparameter lives here)
src/data.py        dataset and seeded train/test split
src/model.py       from-scratch transformer + to_hooked_transformer() (TransformerLens port)
src/train.py       full-batch training loop, CSV logging, checkpointing, determinism settings
experiments/       one script per numbered experiment, plus regenerate_checkpoints.py
results/           raw metrics as CSV (committed), environment record, run logs
figures/           make_all.py regenerates every figure from results/ without retraining
colab/             notebook and bundling script for running on a Colab GPU
tests/             split determinism, seed reproducibility, TransformerLens port
```

## Setup

```
pip install -r requirements.txt
python -m pytest tests
```

## Running

The configs set `runtime.device: cuda`, so training needs a CUDA GPU. To run on a CPU instead, copy a config
under a new name with `device: cpu`. CPU and GPU results are not bit-identical, so they must not overwrite each
other's CSVs.

```
# one seed / many seeds (--jobs = seeds run in parallel processes)
python experiments/01_baseline.py --config configs/baseline.yaml --seed 0
python experiments/01_baseline.py --config configs/baseline.yaml --seeds 0-9 --jobs 2

# weight-decay control
python experiments/01_baseline.py --config configs/baseline_wd0.yaml --seed 0

# figures and milestone table, from committed CSVs only (no training)
python figures/make_all.py

# checkpoints are not committed: regenerate them, verifying the CSVs reproduce byte-for-byte
python experiments/regenerate_checkpoints.py --all --jobs 2
```

Outputs:
- **Metrics:** `results/{experiment}/{config}_seed{k}.csv`. Columns: step, train/test loss, train/test accuracy,
  and the L2 norm of all parameters. A row is written every 100 steps and flushed immediately.
- **Checkpoints:** `checkpoints/{experiment}/{config}_seed{k}/stepNNNNNN.pt`, holding the model and optimizer
  state, at step 0 plus ~20 log-spaced steps.

### On Colab

1. Run `python colab/make_bundle.py` to build `colab/grokking-dynamics.zip`.
2. At colab.research.google.com, use File → Upload notebook → `colab/stage1_colab.ipynb`. Choose a T4 GPU runtime
   and Run all, then upload the zip when asked.
3. The last cell downloads `stage1_results.zip`. Unzip it into the repo root and move its `logs/` folder to
   `results/01_baseline/colab_logs/`.

## Design decisions

- **Architecture:** no LayerNorm, no biases, no dropout, no learning-rate schedule. The TransformerLens port
  zeroes every TransformerLens bias.
- **No warmup:** the paper's reference code used a 10-step linear LR warmup. It is omitted here to keep the
  optimizer schedule-free.
- **Output vocabulary is p + 1**: the `=` token is a possible, never-correct output, matching the reference code.
  Accuracy is the argmax over all p + 1 logits.
- **Loss is computed in float64**, as in the reference code, so very small losses don't underflow in float32.
- **Initialization** follows the reference code: all weights ~ N(0, 1/d_model) except W_U ~ N(0, 1/d_vocab).
- **Split:** `data.split_seed` is fixed across training seeds, so `--seed` varies only the initialization.
- **Device** comes from the config (`cpu` or `cuda`). There is no silent fallback, because results must come from
  the device their config names.
- **Speed:** training evaluates only the final position, and Q/K/V are computed from pre-projected embedding
  tables. Both are exact identities for a 1-layer causal model and are covered by tests.
- **Reproducibility:** deterministic algorithms, a fixed thread count, TF32 disabled, and deterministic cuBLAS.
  Byte-identical CSVs are expected on the same hardware with the same library versions.

## License

MIT (see [LICENSE](LICENSE)).
