# grokking-dynamics

A small decoder-only transformer trained on modular addition mod 113, where the model reaches 100% training accuracy within a few hundred steps and then stays at low test accuracy for thousands more before generalization arrives abruptly. This repo reproduces that delayed-generalization curve across multiple seeds and then investigates what changes inside the network during the plateau — the window where every metric you'd normally watch is flat but the representation is evidently reorganizing. The task is deliberately tiny and fully specified: because the ground truth has known algebraic structure, claims about what the model encodes internally can be checked against the group structure rather than against intuition. Results are reported over multiple seeds with variance, since onset timing on this task is variable enough that single-run curves are misleading.

The implementation replicates Nanda et al. 2023, [*Progress Measures for Grokking via Mechanistic Interpretability*](https://arxiv.org/abs/2301.05217): a 1-layer transformer with no LayerNorm, biases, or dropout, trained on (a + b) mod 113.

## Status

| Stage | Goal | Status |
|---|---|---|
| 1 | Data, model, training: reproduce the grokking curve | **Done** |
| 2 | Reverse-engineer the learned algorithm (Fourier analysis of weights and activations) | **Done** |
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

## Stage 2 results

**The hypothesis under test.** Nanda et al.'s "clock" algorithm: the model embeds each number on a small
number of circles — embedding *a* as cos/sin(w·a) for a few frequencies w = 2πk/113 — multiplies the
*a*-terms and *b*-terms together in attention and the MLP, and reads the answer off as cos(w(a + b − c)),
which is largest exactly when c = a + b.

All ten seeds were analysed from their final (step 40,000) checkpoints, on CPU. A frequency counts as
**key** if it carries at least **1% of the embedding's squared Frobenius norm** — a threshold fixed in
[`configs/fourier.yaml`](configs/fourier.yaml) before looking at any spectrum, and reported alongside the
full spectrum rather than tuned to make the count come out nicely.

### The embedding concentrates on a handful of frequencies

![Embedding DFT for seed 0](figures/fig03_embedding_spectrum.png)

Every seed puts almost all of its embedding norm on **4 or 5 frequencies** (five in seven seeds, four in
seeds 1, 4 and 6), carrying **90.4% to 99.0%** of the squared norm. The threshold is not doing delicate
work: the largest *excluded* frequency is between 0.13% and 0.96% across seeds, so nothing sits ambiguously
on the line, though seeds 2, 4 and 6 come closest (0.82–0.96%).

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Key frequencies | 5 17 24 34 50 | 5 17 32 48 | 4 5 20 27 40 | 6 21 25 26 44 | 21 33 46 56 | 5 29 42 45 48 | 4 5 8 35 | 14 23 30 33 45 | 3 6 21 26 51 | 17 19 34 51 56 |
| Share of squared norm | 98.9% | 98.2% | 93.9% | 97.9% | 90.4% | 99.0% | 97.3% | 98.8% | 98.7% | 98.9% |

**No two seeds choose the same set.** No frequency appears in all ten runs; the most common (k = 5) appears
in five. Which frequencies a model uses is an initialization lottery — but *how many*, and the structure
built on them, is not. The split is fixed across seeds, so this is caused by initialization alone.

### The logits are built out of those frequencies

![Fourier-space ablation](figures/fig05_ablation.png)

Logits are centred over the answer axis first, since softmax ignores a per-example constant and leaving it
in would inflate every number below.

| Kept | Logit variance retained | Test accuracy |
|---|---|---|
| Key frequencies, matched on both input axes | **98.5% ± 1.0%** (min 95.9%) | **100% in every seed** |
| Key frequencies, any pairing | 98.7% ± 0.8% | 100% in every seed |
| Cross terms only | 0.5% | 1.7% |
| Everything *except* the key frequencies | 1.2% | 2.6% |

Reconstructing the logits from the key frequencies **matched on both axes** — the components
cos(w(a + b − c)) actually expands into — retains 98.5% of the variance and full test accuracy. Allowing
*any* pairing of key frequencies adds only 0.2 points, so the computation really does live on the
matched-frequency diagonal rather than using cross-frequency terms. Removing the key frequencies collapses
the model to 2.6% test accuracy (chance is 0.88%).

### Neurons work at a single frequency, but are not functions of a + b alone

![A representative neuron](figures/fig04_neuron_heatmap.png)

This is where the simple form of the hypothesis fails, and the failure is informative.

- Counting **every term at a neuron's best single frequency** (terms in *a*, in *b*, in *a + b*, in
  *a − b*), the median neuron has **94–97%** of its variance explained, and **292–465 of 512** neurons per
  seed clear 90%.
- Counting **only cos/sin(w(a + b))**, the median falls to **12–18%**, and **zero of 512** neurons in any
  seed clear 90%.

That gap is expected rather than a contradiction. A neuron computing cos(w·a)·cos(w·b) is *entirely*
single-frequency, yet only half a function of a + b, because
cos(w·a)cos(w·b) = ½[cos(w(a + b)) + cos(w(a − b))]. The a − b half is real structure that cancels across
the population when the MLP is read out, not inside any individual neuron. Consistent with that, neurons
sharing a frequency have phases spread around the circle (mean circular resultant 0.12, max 0.49, where 0
is uniform and 1 would be identical phases).

### Seed 2's 24 errors

Seed 2 is the one seed that never reaches 100%. Its 24 wrong test pairs
([`misclassified.csv`](results/02_fourier/misclassified.csv)) are sharply clustered: they involve only
**12 distinct values** of *a* or *b*, and land on only **8 distinct correct answers**, four of them with
a = b. The defect is localized in the circuit rather than spread over the input space.

### Verification

- **Number of key frequencies:** 4 or 5 per seed, at a 1% threshold declared in advance.
- **Fraction of clean logit variance the key-frequency-only model retains:** 98.5% ± 1.0% across seeds,
  at 100% test accuracy in all ten.
- **Does it match the paper?** Qualitatively yes: a sparse, single-digit set of frequencies, with the
  computation concentrated on matched frequencies and collapsing when they are removed. The one place our
  result is weaker than a naive reading of the paper is at the neuron level — individual neurons are not
  well-approximated by cos/sin(w(a + b)) alone, for the algebraic reason above. We have **not** re-derived
  the paper's exact per-run numbers for a quantitative comparison, so "matches" here means structural
  agreement, not a matched statistic.

## Next stages

- **Stage 3:** track restricted and excluded loss across training checkpoints, locate the memorization →
  circuit-formation → cleanup phases for each seed, and test whether the circuit forms before the visible
  test-accuracy jump.

## Repository layout

```
configs/           one YAML per experiment (every hyperparameter lives here)
src/data.py        dataset and seeded train/test split
src/model.py       from-scratch transformer + to_hooked_transformer() (TransformerLens port)
src/train.py       full-batch training loop, CSV logging, checkpointing, determinism settings
src/fourier.py     Fourier basis, embedding DFT, neuron fits, Fourier-space ablation (Stage 2)
experiments/       one script per numbered experiment, plus regenerate_checkpoints.py
results/           raw metrics as CSV (committed), environment record, run logs
figures/           make_all.py regenerates every figure from results/ without retraining
colab/             notebooks and bundling script for running on a Colab GPU
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

# Stage 2 Fourier analysis: reads the final checkpoint of each seed, writes results/02_fourier/.
# CPU only, a few minutes; needs checkpoints on disk (see regenerate_checkpoints.py below).
python experiments/02_fourier.py --config configs/fourier.yaml

# figures and milestone table, from committed CSVs only (no training)
python figures/make_all.py

# checkpoints are not committed: regenerate them, verifying the CSVs reproduce byte-for-byte
python experiments/regenerate_checkpoints.py --all --jobs 2
```

Outputs:
- **Metrics:** `results/{experiment}/{config}_seed{k}.csv`. Columns: step, train/test loss, train/test accuracy,
  and the L2 norm of all parameters. A row is written every 100 steps and flushed immediately.
- **Checkpoints:** `checkpoints/{experiment}/{config}_seed{k}/stepNNNNNN.pt`. Two schedules, unioned:
  - **~20 log-spaced steps plus step 0**, saved in full, with optimizer and RNG state (~2.6 MB each).
  - **Every `train.checkpoint_every` steps up to `train.checkpoint_every_until`**, saved weights-only
    (~0.87 MB each). Log spacing straddles the transition — seed 0 groks between steps 5,000 and 6,000,
    entirely inside the 4,297 → 7,506 gap — so the progress measures in Stage 3 need uniform resolution
    through it. These are weights-only because restricted and excluded loss only run the model forward.

  The baseline config uses 250 and 15,000, giving 81 checkpoints per seed (21 full + 60 weights-only,
  ~107 MB per seed). Everything has settled well before step 15,000; the latest seed groks at 14,300.
  Checkpoint saving does not touch training: recomputing train/test loss, accuracy and parameter norm
  from all 320 checkpoints that land on the logged step grid reproduces the committed CSV rows exactly
  (zero difference in every column, on the GPU that produced them).

### On Colab

1. Run `python colab/make_bundle.py` to build `colab/grokking-dynamics.zip`. The bundle includes the
   committed CSVs in `results/`, which `regenerate_checkpoints.py` needs in order to verify anything.
2. At colab.research.google.com, use File → Upload notebook. Choose a T4 GPU runtime and Run all, then
   upload the zip when asked.
   - `colab/stage1_colab.ipynb` trains the baseline and control from scratch and draws the figures.
     Its last cell downloads `stage1_results.zip`; unzip it into the repo root and move its `logs/`
     folder to `results/01_baseline/colab_logs/`.
   - `colab/stage3_checkpoints.ipynb` re-runs the ten baseline seeds to produce the dense checkpoints,
     re-verifying as it goes: it byte-compares the regenerated metrics against the committed CSVs, then
     recomputes metrics from the saved checkpoints and compares those too. It downloads one zip per
     seed, so a dropped download costs only that seed.

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
