# grokking-dynamics

Replication and mechanistic analysis of Nanda et al. 2023, *Progress Measures for
Grokking via Mechanistic Interpretability* (arXiv 2301.05217): a 1-layer transformer
trained on (a + b) mod 113.

## Setup

```
pip install -r requirements.txt
python -m pytest tests
```

## Running on Colab (how the committed results were produced)

The configs set `runtime.device: cuda`. To produce results on a Colab GPU:

1. `python colab/make_bundle.py` builds `colab/grokking-dynamics.zip`.
2. In Colab, use File → Upload notebook → `colab/stage1_colab.ipynb`. Pick a T4 GPU runtime, then Run all, and upload the zip when asked.
3. The last cell downloads `stage1_results.zip`. Unzip it into the repo root.

To run locally without a GPU, copy a config and set `device: cpu` under a new
config name. CPU and GPU results are not bit-identical, so they must not overwrite
each other's CSVs.

## Running

```
# one seed / many seeds (--jobs = seeds run in parallel processes)
python experiments/01_baseline.py --config configs/baseline.yaml --seed 0
python experiments/01_baseline.py --config configs/baseline.yaml --seeds 0-9 --jobs 4

# weight-decay control
python experiments/01_baseline.py --config configs/baseline_wd0.yaml --seed 0

# figures, from committed CSVs only (no training)
python figures/make_all.py

# checkpoints are not committed: regenerate them, verifying the CSVs reproduce byte-for-byte
python experiments/regenerate_checkpoints.py --all --jobs 4
```

Metrics: `results/{experiment}/{config}_seed{k}.csv` (step, train/test loss and
accuracy, L2 norm of all parameters; every 100 steps, flushed per row).
Checkpoints (model + optimizer state): `checkpoints/{experiment}/{config}_seed{k}/stepNNNNNN.pt`
at step 0 plus ~20 log-spaced steps.

## Decisions worth knowing

- **Architecture:** no LayerNorm, no biases, no dropout, no LR schedule. The
  TransformerLens port (`src.model.to_hooked_transformer`) zeroes every TL bias.
- **Output vocabulary is p + 1** (the `=` token is a possible, never-correct output),
  matching the paper's reference code. Accuracy is argmax over all p + 1 logits.
- **Loss is computed in float64** (`cross_entropy_high_precision`), as in the reference
  code, so very small losses don't underflow in float32.
- **Initialisation** follows the reference code: all weights ~ N(0, 1/d_model) except
  W_U ~ N(0, 1/d_vocab).
- **Split:** `data.split_seed` in the YAML is fixed across training seeds, so `--seed`
  varies only the initialisation.
- **Speed (CPU):** training evaluates only the final position (`last_only=True`), and
  Q/K/V are computed from pre-projected embedding tables. Both are exact identities
  for a 1-layer causal model, and both are covered by tests.
- **Reproducibility:** `torch.use_deterministic_algorithms(True)` plus a fixed
  `runtime.num_threads`. Byte-identical CSVs are expected on the same machine with the
  same library versions and thread count. Across different hardware or BLAS builds,
  expect float-level differences.
