# Collaborative Generative Riemannian Flow Matching

## Overview

This repository implements preference-distribution modeling with history-conditioned and dual-view conditioned Riemannian flow matching, conditional source calibration, embedding-conditioned caption decoding, and the evaluation protocols used by the method.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Install the optional caption-decoder dependencies when using a Qwen3.5 backbone:

```bash
pip install -e ".[caption-decoder]"
```

The pretrained text encoder, language model, datasets, and trained weights are not distributed in this repository. Supply them according to their respective licenses.

## Data Preparation

Transport training uses a dense NPZ file containing source samples, paired targets, observed-history embeddings, collaborative-evidence embeddings, and padding masks. Source calibration uses a related NPZ file with a target-validity mask. Evaluation uses a separate NPZ file with generated embeddings and ragged history and target arrays.

Public optimization parameters are stored in `configs/training/method.json`. Run-specific budgets, seeds, and runtime controls are supplied explicitly on the command line and are not embedded in the public method configuration.

See [docs/DATA.md](docs/DATA.md) for the complete array contracts.

Pack individual NPY arrays into a validated NPZ contract:

```bash
python scripts/prepare_arrays.py \
  --schema transport \
  --output data/train_episodes.npz \
  --array source=data/source.npy \
  --array target=data/target.npy \
  --array history=data/history.npy \
  --array history_mask=data/history_mask.npy \
  --array evidence=data/evidence.npy \
  --array evidence_mask=data/evidence_mask.npy
```

## Train Conditional Transport

Train the history-conditioned transport:

```bash
python scripts/train_transport.py \
  --input data/train_episodes.npz \
  --model-config configs/model/history_conditioned_rfm.json \
  --training-config configs/training/method.json \
  --updates <UPDATES> \
  --batch-size <BATCH_SIZE> \
  --seed <SEED> \
  --weight-decay <WEIGHT_DECAY> \
  --gradient-clip <GRADIENT_CLIP> \
  --log-interval <LOG_INTERVAL> \
  --output outputs/history_conditioned_transport.pt
```

Train the dual-view conditioned transport:

```bash
python scripts/train_transport.py \
  --input data/train_episodes.npz \
  --model-config configs/model/dual_view_conditioned_rfm.json \
  --training-config configs/training/method.json \
  --updates <UPDATES> \
  --batch-size <BATCH_SIZE> \
  --seed <SEED> \
  --weight-decay <WEIGHT_DECAY> \
  --gradient-clip <GRADIENT_CLIP> \
  --log-interval <LOG_INTERVAL> \
  --output outputs/dual_view_conditioned_transport.pt
```

Train conditional source calibration against the frozen dual-view transport:

```bash
python scripts/train_source_calibration.py \
  --input data/source_calibration.npz \
  --transport outputs/dual_view_conditioned_transport.pt \
  --model-config configs/model/conditional_source_calibration.json \
  --training-config configs/training/method.json \
  --updates <UPDATES> \
  --seed <SEED> \
  --learning-rate <LEARNING_RATE> \
  --heun-steps <HEUN_STEPS> \
  --weight-decay <WEIGHT_DECAY> \
  --gradient-clip <GRADIENT_CLIP> \
  --log-interval <LOG_INTERVAL> \
  --output outputs/source_calibration.pt
```

The public method configuration contains the paper-level transport and source-calibration settings. It intentionally does not contain run-specific update counts, seeds, selected checkpoints, or dataset-specific tuning outcomes.

Generate semantic samples for caption decoding or embedding evaluation:

```bash
python scripts/generate_embeddings.py \
  --input data/generation.npz \
  --transport outputs/dual_view_conditioned_transport.pt \
  --source-calibration outputs/source_calibration.pt \
  --output outputs/generated_embeddings.npz
```

The reusable transport, source-calibration, and caption-decoder modules are documented in [docs/METHOD_COMPONENTS.md](docs/METHOD_COMPONENTS.md).

## Calibrate the Evaluation Kernel

Calibrate a dataset-specific chordal Gaussian-RBF bandwidth using Train item embeddings only:

```bash
recdiffusion calibrate-bandwidth \
  --items data/train_item_embeddings.npy \
  --pairs 500000 \
  --output outputs/bandwidth.json
```

## Evaluate Generated Embeddings

```bash
recdiffusion evaluate \
  --input data/evaluation.npz \
  --bandwidth <TRAIN_CALIBRATED_BANDWIDTH> \
  --output outputs/metrics.json
```

The evaluator reports History-nearest, Target-best, angular Coverage, Future-mode coverage, Energy, and unbiased MMD. See [docs/PROTOCOL.md](docs/PROTOCOL.md) for definitions.

## Aggregate Judge Annotations

The aggregation script consumes completed annotation records and never calls a remote model:

```bash
python scripts/aggregate_judge.py \
  --input data/judge_annotations.json \
  --methods method_a method_b \
  --output outputs/judge_metrics.json
```

The number of users and candidates is inferred from the input. User-level confidence intervals use the configured bootstrap count.

## Render a Metric Table

```bash
python scripts/reproduce_tables.py \
  --method "Ours=outputs/ours_metrics.json" \
  --method "Baseline=outputs/baseline_metrics.json" \
  --output outputs/table.md
```

## Tests

```bash
pytest
python scripts/smoke_test.py
```
