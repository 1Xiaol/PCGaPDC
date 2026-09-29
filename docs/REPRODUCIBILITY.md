# Reproduction Workflow

## Overview

This workflow keeps model selection, final evaluation, kernel calibration, and judge aggregation on explicit and reproducible data axes.

## Execution Order

1. Fix the dataset splits, text encoder, user axis, and item-embedding version.
2. Use `configs/training/method.json` for the paper-level method settings. Keep run-specific budgets, seeds, and runtime controls outside the public method configuration.
3. Calibrate a chordal bandwidth separately for each dataset using Train items only.
4. Train the history-conditioned and dual-view conditioned transports with an explicit model config, method config, and runtime arguments. Use validation data for model selection.
5. Freeze the selected dual-view transport and optimize conditional source calibration with the same method config and explicit runtime arguments.
6. Decode every internal variant with the same frozen embedding-conditioned caption decoder and decoding parameters.
7. Re-encode generated captions with the same text encoder used for histories and targets.
8. Evaluate all methods on the same users, targets, generated-sample count, and metric implementation.
9. Compute user-macro metrics and paired user-level bootstrap intervals from per-user records.
10. Evaluate judge comparisons under both presentation orders and aggregate only complete target intersections.

## Run Metadata

Record the random seeds, dependency versions, configuration files, input hashes, calibrated Train bandwidth, user count, generated-sample count, effective metric denominators, and validation-selection criterion for each run.

## Caption-Decoder Conditioning Checks

In addition to validation NLL, compare correct, shuffled, and null semantic inputs. Report semantic reconstruction, NLL or perplexity, and template-collapse statistics to verify that caption generation depends on the supplied semantic representation rather than language-model priors alone.
