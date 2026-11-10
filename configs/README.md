
# Experimental Configurations

This directory documents the experimental settings used in the extended study.

## Dataset Splitting

The dataset is divided into three subsets:

- Training: 70%
- Validation: 15%
- Testing: 15%

A group-based splitting strategy is applied using the original email generation seeds to prevent related examples from appearing in different subsets.

- Split random seed: 42
- Training seeds: 13, 21, 42, 87, 100

The same data split is maintained across experiments.

## Face Act Classification (FAC)

We evaluate five configurations:

1. Sentence classification — No Context
2. Sentence classification — Mail
3. Sentence classification — Pair
4. Sequence Labeling — Mail
5. Sequence Labeling — Pair

Backbones:
- BERT: bert-base-uncased
- LLaMA: meta-llama/Llama-3.2-1B with LoRA

Hyperparameter selection and decision-threshold tuning are performed using the validation set.

## Overall Politeness Regression (OPR)

The OPR experiments evaluate document-level politeness prediction using textual representations and additional linguistic features.

The target dimensions are:

- Directness vs. Indirectness
- Positive Face Saving
- Negative Face Saving

Evaluation metrics:
- Mean Absolute Error (MAE)
- Spearman's rank correlation (rho)

## Reproducibility

Exact hyperparameter grids, selected configurations, software dependencies, and execution commands will be added alongside the experimental scripts.
