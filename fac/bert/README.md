
# BERT — Face Act Classification (FAC)

This directory contains the BERT implementations for sentence-level multi-label Face Act Classification (FAC) used in the extended study:

**A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**

## Overview

We investigate the effect of conversational context on Face Act Classification using BERT (bert-base-uncased).

Each sentence can receive one or more of nine Face Act labels:

HNeg+, HNeg-, HPos+, HPos-, Neutral, SNeg+, SNeg-, SPos+, SPos-

## Experimental Configurations

Five configurations are implemented:

| Configuration | Script | Max tokens |
|---|---|---|
| Sentence — No Context | FAC_no_context_BERT_gridsearch.py | 128 |
| Sentence — Mail | FAC_context_mail_BERT_gridsearch.py | 256 |
| Sentence — Pair | FAC_context_pair_BERT_gridsearch.py | 256 |
| Sequence Labeling — Mail | FAC_seqlabel_mail_BERT_gridsearch.py | 512 |
| Sequence Labeling — Pair | FAC_seqlabel_pair_BERT_gridsearch.py | 512 |

### Context Definitions

- **No Context:** Only the target sentence is used.
- **Sentence — Mail:** The target sentence is classified using preceding sentences from the same email.
- **Sentence — Pair:** Request targets use their preceding request sentences; reply targets additionally use the full request email.
- **Sequence Labeling — Mail:** Sentences within an email are jointly encoded and classified.
- **Sequence Labeling — Pair:** The request and reply are jointly encoded and classified.

## Data

All scripts use the same fixed data split:

- `data/splits/train_seed42.csv`
- `data/splits/val_seed42.csv`
- `data/splits/test_seed42.csv`

The split is grouped by the original email generation seed.

Training runs use five random seeds:

13, 21, 42, 87, 100

## Training Configuration

Backbone: `bert-base-uncased`

Loss: Asymmetric Loss (ASL)

Fixed settings:
- Weight decay: 0.01
- Early stopping patience: 2
- ASL gamma positive: 0.0
- ASL gamma negative: 4.0
- ASL clipping: 0.05

Hyperparameter grid:
- Learning rates: 1e-5, 2e-5, 3e-5
- Training epochs: 3, 5, 8
- Dropout: 0.1, 0.2

Batch sizes:
- Sentence configurations: 8, 16
- Sequence-labeling configurations: 2, 4

Each configuration searches 36 hyperparameter combinations per training seed.

## Model Selection and Evaluation

Model selection uses validation micro-F1.

A global decision threshold is selected on the validation set from 0.05 to 0.95 in increments of 0.01.

The selected configuration and threshold are then evaluated on the fixed test set.

Evaluation includes:
- Micro-F1
- Micro-precision
- Micro-recall
- Class-wise performance
- Request/reply performance

## Running the Experiments

The scripts require Python and compatible installations of PyTorch, Transformers, Accelerate, NumPy, pandas, and scikit-learn.

Run commands from the repository root so that the relative data paths resolve correctly.

For example, on Linux or macOS:

```bash
RUN_SEED=42 python fac/bert/FAC_no_context_BERT_gridsearch.py
```

To run all five seeds for the no-context configuration:

```bash
for s in 13 21 42 87 100; do
    RUN_SEED=$s python fac/bert/FAC_no_context_BERT_gridsearch.py
done
```

The remaining configurations can be executed by replacing the script filename with the corresponding filename in the table above.

## Outputs

Scripts write experimental artifacts under `outputs/`, including hyperparameter-search summaries, selected configurations, evaluation metrics, and test predictions.

Model checkpoints and generated output files are not included in this directory.

## Reproducibility Notes

These scripts correspond to the extended BERT experiments rather than the earlier EMNLP implementation.

Exact software versions and execution-environment details should be recorded to support full reproduction of the reported results.
