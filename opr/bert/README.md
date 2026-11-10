
# BERT — Overall Politeness Regression (OPR)

This directory contains the BERT implementation for document-level Overall Politeness Regression (OPR) in:

**A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**

## Overview

The OPR task predicts three continuous document-level politeness dimensions:

1. Directness vs. Indirectness
2. Structural Politeness and Politeness Markers
3. Tone and Overall Consideration

We investigate whether representations derived from predicted sentence-level Face Acts (PredFA) complement document-level text representations for politeness prediction.

### Politeness Dimensions and Dataset Columns

The three politeness dimensions are represented by the following columns in the dataset:

```python
# "Directness": "Directness_vs_Indirectness__GOLD"
# "Positive Face": "Structural_Politeness_and_Politeness_Markers__GOLD"
# "Negative Face": "Tone_and_Overall_Consideration__GOLD"
```

The models predict all three dimensions, either jointly (multi-task) or independently (single-task).

## Implementation

**Script:** `OPR_BERT_PredFA_gridsearch.py`

Three model families are implemented:

| Model | Description |
|---|---|
| BERT Text-only | Predicts politeness scores from email text |
| PredFA-only | Uses aggregated predicted Face Act features with an MLP |
| BERT + PredFA | Combines document-level BERT representations with PredFA features |

Each model family supports:

- **Multi-task regression:** Jointly predicts all three politeness dimensions.
- **Single-task regression:** Trains separate predictors for each politeness dimension.

## Predicted Face Act Features

The PredFA features are obtained from a trained sentence-level BERT Face Act classifier.

The implementation uses a 13-dimensional representation (HSPT-13):

- Nine aggregated Face Act label counts:
  - HNeg+
  - HNeg-
  - HPos+
  - HPos-
  - Neutral
  - SNeg+
  - SNeg-
  - SPos+
  - SPos-

- Four additional aggregate features:
  - H
  - S
  - Praise
  - Threat

The Face Act classification threshold is selected using the validation set.

The current implementation uses hard predicted Face Act labels, with the threshold tuned to maximize validation micro-F1.

PredFA features are standardized using statistics computed from the training partition and applied to the validation and test partitions.

## Datasets

The script uses the following files:

- `data/corpus/email_text_gold_three_dimensions_politeness_score_with_seed_correct.csv`
- `data/corpus/sentences_with_golden_face_act.csv`

The document-level data is divided into training, validation, and test partitions using group-based splitting by generation seed.

- Training: 70%
- Validation: 15%
- Test: 15%
- Split random state: 42

Training random seeds:

`13, 21, 42, 87, 100`

The document-level split remains fixed across training seeds.

## Training and Evaluation

### BERT Regressor

Backbone: `bert-base-uncased`

Loss function: Smooth L1 (Huber) loss

Hyperparameter search:

- Learning rate: 2e-5, 3e-5
- Epochs: 3, 5
- Batch size: 8
- Dropout: 0.1, 0.2
- Weight decay: 0.01
- Hidden dimension: 128

### PredFA-only MLP

The PredFA-only MLP uses the following hyperparameter grid:

- Learning rate: 1e-3, 5e-4
- Epochs: 40, 60
- Batch size: 16, 32
- Dropout: 0.1, 0.2
- Weight decay: 1e-4
- Hidden dimension: 128, 256

### Evaluation Metrics

The models are evaluated using:

- Mean Absolute Error (MAE)
- Spearman's rank correlation (ρ)

Results are computed for each politeness dimension and averaged across the three dimensions.

Model selection is based on validation Spearman correlation.

## Requirements

The implementation requires Python and compatible versions of:

- PyTorch
- Transformers
- NumPy
- pandas
- SciPy
- scikit-learn

The experiments also require access to the pretrained `bert-base-uncased` model.

## Running the Experiments

Run all commands from the repository root so that the dataset paths resolve correctly.

### BERT Text-only

This configuration predicts document-level politeness scores using only email text.

It does not require a trained Face Act classifier.

```bash
OPR_RUN_SEED=42 \
SPLIT_RANDOM_STATE=42 \
RUN_BERT_TEXT=1 \
RUN_PREDFA_MLP=0 \
RUN_BERT_PLUS_PREDFA=0 \
python opr/bert/OPR_BERT_PredFA_gridsearch.py
```

### PredFA-only and BERT + PredFA

These configurations require a trained sentence-only BERT FAC checkpoint from `fac/bert/`.

Set `ASL_MODEL_DIR` to the corresponding checkpoint directory.

```bash
ASL_MODEL_DIR="/path/to/bert_fac_checkpoint" \
OPR_RUN_SEED=42 \
SPLIT_RANDOM_STATE=42 \
RUN_BERT_TEXT=0 \
RUN_PREDFA_MLP=1 \
RUN_BERT_PLUS_PREDFA=1 \
python opr/bert/OPR_BERT_PredFA_gridsearch.py
```

Replace `/path/to/bert_fac_checkpoint` with the actual FAC checkpoint path.

### Multi-seed Evaluation

The experiments use five training seeds:

`13, 21, 42, 87, 100`

The grouped data split is kept fixed using `SPLIT_RANDOM_STATE=42`.

For paired FAC/OPR experiments, the FAC and OPR training seeds are matched.

The text-only OPR baseline can be run independently of the FAC checkpoints.

## Outputs

The script automatically creates an output directory under `outputs/`.

Generated artifacts include:

- Hyperparameter grid-search results
- Best configuration per model family
- Validation and test evaluation metrics
- Predicted politeness scores
- Predicted Face Act probabilities
- Validation-based Face Act threshold information
- Data split information

Generated experimental outputs are not included in the repository.

The root `.gitignore` excludes the `outputs/` directory.

## Reproducibility

The implementation uses a fixed group-based document split while allowing the training random seed to vary.

The BERT Text-only system can run independently, whereas the PredFA-only and BERT + PredFA systems require a trained sentence-level FAC model.

The code supports both multi-task and single-task regression.

This implementation forms part of the extended BERT–LLaMA comparison presented in the journal manuscript and is separate from the earlier Findings of EMNLP 2026 implementation.
