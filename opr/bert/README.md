# BERT — Overall Politeness Regression (OPR)

This directory contains the BERT implementation for document-level Overall Politeness Regression (OPR) in:

**A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**

## Overview

The Overall Politeness Regression (OPR) task predicts three continuous document-level politeness dimensions:

1. **Directness**
2. **Positive Face**
3. **Negative Face**

We investigate whether representations derived from predicted sentence-level Face Acts (PredFA) complement document-level text representations for politeness prediction.

### Politeness Dimensions and Dataset Columns

The three politeness dimensions correspond to the following columns in the dataset:

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

- **Multi-task regression:** Jointly predicts Directness, Positive Face, and Negative Face.
- **Single-task regression:** Trains a separate predictor for each politeness dimension.

## Predicted Face Act Features

The PredFA features are obtained from a trained sentence-level BERT Face Act classifier.

The implementation uses a 13-dimensional representation (HSPT-13).

### Nine Face Act Features

The nine Face Act labels are:

- HNeg+
- HNeg-
- HPos+
- HPos-
- Neutral
- SNeg+
- SNeg-
- SPos+
- SPos-

Predicted Face Act labels are aggregated into document-level features.

### Four Additional Features

The representation also includes four aggregate features:

- **H:** Sum of hearer-related Face Act counts.
- **S:** Sum of speaker-related Face Act counts.
- **Praise:** Sum of positive Face Act counts.
- **Threat:** Sum of negative Face Act counts.

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

The five training seeds are:

`13, 21, 42, 87, 100`

The document-level split remains fixed across training seeds.

## Training and Evaluation

### BERT Regressor

Backbone: `bert-base-uncased`

Loss function: Smooth L1 (Huber) loss.

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

Metrics are computed separately for:

1. Directness
2. Positive Face
3. Negative Face

Macro-averaged MAE and Spearman correlation are also computed across the three dimensions.

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

The following examples use Linux or macOS shell syntax.

### 1. BERT Text-only

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

### 2. PredFA-only and BERT + PredFA

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

The checkpoint is generated by training the sentence-only BERT Face Act classifier.

### 3. Multi-seed Evaluation

The experiments use five training seeds:

`13, 21, 42, 87, 100`

The grouped data split is kept fixed using:

```bash
SPLIT_RANDOM_STATE=42
```

For text-only experiments, vary `OPR_RUN_SEED`.

For paired FAC/OPR experiments, the FAC and OPR training seeds are matched.

The corresponding FAC checkpoint must be supplied through `ASL_MODEL_DIR` for each run.

The text-only OPR baseline can be run independently of the FAC checkpoints.

### 4. Multi-task and Single-task Regression

By default, the script executes both multi-task and single-task variants.

These settings can be controlled using:

```bash
DO_MULTI_TASK=1
DO_SINGLE_TASK=1
```

Set either variable to `0` to disable the corresponding training mode.

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
- PredFA feature normalization parameters

Results are organized by model family and experimental configuration.

The script generates summary files including:

- `opr_grid_search_results.csv`
- `opr_best_by_family.csv`
- `split_info.json`

Generated experimental outputs and trained checkpoints are not included in the GitHub repository.

The root `.gitignore` excludes the `outputs/` directory.

## Reproducibility

The implementation uses a fixed group-based document split while allowing the training random seed to vary.

The BERT Text-only configuration can run independently, whereas the PredFA-only and BERT + PredFA configurations require a previously trained sentence-level BERT FAC model.

The code supports both multi-task and single-task regression.

The experiments compare document text representations, predicted Face Act representations, and their combination to investigate the contribution of Face Act information to document-level politeness prediction.

This implementation forms part of the extended BERT–LLaMA comparison presented in the journal manuscript and is separate from the earlier Findings of EMNLP 2026 implementation.
