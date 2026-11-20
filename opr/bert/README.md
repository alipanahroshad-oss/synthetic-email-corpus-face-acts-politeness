# BERT — Overall Politeness Regression (OPR)

This directory contains the BERT implementation for document-level Overall Politeness Regression (OPR) in **A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**.

## Overview

We investigate whether features derived from predicted sentence-level Face Acts (PredFA) complement BERT representations of email text when predicting document-level politeness.

Each email has three continuous target scores. The table maps the dimension names used in the paper to the corresponding dataset columns.

| Politeness dimension | Dataset column |
|---|---|
| Directness vs. Indirectness | `Directness_vs_Indirectness__GOLD` |
| Positive Face Saving | `Structural_Politeness_and_Politeness_Markers__GOLD` |
| Negative Face Saving | `Tone_and_Overall_Consideration__GOLD` |

## Implementation

**Script:** `OPR_BERT_PredFA_gridsearch.py`

| Model family | Input and prediction method |
|---|---|
| **Text** | Encodes the email text with `bert-base-uncased` to predict politeness scores. |
| **PredFA-only** | Uses a multilayer perceptron (MLP) to predict scores from the email’s aggregated PredFA features. |
| **Text + PredFA** | Combines the BERT representation of the email with its aggregated PredFA features. |

Each family supports two training modes:

- **Multi-task regression:** One model predicts all three dimensions jointly.
- **Single-task regression:** A separate model is trained for each dimension.

## Predicted Face Act Features

A trained sentence-only BERT FAC classifier produces a probability for each of nine FA labels. A single threshold shared across all nine labels is selected by maximizing sentence-level micro-F1 on the validation set. This threshold converts the probabilities into binary predictions.

For each email, the script counts how many sentences receive each label. A sentence with multiple predicted labels contributes to each corresponding count.

`HNeg+`, `HNeg-`, `HPos+`, `HPos-`, `Neutral`, `SNeg+`, `SNeg-`, `SPos+`, `SPos-`

The nine counts and four aggregates form a 13-dimensional document representation (HSPT-13):

| Aggregate | Definition |
|---|---|
| `sum_H` | Total count of the four hearer-oriented FA labels. |
| `sum_S` | Total count of the four speaker-oriented FA labels. |
| `sum_praise` | Total count of the four face-saving (`+`) labels: `HNeg+`, `HPos+`, `SNeg+`, and `SPos+`. |
| `sum_threat` | Total count of the four face-threatening (`-`) labels: `HNeg-`, `HPos-`, `SNeg-`, and `SPos-`. |

`Neutral` has its own count and contributes to none of the four aggregates. The script standardizes all 13 features using means and standard deviations computed on the training partition and applies the same transformation to validation and test emails.

## Datasets

The script uses:

- `data/corpus/email_text_gold_three_dimensions_politeness_score_with_seed_correct.csv`
- `data/corpus/sentences_with_golden_face_act.csv`

The group-based split uses the original email generation seed to keep related emails in the same partition. The target proportions are 70% training, 15% validation, and 15% test. The split random state is `42`.

The five training seeds are `13`, `21`, `42`, `87`, and `100`. All runs use the same document split. For experiments using PredFA, each OPR run uses the FAC checkpoint from the matching training seed.

## Training and Evaluation

The regressors use Smooth L1 (Huber) loss. The Text and Text + PredFA families use `bert-base-uncased` as the text encoder.

### Hyperparameter grids

| Hyperparameter | Text and Text + PredFA | PredFA-only MLP |
|---|---|---|
| Learning rate | `2e-5`, `3e-5` | `1e-3`, `5e-4` |
| Maximum epochs | `3`, `5` | `40`, `60` |
| Batch size | `8` | `16`, `32` |
| Dropout | `0.1`, `0.2` | `0.1`, `0.2` |
| Weight decay | `0.01` | `1e-4` |
| Hidden dimension | `128` | `128`, `256` |

### Model selection and evaluation

For multi-task training, checkpoint selection uses validation Spearman’s correlation averaged across the three dimensions. For single-task training, each predictor is selected using validation Spearman’s correlation for its own dimension. Within each model family and training mode, the best hyperparameter configuration is selected by macro-averaged validation Spearman’s correlation.

On the fixed test partition, we report mean absolute error (MAE) and Spearman’s rank correlation (ρ) separately for each dimension. Macro-MAE and macro-ρ are the averages of the three corresponding per-dimension metrics. The paper summarizes performance as mean ± standard error across five training seeds.

## Requirements

The implementation requires Python and compatible versions of PyTorch, Transformers, NumPy, pandas, SciPy, and scikit-learn. The text-based models also require access to `bert-base-uncased`.

## Running the Experiments

Run all commands from the repository root. The examples use Linux or macOS shell syntax.

### 1. BERT Text-only

The text-only baseline can run independently of the FAC checkpoints.

```bash
OPR_RUN_SEED=42 \
SPLIT_RANDOM_STATE=42 \
RUN_BERT_TEXT=1 \
RUN_PREDFA_MLP=0 \
RUN_BERT_PLUS_PREDFA=0 \
python opr/bert/OPR_BERT_PredFA_gridsearch.py
```

### 2. PredFA-only and BERT + PredFA

Train the sentence-only BERT FAC classifier using the script in `fac/bert/`. Set `ASL_MODEL_DIR` to the resulting checkpoint directory for the corresponding training seed.

```bash
ASL_MODEL_DIR="/path/to/bert_fac_checkpoint" \
FAC_RUN_SEED=42 \
OPR_RUN_SEED=42 \
SPLIT_RANDOM_STATE=42 \
RUN_BERT_TEXT=0 \
RUN_PREDFA_MLP=1 \
RUN_BERT_PLUS_PREDFA=1 \
python opr/bert/OPR_BERT_PredFA_gridsearch.py
```

`ASL_MODEL_DIR` selects the checkpoint to load. `FAC_RUN_SEED` records its training seed in the run metadata.

### 3. Multi-seed evaluation

Repeat the relevant command for training seeds `13`, `21`, `42`, `87`, and `100`, keeping `SPLIT_RANDOM_STATE=42`. For text-only runs, vary `OPR_RUN_SEED`. For PredFA runs, also set `FAC_RUN_SEED` and `ASL_MODEL_DIR` to the matching FAC seed and checkpoint. Use a separate `OUT_DIR` under `outputs/` for each run to preserve its results.

### 4. Multi-task and single-task regression

Both modes are enabled by default. To run only one mode, add the following flags to the command before `python`:

| Training mode | Flags |
|---|---|
| Multi-task only | `DO_MULTI_TASK=1 DO_SINGLE_TASK=0` |
| Single-task only | `DO_MULTI_TASK=0 DO_SINGLE_TASK=1` |

## Outputs

The script writes experimental outputs under `outputs/`. Summary files include:

| File | Contents |
|---|---|
| `opr_grid_search_results.csv` | Results for the evaluated hyperparameter configurations. |
| `opr_best_by_family.csv` | Validation-selected configuration and associated metrics for each model family and training mode. |
| `split_info.json` | Data split and training-run metadata. |

Run outputs include validation and test metrics and predicted politeness scores. Runs using PredFA also save the sentence-level FA probabilities, validation threshold information, and document-level feature standardization parameters.

The root `.gitignore` excludes `outputs/`. Generated results and trained checkpoints are not included in the repository.

## Reproducibility

Supplementary Appendix F, Table II, lists the hyperparameter search grids used in the extended study. This implementation provides the BERT OPR experiments used in the journal manuscript’s comparison of BERT and LLaMA.
