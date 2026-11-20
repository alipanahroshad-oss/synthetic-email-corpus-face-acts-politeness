# BERT — Overall Politeness Regression (OPR)

This directory contains the BERT implementation for document-level Overall Politeness Regression (OPR) in **A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**.

## Overview

We investigate whether features derived from predicted sentence-level Face Acts (PredFA) complement BERT representations of email text when predicting document-level politeness.

Each email has three continuous target scores. The table maps the dimension names used in the paper to their dataset columns.

| Politeness dimension | Dataset column |
|---|---|
| Directness vs. Indirectness | `Directness_vs_Indirectness__GOLD` |
| Positive Face Saving | `Structural_Politeness_and_Politeness_Markers__GOLD` |
| Negative Face Saving | `Tone_and_Overall_Consideration__GOLD` |

## Implementation

**Script:** `OPR_BERT_PredFA_gridsearch.py`

| Model family | Inputs |
|---|---|
| **Text** | The email text, encoded with `bert-base-uncased`. |
| **PredFA-only** | Document-level PredFA features, passed to a multilayer perceptron (MLP). |
| **Text + PredFA** | The BERT representation of the email combined with its document-level PredFA features. |

## Predicted Face Act Features

A trained sentence-only BERT FAC classifier produces a probability for each of nine FA labels. A single threshold, shared across the labels, is selected by maximizing sentence-level micro-F1 on the validation set. The script applies this threshold to obtain binary predictions.

For each email, the script counts how many sentences receive each label. A sentence with multiple predicted labels contributes to each corresponding count.

`HNeg+`, `HNeg-`, `HPos+`, `HPos-`, `Neutral`, `SNeg+`, `SNeg-`, `SPos+`, `SPos-`

The nine counts and four aggregates form a 13-dimensional document representation (HSPT-13).

| Aggregate | Definition |
|---|---|
| `sum_H` | Total count of the four hearer-oriented FA labels. |
| `sum_S` | Total count of the four speaker-oriented FA labels. |
| `sum_praise` | Total count of the four face-saving labels marked `+`. |
| `sum_threat` | Total count of the four face-threatening labels marked `-`. |

`Neutral` contributes only to its own count. All 13 features are standardized using means and standard deviations computed on the training partition. The same transformation is applied to validation and test emails.

## Datasets

The script uses these files:

- `data/corpus/email_text_gold_three_dimensions_politeness_score_with_seed_correct.csv`
- `data/corpus/sentences_with_golden_face_act.csv`

The split groups emails by their original generation seed, keeping related emails in the same partition. The target proportions are 70% training, 15% validation, and 15% test. The split random state is `42`.

All experiments use this fixed split across five training seeds: `13`, `21`, `42`, `87`, and `100`. Each PredFA experiment uses the FAC checkpoint from the corresponding OPR training seed.

## Training and Evaluation

The regressors use Smooth L1 (Huber) loss.

| Hyperparameter | Text and Text + PredFA | PredFA-only MLP |
|---|---|---|
| Learning rate | `2e-5`, `3e-5` | `1e-3`, `5e-4` |
| Maximum epochs | `3`, `5` | `40`, `60` |
| Batch size | `8` | `16`, `32` |
| Dropout | `0.1`, `0.2` | `0.1`, `0.2` |
| Weight decay | `0.01` | `1e-4` |
| Hidden dimension | `128` | `128`, `256` |

For each model family, hyperparameter selection uses validation Spearman’s rank correlation averaged across the three politeness dimensions.

On the fixed test partition, we report mean absolute error (MAE) and Spearman’s rank correlation (ρ) for each dimension. Macro-MAE and macro-ρ are the averages of the three corresponding per-dimension metrics. The paper reports means and standard errors across the five training seeds.

## Requirements

The script uses Python, PyTorch, Transformers, NumPy, pandas, SciPy, and scikit-learn. The text-based models require access to `bert-base-uncased`.

## Running the Experiments

Run all commands from the repository root. The examples use Linux or macOS shell syntax.

### BERT Text-only

The text-only baseline uses the email text and can run independently of the FAC checkpoints.

```bash
OPR_RUN_SEED=42 \
SPLIT_RANDOM_STATE=42 \
RUN_BERT_TEXT=1 \
RUN_PREDFA_MLP=0 \
RUN_BERT_PLUS_PREDFA=0 \
python opr/bert/OPR_BERT_PredFA_gridsearch.py
```

### PredFA-only and BERT + PredFA

Set `ASL_MODEL_DIR` to the checkpoint produced by the sentence-only BERT FAC classifier in `fac/bert/` for the corresponding training seed.

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

### Running all five seeds

Repeat the relevant command for seeds `13`, `21`, `42`, `87`, and `100`, keeping `SPLIT_RANDOM_STATE=42`. For text-only runs, vary `OPR_RUN_SEED`. For PredFA runs, also set `FAC_RUN_SEED` and `ASL_MODEL_DIR` to the matching seed and checkpoint. Use a separate `OUT_DIR` under `outputs/` for each run to preserve its results.

## Outputs

The script writes experimental results under `outputs/`.

| File | Contents |
|---|---|
| `opr_grid_search_results.csv` | Results for the evaluated hyperparameter configurations. |
| `opr_best_by_family.csv` | Validation-selected configurations and associated metrics. |
| `split_info.json` | Data split and run metadata. |

Run outputs include validation and test metrics and predicted politeness scores. PredFA runs also save sentence-level FA probabilities, the selected FAC threshold, and the document-level feature standardization parameters.

## Reproducibility

Supplementary Appendix F, Table II, lists the hyperparameter search grids used in the extended study.
