# LLaMA — Overall Politeness Regression (OPR)

This directory contains the LLaMA implementation for document-level Overall Politeness Regression (OPR) in the extended study **A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**.

## Overview

We investigate whether representations derived from predicted sentence-level Face Acts (PredFA) complement document-level text representations for politeness prediction.

### Politeness Dimensions and Dataset Columns

The models predict three continuous politeness scores for each email.

| Politeness dimension | Dataset column |
|---|---|
| Directness vs. Indirectness | `Directness_vs_Indirectness__GOLD` |
| Positive Face Saving | `Structural_Politeness_and_Politeness_Markers__GOLD` |
| Negative Face Saving | `Tone_and_Overall_Consideration__GOLD` |

## Implementation

**Script:** `OPR_LLAMA_PredFA_gridsearch.py`

| Model | Description |
|---|---|
| LLaMA Text-only | Predicts politeness scores from email text using the last-token representation of the LLaMA encoder. |
| PredFA-only | Predicts politeness scores from aggregated Face Act features using a multilayer perceptron (MLP). |
| LLaMA + PredFA | Concatenates the LLaMA document representation with the PredFA features and passes the combined representation to an MLP regression head. |

## Predicted Face Act Features

The PredFA features come from a trained sentence-only LLaMA Face Act classifier. A single decision threshold shared across all nine labels is selected by maximizing FAC validation micro-F1. The classifier uses this threshold to assign one or more FA labels to a sentence.

### Nine Face Act Features

For each email, the script counts the sentence-level assignments of each FA label:

`HNeg+`, `HNeg-`, `HPos+`, `HPos-`, `Neutral`, `SNeg+`, `SNeg-`, `SPos+`, `SPos-`

### Four Additional Features

The script derives four aggregate features from the label counts:

- **H.** Sum of the four hearer-oriented FA counts.
- **S.** Sum of the four speaker-oriented FA counts.
- **Praise.** Sum of the four face-saving (`+`) FA counts.
- **Threat.** Sum of the four face-threatening (`-`) FA counts.

Together, the nine label counts and four aggregate features form the 13-dimensional HSPT-13 representation. The script standardizes these features using training-partition statistics and applies the same transformation to the validation and test partitions.

## Datasets

The script uses the following files:

- `data/corpus/email_text_gold_three_dimensions_politeness_score_with_seed_correct.csv`
- `data/corpus/sentences_with_golden_face_act.csv`

The document-level split groups emails by their original generation seed. All emails derived from the same seed belong to the same partition.

- Target partition sizes: 70% training, 15% validation, and 15% test
- Split random state: `42`
- Training seeds: `13`, `21`, `42`, `87`, `100`

All five training runs use the same document-level split.

## Training and Evaluation

### LLaMA Regressor

**Backbone:** `meta-llama/Llama-3.2-1B`  
**Adaptation:** Low-Rank Adaptation (LoRA) on `q_proj`, `k_proj`, `v_proj`, and `o_proj`  
**Loss:** Smooth L1 loss

The extended hyperparameter grid uses:

- Learning rate: `2e-4`
- Epochs: `5`, `8`, `12`
- LoRA rank: `8`, `16`, `32`
- LoRA alpha: twice the rank
- LoRA dropout: `0.05`
- Batch size: `4`
- Regression-head dropout: `0.1`
- Weight decay: `0.01`
- Hidden dimension: `128`

The three epoch settings and three LoRA ranks give nine configurations per training seed. The commands below enable the extended grid with `EXTENDED_LORA=1` and `EXTLORA_LRS=2e-4`.

### PredFA-only MLP

The PredFA-only MLP uses the following hyperparameter grid:

- Learning rate: `1e-3`, `5e-4`
- Epochs: `40`, `60`
- Batch size: `16`, `32`
- Dropout: `0.1`, `0.2`
- Weight decay: `1e-4`
- Hidden dimension: `128`, `256`

### Evaluation Metrics

We report mean absolute error (MAE) and Spearman's rank correlation (ρ) separately for Directness vs. Indirectness, Positive Face Saving, and Negative Face Saving. For each metric, the macro-average is the mean of the three per-dimension values.

For each model family and training seed, we select the configuration with the highest validation macro-Spearman's ρ and evaluate it on the fixed test set.

## Requirements

The experiments use Python, PyTorch, Transformers, PEFT, NumPy, pandas, SciPy, and scikit-learn. Learning-curve plots use Matplotlib.

Downloading `meta-llama/Llama-3.2-1B` requires authentication with a Hugging Face account that has access to the model.

## Running the Experiments

Run the following commands from the repository root. The examples use Linux or macOS shell syntax.

### 1. LLaMA Text-only

Run the text-only configuration with training seed `42`:

```bash
EXTENDED_LORA=1 \
EXTLORA_LRS=2e-4 \
OPR_RUN_SEED=42 \
SPLIT_RANDOM_STATE=42 \
RUN_LLAMA_TEXT=1 \
RUN_PREDFA_MLP=0 \
RUN_LLAMA_PLUS_PREDFA=0 \
python opr/llama/OPR_LLAMA_PredFA_gridsearch.py
```

This configuration runs independently of the FAC checkpoints.

### 2. PredFA-only and LLaMA + PredFA

These configurations use sentence-only LLaMA FAC adapters trained with the scripts in `fac/llama/`. The OPR script uses `FAC_REGISTRY_CSV` to locate the adapter corresponding to `FAC_RUN_SEED`.

Prepare a registry CSV with the following structure, replacing the example paths with the trained adapter directories:

```csv
run_seed,adapter_dir
13,/path/to/fac_adapter_seed13
21,/path/to/fac_adapter_seed21
42,/path/to/fac_adapter_seed42
87,/path/to/fac_adapter_seed87
100,/path/to/fac_adapter_seed100
```

Run both PredFA configurations with training seed `42`:

```bash
EXTENDED_LORA=1 \
EXTLORA_LRS=2e-4 \
FAC_RUN_SEED=42 \
OPR_RUN_SEED=42 \
SPLIT_RANDOM_STATE=42 \
FAC_REGISTRY_CSV="outputs/llama_lora_sentence_only_multiseed/llama_sentence_only_best_checkpoints.csv" \
RUN_LLAMA_TEXT=0 \
RUN_PREDFA_MLP=1 \
RUN_LLAMA_PLUS_PREDFA=1 \
python opr/llama/OPR_LLAMA_PredFA_gridsearch.py
```

### 3. Multi-seed Evaluation

Repeat the commands with `OPR_RUN_SEED` set to `13`, `21`, `42`, `87`, and `100`. Keep `SPLIT_RANDOM_STATE=42` for every run. For PredFA-only and LLaMA + PredFA, set `FAC_RUN_SEED` to the same value as `OPR_RUN_SEED`.

## Outputs

The script writes experimental results under `outputs/`, including:

- Hyperparameter grid-search results and the selected configuration for each model family
- Validation and test metrics and predicted politeness scores
- Data split information
- Predicted Face Act probabilities, FAC threshold information, and feature-standardization parameters for PredFA-based runs
- Learning-curve data and plots

The main summary files are:

- `opr_grid_search_results.csv`
- `opr_best_by_family.csv`
- `split_info.json`

## Reproducibility

To reproduce a reported run, use its training seed and the fixed document split. For PredFA-based runs, use the matching sentence-only FAC adapter, its validation-selected threshold, and the feature-standardization parameters computed on the training partition.
