# LLaMA — Overall Politeness Regression (OPR)

This directory contains the LLaMA implementation for document-level Overall Politeness Regression (OPR) in **A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**.

## Overview

We investigate whether predicted sentence-level Face Acts (PredFA) complement email text when predicting document-level politeness. The experiments use three human-annotated politeness dimensions.

| Politeness dimension | Dataset column |
|---|---|
| Directness vs. Indirectness | `Directness_vs_Indirectness__GOLD` |
| Positive Face Saving | `Structural_Politeness_and_Politeness_Markers__GOLD` |
| Negative Face Saving | `Tone_and_Overall_Consideration__GOLD` |

## Model Configurations

The script `opr/llama/OPR_LLAMA_PredFA_gridsearch.py` implements three model families.

| Model | Input and architecture |
|---|---|
| LLaMA Text-only | Encodes the email with `meta-llama/Llama-3.2-1B` adapted using LoRA. A regression head uses the last-token representation to predict politeness scores. |
| PredFA-only | Uses a multilayer perceptron (MLP) to predict politeness scores from 13 document-level PredFA features. |
| LLaMA + PredFA | Concatenates the LLaMA document representation with the 13 PredFA features and passes the combined representation to an MLP regression head. |

## Predicted Face Act Features

A sentence-only LLaMA Face Act classifier supplies the predictions used to construct the document-level features. A single threshold shared across the nine FA labels converts sentence-level probabilities into label assignments. The threshold is selected by maximizing micro-F1 on the FAC validation set.

For each email, the script counts assignments of each of the nine labels and computes four additional features, giving 13 features in total (HSPT-13). A sentence can contribute to more than one label count.

| Feature | Labels included in the sum |
|---|---|
| Hearer | `HNeg+`, `HNeg-`, `HPos+`, `HPos-` |
| Speaker | `SNeg+`, `SNeg-`, `SPos+`, `SPos-` |
| Praise (face-saving acts) | `HNeg+`, `HPos+`, `SNeg+`, `SPos+` |
| Threat (face-threatening acts) | `HNeg-`, `HPos-`, `SNeg-`, `SPos-` |

`Neutral` has its own label count. The script standardizes the 13 features using statistics from the training partition and applies the same transformation to validation and test emails.

## Data and Training Seeds

The script uses two dataset files:

- `data/corpus/email_text_gold_three_dimensions_politeness_score_with_seed_correct.csv`
- `data/corpus/sentences_with_golden_face_act.csv`

Emails derived from the same original generation seed are assigned to the same partition. The target split proportions are 70% training, 15% validation, and 15% test. The split random state is `42`.

All experiments use the same partition across five training seeds: `13`, `21`, `42`, `87`, and `100`. For configurations that use PredFA, each OPR run uses the FAC adapter with the matching training seed.

## Training Configuration

### LLaMA Regressor

The text-based models use `meta-llama/Llama-3.2-1B` with LoRA and Smooth L1 loss. The reported OPR configuration applies LoRA to the attention projections `q_proj`, `k_proj`, `v_proj`, and `o_proj`. The script also supports adding `gate_proj`, `up_proj`, and `down_proj`.

| Hyperparameter | Extended search settings |
|---|---|
| Learning rate | `2e-4` |
| Epochs | `5`, `8`, `12` |
| LoRA rank | `8`, `16`, `32` |
| LoRA alpha | `2 × rank` |
| LoRA dropout | `0.05` |
| Batch size | `4` |
| Regression-head dropout | `0.1` |
| Weight decay | `0.01` |
| Hidden dimension | `128` |

Crossing three epoch settings with three LoRA ranks gives nine configurations per training seed. The commands below enable this grid with `EXTENDED_LORA=1` and `EXTLORA_LRS=2e-4`.

### PredFA-only MLP

| Hyperparameter | Search settings |
|---|---|
| Learning rate | `1e-3`, `5e-4` |
| Epochs | `40`, `60` |
| Batch size | `16`, `32` |
| Dropout | `0.1`, `0.2` |
| Weight decay | `1e-4` |
| Hidden dimension | `128`, `256` |

## Model Selection and Evaluation

For each model family and training seed, we select the OPR configuration with the highest validation macro-Spearman’s ρ. This is the average of the three correlations computed separately for the politeness dimensions.

On the fixed test set, we report mean absolute error (MAE) and Spearman’s ρ for each dimension. We compute macro-MAE and macro-Spearman’s ρ by averaging the corresponding per-dimension metrics. Lower MAE indicates smaller prediction errors, and higher ρ indicates closer agreement with the human ranking of emails.

## Requirements

The experiments use Python, PyTorch, Transformers, PEFT, NumPy, pandas, SciPy, and scikit-learn. Learning-curve plots use Matplotlib. To download `meta-llama/Llama-3.2-1B`, authenticate with a Hugging Face account that has access to the model.

## Running the Experiments

Run the following commands from the repository root using a Linux or macOS shell.

### 1. LLaMA Text-only

Run the extended search with training seed `42`:

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

This configuration can run independently of the FAC checkpoints.

### 2. PredFA-only and LLaMA + PredFA

These configurations use the trained sentence-only LLaMA FAC adapters from the experiments in `fac/llama/`. Supply a registry CSV mapping each FAC training seed to its adapter directory. Replace the example adapter paths with the directories containing your trained adapters.

```csv
run_seed,adapter_dir
13,/path/to/fac_adapter_seed13
21,/path/to/fac_adapter_seed21
42,/path/to/fac_adapter_seed42
87,/path/to/fac_adapter_seed87
100,/path/to/fac_adapter_seed100
```

The following command reads the existing registry at `outputs/llama_lora_sentence_only_multiseed/llama_sentence_only_best_checkpoints.csv` and selects the adapter for seed `42`:

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

### 3. Repeat Across Training Seeds

Use `13`, `21`, `42`, `87`, and `100` for `OPR_RUN_SEED`, keeping `SPLIT_RANDOM_STATE=42` in every run. For PredFA-based experiments, set `FAC_RUN_SEED` to the same value as `OPR_RUN_SEED`.

## Outputs

The script writes results under `outputs/`. The main summary files are:

- `opr_grid_search_results.csv` — validation and test metrics for the evaluated configurations.
- `opr_best_by_family.csv` — the validation-selected configuration for each model family.
- `split_info.json` — information about the document-level partition.

Run outputs also include predicted politeness scores and learning-curve data and plots. PredFA-based runs additionally record Face Act predictions, threshold information, and feature-standardization parameters.
