# LLaMA — Overall Politeness Regression (OPR)

This directory contains the LLaMA implementation for document-level Overall Politeness Regression (OPR) in the extended study:

**A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**

## Overview

The Overall Politeness Regression (OPR) task predicts three continuous document-level politeness dimensions:

1. **Directness**
2. **Positive Face**
3. **Negative Face**

We investigate whether representations derived from predicted sentence-level Face Acts (PredFA) complement document-level text representations for politeness prediction.

We use LLaMA 3.2-1B with Low-Rank Adaptation (LoRA) and compare text-only, PredFA-only, and text-plus-PredFA regression models.

### Politeness Dimensions and Dataset Columns

The three politeness dimensions correspond to the following dataset columns:

```python
# "Directness": "Directness_vs_Indirectness__GOLD"
# "Positive Face": "Structural_Politeness_and_Politeness_Markers__GOLD"
# "Negative Face": "Tone_and_Overall_Consideration__GOLD"
```

The models predict all three dimensions, either jointly (multi-task) or independently (single-task).

## Implementation

**Script:** `OPR_LLAMA_PredFA_gridsearch.py`

Three model families are implemented:

| Model | Description |
|---|---|
| LLaMA Text-only | Predicts document-level politeness scores using email text |
| PredFA-only | Uses aggregated predicted Face Act features with an MLP |
| LLaMA + PredFA | Combines LLaMA document representations with predicted Face Act features |

Each model family supports:

- **Multi-task regression:** Jointly predicts Directness, Positive Face, and Negative Face.
- **Single-task regression:** Trains a separate predictor for each politeness dimension.

## Predicted Face Act Features

The PredFA features are obtained from a trained sentence-level LLaMA Face Act classifier using a LoRA adapter.

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

The implementation uses hard predicted Face Act labels, with a global threshold selected on the validation set to maximize micro-F1.

PredFA features are standardized using statistics computed from the training partition. The same normalization parameters are applied to validation and test data.

## Model Architecture

### LLaMA Text Encoder

The document-level regressor uses:

`meta-llama/Llama-3.2-1B`

The model is adapted using Low-Rank Adaptation (LoRA).

The implementation extracts a document representation from the model's final hidden states using last-token pooling.

This representation is passed through a regression head to predict the three politeness dimensions.

### LLaMA + PredFA Fusion

The fusion model concatenates the document-level LLaMA representation with the 13-dimensional PredFA feature vector.

The combined representation is processed by an MLP regression head.

### PredFA-only Model

The PredFA-only configuration uses an MLP to predict the three politeness dimensions without using document text embeddings.

## Datasets

The script uses the following dataset files:

- `data/corpus/email_text_gold_three_dimensions_politeness_score_with_seed_correct.csv`
- `data/corpus/sentences_with_golden_face_act.csv`

The document-level dataset is divided into training, validation, and test partitions using group-based splitting by the original generation seed.

- Training: 70%
- Validation: 15%
- Test: 15%
- Split random state: 42

The five training seeds are:

`13, 21, 42, 87, 100`

The document-level data split remains fixed across training seeds.

## Training Configuration

### LLaMA Regressor

Backbone:

`meta-llama/Llama-3.2-1B`

Loss function:

Smooth L1 (Huber) loss.

The implementation uses LoRA for parameter-efficient fine-tuning.

### LoRA Target Modules

The OPR experiments use the attention projection modules:

- q_proj
- k_proj
- v_proj
- o_proj

The script also supports an alternative configuration that additionally includes:

- gate_proj
- up_proj
- down_proj

The attention-only configuration is the default for the reported OPR experiments.

### Hyperparameter Grid Used in the Extended Study

The extended LoRA grid uses:

- Learning rate: 2e-4
- Epochs: 5, 8, 12
- LoRA rank: 8, 16, 32
- LoRA alpha: 2 × rank
- LoRA dropout: 0.05
- Batch size: 4
- Classifier dropout: 0.1
- Weight decay: 0.01
- Hidden dimension: 128

This produces nine hyperparameter configurations.

The extended grid is enabled using:

```bash
export EXTENDED_LORA=1
export EXTLORA_LRS=2e-4
```

Without these environment variables, the script uses its smaller default search grid.

### PredFA-only MLP

The PredFA-only model uses the following hyperparameter grid:

- Learning rate: 1e-3, 5e-4
- Epochs: 40, 60
- Batch size: 16, 32
- Dropout: 0.1, 0.2
- Weight decay: 1e-4
- Hidden dimension: 128, 256

## Evaluation Metrics

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
- PEFT
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib (for learning-curve plots)

The pretrained model `meta-llama/Llama-3.2-1B` requires authorized Hugging Face access.

Users must accept the applicable model license and authenticate before running the experiments.

GPU acceleration is recommended.

## Running the Experiments

All commands should be executed from the repository root.

The following examples use Linux or macOS shell syntax.

### 1. LLaMA Text-only

This configuration predicts document-level politeness scores using only email text.

It does not require a trained FAC classifier or a LoRA adapter registry.

To run the extended configuration with training seed 42:

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

### 2. PredFA-only and LLaMA + PredFA

These configurations require a previously trained sentence-level LLaMA FAC model from `fac/llama/`.

The OPR script loads the corresponding FAC LoRA adapter using a registry CSV.

The registry must contain a training-seed column and an adapter-path column.

For example:

```csv
run_seed,adapter_dir
13,/path/to/fac_adapter_seed13
21,/path/to/fac_adapter_seed21
42,/path/to/fac_adapter_seed42
87,/path/to/fac_adapter_seed87
100,/path/to/fac_adapter_seed100
```

Replace the placeholder paths with the actual directories containing the trained FAC LoRA adapters.

The registry can be stored locally under `outputs/`.

The script uses `FAC_RUN_SEED` to select the corresponding adapter.

Example for training seed 42:

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

The `FAC_REGISTRY_CSV` path must point to an existing registry containing valid adapter locations. The registry is a prerequisite for these configurations; it is not automatically generated by this OPR script.

### 3. Multi-seed Experiments

The experiments use five training seeds:

`13, 21, 42, 87, 100`

The document-level split is kept fixed with:

```bash
SPLIT_RANDOM_STATE=42
```

For text-only experiments, vary `OPR_RUN_SEED`.

For PredFA-based experiments, pair the FAC and OPR training seeds:

```bash
FAC_RUN_SEED=42
OPR_RUN_SEED=42
```

Repeat the experiments with the remaining four seeds.

### 4. Multi-task and Single-task Regression

By default, the script executes both multi-task and single-task variants.

These settings can be controlled using:

```bash
DO_MULTI_TASK=1
DO_SINGLE_TASK=1
```

Set either variable to `0` to disable the corresponding training mode.

## Outputs

The script automatically creates the required output directories under `outputs/`.

Generated artifacts include:

- Hyperparameter grid-search results
- Best configuration per model family
- Validation and test evaluation metrics
- Predicted politeness scores
- Predicted Face Act probabilities
- Validation-based Face Act threshold information
- Data split information
- PredFA feature normalization parameters
- Learning-curve data and plots

The script generates summary files including:

- `opr_grid_search_results.csv`
- `opr_best_by_family.csv`
- `split_info.json`

Experimental outputs and trained checkpoints are not included in the GitHub repository.

The root `.gitignore` excludes the `outputs/` directory.

## Reproducibility

The implementation uses a fixed group-based document split while varying the training random seed.

The LLaMA Text-only configuration can run independently.

The PredFA-only and LLaMA + PredFA configurations require previously trained sentence-level FAC LoRA adapters and a local registry containing their paths.

Both multi-task and single-task regression are supported.

The experiments compare document text representations, predicted Face Act representations, and their combination to investigate the contribution of Face Act information to document-level politeness prediction.

This implementation is part of the extended BERT–LLaMA evaluation in the journal manuscript and is separate from the earlier Findings of EMNLP 2026 implementation.
