
# LLaMA — Face Act Classification (FAC)

This directory contains the LLaMA-based Face Act Classification implementations for:

**A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**

## Overview

We evaluate LLaMA 3.2-1B with Low-Rank Adaptation (LoRA) for sentence-level multi-label Face Act Classification.

The task predicts nine Face Act labels:

`HNeg+`, `HNeg-`, `HPos+`, `HPos-`, `Neutral`, `SNeg+`, `SNeg-`, `SPos+`, `SPos-`

## Experimental Configurations

| Configuration | Script | Max tokens |
|---|---|---|
| Sentence — No Context | FAC_no_context_LLAMA_gridsearch.py | 256 |
| Sentence — Mail | FAC_context_mail_LLAMA_gridsearch.py | 256 |
| Sentence — Pair | FAC_context_pair_LLAMA_gridsearch.py | 512 |
| Sequence Labeling — Mail | FAC_seqlabel_mail_LLAMA_gridsearch.py | 1024 |
| Sequence Labeling — Pair | FAC_seqlabel_pair_LLAMA_gridsearch.py | 1024 |

### Context Definitions

- **No Context:** The classifier receives only the target sentence.
- **Sentence — Mail:** The target sentence is followed by its preceding sentences from the same email.
- **Sentence — Pair:** Request targets receive preceding request sentences; reply targets receive the complete paired request and preceding reply sentences.
- **Sequence Labeling — Mail:** All sentences within an email are encoded together, with one prediction per sentence.
- **Sequence Labeling — Pair:** The request and reply are jointly encoded, with sentence-level predictions.

## Model and Training

### Backbone

`meta-llama/Llama-3.2-1B`

### LoRA Configuration

- Rank: 16
- Alpha: 32
- LoRA dropout: 0.05
- Target modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

### Loss Function

Binary cross-entropy with logits (BCE).

Sentence classification uses the model's multi-label classification loss.

Sequence labeling uses a custom sentence-level classification head with BCE loss.

### Hyperparameter Grid

- Learning rates: 5e-5, 1e-4, 2e-4
- Epochs: 3, 5, 8
- Classifier dropout: 0.1, 0.2
- Weight decay: 0.0
- Early stopping patience: 2

Batch sizes:

| Configuration | Per-device batch sizes | Target effective batch size |
|---|---|---|
| Sentence — No Context | 4, 8 | 16 |
| Sentence — Mail | 4, 8 | 16 |
| Sentence — Pair | 2, 4 | 8 |
| Sequence Labeling — Mail | 2, 4 | 8 |
| Sequence Labeling — Pair | 2, 4 | 8 |

The implementations use gradient accumulation and configure bfloat16 precision.

## Dataset

The experiments use a fixed group-based split:

- `data/splits/train_seed42.csv`
- `data/splits/val_seed42.csv`
- `data/splits/test_seed42.csv`

Split seed: 42

Training seeds: 13, 21, 42, 87, 100

The data partition remains fixed across training seeds.

## Model Selection and Evaluation

Each script performs hyperparameter grid search using validation micro-F1.

A global classification threshold is selected on the validation set from 0.05 to 0.95 in increments of 0.01.

Evaluation includes:

- Micro-F1
- Micro-precision
- Micro-recall
- Class-wise performance
- Request and reply performance

## Requirements

The implementation requires Python and compatible versions of:

- PyTorch
- Transformers
- PEFT
- Accelerate
- NumPy
- pandas
- scikit-learn

The model `meta-llama/Llama-3.2-1B` requires authorized access through Hugging Face.

Users must accept the applicable model license and authenticate before downloading the model.

## Running the Experiments

Run all commands from the repository root.

For example, on Linux or macOS, run the no-context configuration with training seed 42:

```bash
RUN_SEED=42 python fac/llama/FAC_no_context_LLAMA_gridsearch.py
```

To execute the five training seeds:

```bash
for s in 13 21 42 87 100; do
    RUN_SEED=$s python fac/llama/FAC_no_context_LLAMA_gridsearch.py
done
```

Replace the script filename to run the other configurations.

## Outputs

Experimental artifacts are written to `outputs/`, including validation reports, selected hyperparameters, checkpoints, and evaluation outputs.

These generated files are not included in the source-code repository.

## Relationship to Previous Work

This implementation corresponds to the extended study and is separate from the original Findings of EMNLP 2026 implementation.

The journal work includes comparisons between BERT and LLaMA across the five FAC configurations.
