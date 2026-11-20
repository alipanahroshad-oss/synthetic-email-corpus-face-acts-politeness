# LLaMA — Face Act Classification (FAC)

This directory contains the LLaMA implementations for sentence-level multi-label Face Act Classification (FAC) used in **A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**.

## Overview

We use LLaMA 3.2-1B with Low-Rank Adaptation (LoRA) to study how conversational context affects Face Act classification in request–reply emails.

Each sentence can receive one or more of nine Face Act labels:

`HNeg+`, `HNeg-`, `HPos+`, `HPos-`, `Neutral`, `SNeg+`, `SNeg-`, `SPos+`, `SPos-`

## Experimental Configurations

| Configuration | Script | Max input tokens |
|---|---|---:|
| Sentence — No Context | `FAC_no_context_LLAMA_gridsearch.py` | 256 |
| Sentence — Mail | `FAC_context_mail_LLAMA_gridsearch.py` | 256 |
| Sentence — Pair | `FAC_context_pair_LLAMA_gridsearch.py` | 512 |
| Sequence Labeling — Mail | `FAC_seqlabel_mail_LLAMA_gridsearch.py` | 1024 |
| Sequence Labeling — Pair | `FAC_seqlabel_pair_LLAMA_gridsearch.py` | 1024 |

### Context Definitions

- **Sentence — No Context.** LLaMA predicts the FA labels of the target sentence using only that sentence.
- **Sentence — Mail.** The input starts with the target sentence, followed by the preceding sentences from the same email.
- **Sentence — Pair.** Request targets use the same context as in Mail. For reply targets, the target sentence is followed by the initiating request and the preceding sentences from the reply.
- **Sequence Labeling — Mail.** LLaMA encodes the sentences of one email in order and predicts a set of FA labels for each sentence.
- **Sequence Labeling — Pair.** LLaMA encodes the request followed by its reply and predicts a set of FA labels for each sentence in both emails.

In sequence labeling, the classifier uses the representation at each sentence’s final token. With LLaMA’s causal attention, that representation can incorporate preceding sentences. In the Pair configuration, this lets reply predictions use context from the initiating request.

## Model and Training

### Backbone and LoRA

- **Backbone:** `meta-llama/Llama-3.2-1B`
- **LoRA rank:** 16
- **LoRA alpha:** 32
- **LoRA dropout:** 0.05
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### Loss Function

All configurations use binary cross-entropy with logits. Sentence classification uses the model’s multi-label classification loss. Sequence labeling applies the loss to the outputs of a custom sentence-level classification head.

### Hyperparameter Grid

| Hyperparameter | Values searched |
|---|---|
| Learning rate | `5e-5`, `1e-4`, `2e-4` |
| Maximum training epochs | `3`, `5`, `8` |
| Classifier dropout | `0.1`, `0.2` |

| Configuration | Per-device batch sizes | Target effective batch size |
|---|---|---:|
| Sentence — No Context | 4, 8 | 16 |
| Sentence — Mail | 4, 8 | 16 |
| Sentence — Pair | 2, 4 | 8 |
| Sequence Labeling — Mail | 2, 4 | 8 |
| Sequence Labeling — Pair | 2, 4 | 8 |

Each FAC configuration searches 36 hyperparameter combinations per training seed. Weight decay is fixed at `0.0`, and early stopping patience is set to `2`. The scripts use gradient accumulation to reach the target effective batch size and are configured for `bfloat16` precision.

## Dataset

All configurations use the same fixed split, grouped by the original email generation seed. This keeps related versions of each request–reply exchange in the same partition.

- `data/splits/train_seed42.csv`
- `data/splits/val_seed42.csv`
- `data/splits/test_seed42.csv`

The split seed is `42`. Each experiment uses five training seeds on that split: `13`, `21`, `42`, `87`, and `100`.

## Model Selection and Evaluation

For each checkpoint, we evaluate 91 decision thresholds from `0.05` to `0.95` in steps of `0.01`. Each threshold applies to all nine FA labels. We retain the threshold that maximizes validation micro-F1 for that checkpoint.

For each training seed and FAC configuration, we select the hyperparameter setting and checkpoint with the highest validation micro-F1. We then evaluate the selected model on the fixed test set using its validation-selected threshold.

We report micro-F1, micro-precision, and micro-recall overall and separately for request and reply sentences. The scripts also produce class-wise results. The paper reports means and standard errors across the five training seeds.

## Requirements

The scripts require Python and compatible versions of PyTorch, Transformers, PEFT, Accelerate, NumPy, pandas, and scikit-learn.

To download `meta-llama/Llama-3.2-1B`, accept the applicable model license and authenticate with a Hugging Face account that has access to the model.

## Running the Experiments

Run all commands from the repository root.

To run the no-context configuration with training seed 42 on Linux or macOS:

```bash
RUN_SEED=42 python fac/llama/FAC_no_context_LLAMA_gridsearch.py
```

To run the same configuration for all five training seeds:

```bash
for s in 13 21 42 87 100; do
    RUN_SEED=$s python fac/llama/FAC_no_context_LLAMA_gridsearch.py
done
```

For another configuration, replace the script filename with the corresponding filename in the configuration table.

## Outputs

The scripts write validation reports, selected hyperparameters, checkpoints, and evaluation results under `outputs/`. These files are generated when the experiments run.

## Relationship to Previous Work

The *Findings of EMNLP 2026* publication evaluated BERT-based FAC models. The extended journal study adds LLaMA with LoRA and compares both backbones across the five FAC configurations.
