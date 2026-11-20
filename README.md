# A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness

**Roshad Alipanah, Valentin Barriere, and Jorge A. Baier**

Companion repository for the extended manuscript submitted to *IEEE Access*.

## Overview

We introduce a synthetic corpus of long-form request–reply emails grounded in Brown and Levinson's politeness theory. The corpus is manually annotated with sentence-level Face Acts (FAs) and document-level politeness scores.

We investigate two questions:

1. Does conversational context improve Face Act Classification (FAC)?
2. Do representations derived from predicted Face Acts complement text-based models for Overall Politeness Regression (OPR)?

## Dataset

The corpus contains **480 synthetic English-language emails**: 240 requests and 240 replies, comprising **2,124 sentences**. The emails were generated using GPT-4o from Enron-derived topics.

Sentence-level annotation is multi-label. A sentence can receive more than one of the nine FA labels:

`HNeg+`, `HNeg-`, `HPos+`, `HPos-`, `SNeg+`, `SNeg-`, `SPos+`, `SPos-`, and `Neutral`.

Each email is scored along three politeness dimensions:

- Directness vs. Indirectness
- Positive Face Saving
- Negative Face Saving

Overall Politeness is the average of these three scores.

## Experimental Framework

### Face Act Classification (FAC)

We compare BERT with a LoRA-adapted LLaMA 3.2-1B across five configurations:

| Modelling approach | Context |
|---|---|
| Sentence classification | no-context |
| Contextual sentence classification | Current-email (Mail) |
| Contextual sentence classification | Paired-email history (Pair) |
| Sequence labelling | Current-email (Mail) |
| Sequence labelling | Paired-email history (Pair) |

Contextual sentence classification predicts the FA labels of one target sentence at a time. In the Pair setting, reply targets also receive the initiating request. Sequence labelling encodes multiple sentences together and produces a separate FA prediction for each sentence.

### Overall Politeness Regression (OPR)

We predict the three document-level politeness scores and examine whether FA features complement the email text. We compare text-only models with models combining text and either predicted FAs (PredFA) or human-annotated FAs (GoldFA).

The experiments also evaluate FA-only models, misaligned PredFA features, and classical politeness and sentiment features. A category-wise permutation analysis examines how individual predicted and gold FA categories contribute to OPR.

## Relationship to Previous Work

The *Findings of EMNLP 2026* publication introduced the corpus, the annotation scheme, and the initial BERT-based FAC and OPR experiments.

[Read the EMNLP 2026 paper](https://aclanthology.org/2026.findings-emnlp.22/)

The journal manuscript extends this work by:

- Comparing BERT and LLaMA across five FAC configurations and for document-level OPR.
- Conducting systematic hyperparameter grid searches and comparing the backbones using individual 95% Welch confidence intervals across five training runs per backbone on a fixed test partition.
- Extending the GoldFA, PredFA-only, and misaligned-PredFA evaluations to both backbones, adding classical politeness and sentiment baselines, and examining individual FA categories through permutation analysis.
- Expanding the sentence-level naturalness evaluation from 124 sentences assessed by one annotator to 324 sentences assessed by two annotators, with inter-annotator agreement reported for naturalness ratings and human-versus-AI source judgments.

## Reproducibility

The manuscript describes the group-aware training, validation, and test split, the hyperparameter search budgets, and the validation-based selection criteria. Supplementary Appendix F, Table II, lists the hyperparameter search grids for FAC and OPR.

## Limitations

All reported FAC and OPR experiments use the synthetic English-language corpus. Generalization to human-written workplace emails remains to be established.

## Citation

The journal manuscript has been submitted to *IEEE Access*. Citation information will be updated following publication.
