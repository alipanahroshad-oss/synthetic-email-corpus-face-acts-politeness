# A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness

**Roshad Alipanah, Valentin Barriere, and Jorge A. Baier**

Companion repository for the extended manuscript submitted to *IEEE Access*.

## Overview

This repository accompanies our research on computational politeness grounded in Brown and Levinson's politeness theory.

We introduce a synthetic corpus of long-form request–reply email conversations, manually annotated with sentence-level Face Acts (FAs) and document-level politeness scores.

Our research investigates two questions: whether conversational context improves sentence-level Face Act Classification (FAC), and whether representations derived from predicted Face Acts complement text-based models for Overall Politeness Regression (OPR).

## Dataset

The corpus consists of **480 synthetic emails**, including 240 requests and 240 replies, comprising **2,124 sentences**.

Sentence-level annotations use nine Face Act labels:

`HNeg+`, `HNeg-`, `HPos+`, `HPos-`, `SNeg+`, `SNeg-`, `SPos+`, `SPos-`, and `Neutral`.

Document-level politeness is evaluated along three dimensions:

- Directness vs. Indirectness
- Positive Face Saving
- Negative Face Saving

Overall Politeness is defined as the average of these dimensions.

## Experimental Framework

### Face Act Classification (FAC)

We investigate context-agnostic and context-sensitive models using BERT and LLaMA 3.2-1B with LoRA.

Five experimental configurations compare sentence-level classification and sequence labeling with different amounts of conversational context.

### Overall Politeness Regression (OPR)

We investigate whether Face Act representations provide complementary information for document-level politeness prediction.

Our experiments compare text-only models with models incorporating predicted or gold Face Act features, alongside additional baselines and ablation studies.

## Relationship to Previous Work

An earlier version of this research was published in **Findings of EMNLP 2026**.

[Read the EMNLP 2026 paper](https://aclanthology.org/2026.findings-emnlp.22/)

The journal manuscript extends the conference work with LLaMA-based experiments, systematic hyperparameter searches, multiple training runs, statistical comparisons, and additional ablation analyses.

**This repository contains the extended journal implementation and is separate from the original EMNLP codebase.**

## Reproducibility

The repository is being prepared to provide the implementation, experimental configurations, and reproducibility instructions for the extended study.

## Limitations

The corpus consists of synthetic English-language emails. All reported modeling experiments are conducted in this controlled setting, and generalization to human-written workplace communication remains to be established.

## Citation

The journal manuscript is currently under review. Citation information will be updated following publication.
