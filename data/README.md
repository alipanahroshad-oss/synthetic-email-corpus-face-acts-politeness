
# Dataset

This directory contains the synthetic email corpus, human annotations, processed datasets, and experimental data splits accompanying:

**A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**

## Corpus Overview

The dataset consists of 480 synthetic emails (240 requests and 240 replies), comprising 2,124 sentences.

Each email is annotated at two levels:

- **Sentence-level:** Multi-label Face Act (FA) annotations based on Brown and Levinson's politeness theory.
- **Document-level:** Politeness scores across three dimensions:
  1. Directness vs. Indirectness
  2. Positive Face Saving
  3. Negative Face Saving

Overall Politeness is defined as the average of the three dimensions.

## Directory Structure

### 1. annotation/

Human annotation data:

- `480email_politeness_scores_of_the_annotators.csv` — Document-level politeness annotations.
- `face_acts_annotators.csv` — Sentence-level Face Act annotations.

### 2. corpus/

Processed corpus files:

- `email_text_gold_three_dimensions_politeness_score_with_seed_correct.csv` — Email-level corpus with gold politeness scores and seed information.
- `sentences_with_golden_face_act.csv` — Sentence-level corpus with gold Face Act labels.

### 3. splits/

Data splits used for model training and evaluation:

- `train_seed42.csv` — Training partition.
- `val_seed42.csv` — Validation partition.
- `test_seed42.csv` — Test partition.

The experiments use a 70/15/15 train/validation/test split, grouped by original generation seed to prevent related emails from appearing across partitions.

Split random seed: 42.

### 4. validation/

Supporting corpus validation analysis:

- `gpt4o_intended_politeness_vs_human_gold.csv` — Data for comparing intended generation politeness levels with human-annotated politeness scores.

## Research Tasks

### Face Act Classification (FAC)

The task predicts nine sentence-level labels:

`HNeg+`, `HNeg-`, `HPos+`, `HPos-`, `SNeg+`, `SNeg-`, `SPos+`, `SPos-`, `Neutral`.

Multiple Face Act labels may be assigned to a sentence.

### Overall Politeness Regression (OPR)

The task predicts three document-level politeness dimensions from email text and, optionally, auxiliary linguistic features.

The experiments investigate the contribution of predicted and gold Face Act representations to politeness prediction.

## Annotation Guidelines

Detailed annotation instructions are available in the [documentation directory](../docs/).

## Limitations

The corpus consists of synthetic English-language email conversations. The modeling experiments are conducted within this controlled setting; generalization to human-written workplace emails remains to be established.
