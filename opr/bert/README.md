
# BERT — Overall Politeness Regression (OPR)

This directory contains BERT-based implementations for document-level politeness prediction.

## Model

- Backbone: bert-base-uncased
- Task: Document-level regression
- Prediction targets:
  - Directness vs. Indirectness
  - Positive Face Saving
  - Negative Face Saving

## Experimental Systems

The study evaluates the following system families:

1. Text — Text-only baseline
2. Pol — Politeness features only
3. Sent — Sentiment features only
4. Text+Pol+Sent — Text with politeness and sentiment features
5. Text+PredFA — Text with predicted Face Act features
6. Text+GoldFA — Text with human-annotated Face Act features (oracle)
7. PredFA-only — Predicted Face Act features without text
8. Text+Misaligned PredFA — Text with deliberately misaligned predicted Face Act features at test time

## Evaluation

Performance is evaluated using:
- Mean Absolute Error (MAE)
- Spearman's rank correlation (rho)

## Reproducibility

Training scripts, feature extraction, evaluation code, and execution instructions will be added here.
