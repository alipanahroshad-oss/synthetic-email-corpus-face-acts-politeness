
# LLaMA — Overall Politeness Regression (OPR)

This directory contains LLaMA-based implementations for document-level politeness prediction.

## Model

- Backbone: meta-llama/Llama-3.2-1B
- Fine-tuning: Low-Rank Adaptation (LoRA)
- Task: Document-level regression

## Prediction Targets

1. Directness vs. Indirectness
2. Positive Face Saving
3. Negative Face Saving

## Experimental Systems

The study evaluates the following system families:

1. Text — Text-only baseline
2. Pol — Politeness features only
3. Sent — Sentiment features only
4. Text+Pol+Sent — Text with politeness and sentiment features
5. Text+PredFA — Text with predicted Face Act features
6. Text+GoldFA — Text with gold Face Act features
7. PredFA-only — Predicted Face Act features without text
8. Text+Misaligned PredFA — Text with deliberately misaligned predicted Face Act features at test time

The feature-only systems do not use the LLaMA text encoder; they use a feature-based model.

## Evaluation

Performance is measured using:
- Mean Absolute Error (MAE)
- Spearman's rank correlation (rho)

## Reproducibility

Training scripts, feature construction, evaluation code, and instructions for reproducing the experiments will be added here.
