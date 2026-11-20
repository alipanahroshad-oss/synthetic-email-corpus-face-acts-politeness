# LLaMA Ablations for Overall Politeness Regression

These experiments examine how much Face Act (FA) features contribute to document-level politeness prediction and whether those features need to match the email being scored.

The reference conditions are **Text**, which encodes the email using `meta-llama/Llama-3.2-1B` adapted with LoRA, and **Text + PredFA**, which combines that encoding with features aggregated from predicted sentence-level FA labels.

| Condition | What it evaluates |
|---|---|
| **Text + GoldFA** | Combines the LLaMA text representation with features derived from human-annotated FA labels. This oracle condition measures performance when gold FA labels are available. |
| **PredFA-only** | Uses a multilayer perceptron (MLP) to predict politeness from aggregated PredFA features alone. |
| **Text + Misaligned PredFA** | Reuses the trained Text + PredFA model. At test time, each email receives another test email’s PredFA vector to assess whether predictions depend on matching the features to the correct email. |
| **Text + Pol + Sent** | Combines the LLaMA text representation with classical politeness indicators and VADER sentiment features for comparison with Text + PredFA. |

The paper reports mean absolute error (MAE) and Spearman’s correlation for **Directness vs. Indirectness**, **Positive Face Saving**, and **Negative Face Saving**. It also reports the average of each metric across the three dimensions.
