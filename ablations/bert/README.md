# BERT Ablations for Overall Politeness Regression

These experiments examine how much Face Act (FA) features contribute to document-level politeness prediction and whether those features need to match the email being scored.

The reference conditions are **Text**, which encodes the email using `bert-base-uncased`, and **Text + PredFA**, which combines that encoding with features aggregated from predicted sentence-level FA labels.

| Condition | What it evaluates |
|---|---|
| **Text + GoldFA** | Combines email text with features derived from human-annotated FA labels. This oracle condition measures performance when gold labels are available. |
| **PredFA-only** | Predicts politeness from aggregated PredFA features alone. |
| **Text + Misaligned PredFA** | Reuses the trained Text + PredFA model. At test time, each email receives another test email’s PredFA vector to assess whether predictions depend on matching the features to the correct email. |
| **Text + Pol + Sent** | Combines email text with classical politeness indicators and VADER sentiment features for comparison with Text + PredFA. |

The paper reports mean absolute error (MAE) and Spearman’s correlation for **Directness vs. Indirectness**, **Positive Face Saving**, and **Negative Face Saving**. It also reports the average of each metric across the three dimensions.
