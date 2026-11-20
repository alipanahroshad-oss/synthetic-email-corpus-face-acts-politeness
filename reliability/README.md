# Inter-Annotator Agreement and Reliability Analysis

This directory contains the notebooks used to assess annotation reliability for **A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**.

The analyses use the individual judgments of three annotators (Admin, Brianna, and Amber). We compute Krippendorff's alpha at two annotation levels.

| Annotation task | Unit | Disagreement measure | Notebook |
|---|---|---|---|
| Multi-label Face Act annotation | Sentence | Jaccard distance between label sets | `krippendorff_alpha_face_acts.ipynb` |
| Politeness rating | Email | Squared difference between ratings, separately for each dimension | `krippendorff_alpha_politeness.ipynb` |

Both notebooks calculate agreement as $\alpha = 1 - D_o/D_e$, where $D_o$ is observed disagreement and $D_e$ is expected disagreement.

## Sentence-Level Face Act Agreement

**Input file:** `data/annotation/face_acts_annotators.csv`

**Annotation columns:** `FaceActsAdmin`, `FaceActsBrianna`, `FaceActsAmber`

Each annotator can assign one or more of nine Face Act labels to a sentence:

`HNeg+`, `HNeg-`, `HPos+`, `HPos-`, `Neutral`, `SNeg+`, `SNeg-`, `SPos+`, `SPos-`

The notebook compares two annotators' label sets $A$ and $B$ using Jaccard distance:

$$
d(A,B) = 1 - \frac{|A \cap B|}{|A \cup B|}
$$

Identical sets have distance zero, disjoint sets have distance one, and partial overlap gives an intermediate value. The analysis includes sentences with at least two nonempty annotations. An explicit `Neutral` label counts as a valid annotation.

The notebook reports:

- Krippendorff's alpha using Jaccard distance, together with observed and expected disagreement.
- The percentage of included sentences for which at least one annotator assigned multiple labels.
- The exact consensus rate, defined as the percentage of included sentences for which all available annotators assigned the same label set.

## Document-Level Politeness Agreement

**Input file:** `data/annotation/480email_politeness_scores_of_the_annotators.csv`

The notebook computes a separate Krippendorff's alpha for each politeness dimension using the individual annotator columns below.

| Politeness dimension | Admin | Brianna | Amber |
|---|---|---|---|
| Directness vs. Indirectness | `Directness_vs_Indirectness_admin` | `Directness_vs_Indirectness_Brianna` | `Directness_vs_Indirectness_Amber` |
| Positive Face Saving | `Structural_Politeness_and_Politeness_Markers_admin` | `Structural_Politeness_and_Politeness_Markers_Brianna` | `Structural_Politeness_and_Politeness_Markers_Amber` |
| Negative Face Saving | `Tone_and_Overall_Consideration_admin` | `Tone_and_Overall_Consideration_Brianna` | `Tone_and_Overall_Consideration_Amber` |

For two ratings $a$ and $b$ on the same email and dimension, interval disagreement is $(a-b)^2$. A two-point difference therefore contributes more disagreement than a one-point difference.

For each dimension, the notebook reports interval Krippendorff's alpha and the number of emails with at least two valid ratings.

## Requirements

The notebooks use Python, pandas, NumPy, and Jupyter Notebook or JupyterLab.

## Running the Notebooks

1. From the repository root, start JupyterLab:

   ```bash
   jupyter lab
   ```

2. Open either notebook in `reliability/`. The dataset paths above are relative to the repository root. If the kernel starts in `reliability/`, run `%cd ..` in a notebook cell before loading the data.

3. Run all cells in the selected notebook. Each notebook displays its results directly and can be run independently.
