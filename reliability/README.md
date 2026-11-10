# Inter-Annotator Agreement and Reliability Analysis

This directory contains the implementations used to evaluate inter-annotator agreement for the corpus presented in:

**A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**

We assess annotation reliability at two levels:

1. **Sentence-level Face Act annotation**
2. **Document-level politeness annotation**

Agreement is evaluated using Krippendorff's alpha, with distance functions appropriate to each annotation task.

## Implementation

This directory contains two Jupyter notebooks:

| Notebook | Description |
|---|---|
| `krippendorff_alpha_face_acts.ipynb` | Computes Krippendorff's alpha for multi-label sentence-level Face Act annotations |
| `krippendorff_alpha_politeness.ipynb` | Computes Krippendorff's alpha for document-level politeness ratings |

Both analyses consider annotations from three annotators: Admin, Brianna, and Amber.

## 1. Sentence-Level Face Act Reliability

**Notebook:** `krippendorff_alpha_face_acts.ipynb`

### Annotation Scheme

The sentence-level annotation task uses nine Face Act labels:

- HNeg+
- HNeg-
- HPos+
- HPos-
- Neutral
- SNeg+
- SNeg-
- SPos+
- SPos-

Since a sentence may receive more than one Face Act label, this task is treated as a multi-label annotation problem.

### Agreement Metric

We compute Krippendorff's alpha using Jaccard set distance.

For two annotation sets A and B, Jaccard distance is defined as:

\[
d(A,B) = 1 - \frac{|A \cap B|}{|A \cup B|}
\]

The implementation measures disagreement between annotators' sets of Face Act labels rather than treating each label combination as an unrelated category.

Krippendorff's alpha is calculated as:

\[
\alpha = 1 - \frac{D_o}{D_e}
\]

where:

- \(D_o\) is observed disagreement.
- \(D_e\) is expected disagreement.

### Input Dataset

`data/annotation/face_acts_annotators.csv`

The notebook reads the following annotation columns:

- `FaceActsAdmin`
- `FaceActsBrianna`
- `FaceActsAmber`

Missing or empty annotations are excluded. Items with at least two valid annotations are included in the reliability calculation.

### Reported Statistics

The notebook computes:

- Krippendorff's alpha using Jaccard set distance
- Observed disagreement
- Expected disagreement
- Proportion of items containing at least one multi-label annotation
- Exact annotation-set consensus rate

The calculations are implemented directly in the notebook.

## 2. Document-Level Politeness Reliability

**Notebook:** `krippendorff_alpha_politeness.ipynb`

### Annotation Scheme

Document-level politeness is evaluated along three dimensions:

1. **Directness**
2. **Positive Face**
3. **Negative Face**

### Politeness Dimensions and Dataset Columns

The three dimensions correspond to the following dataset columns:

```python
# "Directness": "Directness_vs_Indirectness__GOLD"
# "Positive Face": "Structural_Politeness_and_Politeness_Markers__GOLD"
# "Negative Face": "Tone_and_Overall_Consideration__GOLD"
```

The notebook uses the original individual annotator ratings rather than the aggregated gold scores.

### Agreement Metric

We compute Krippendorff's alpha with an interval distance function.

The interval distance between two ratings is:

\[
d(a,b) = (a-b)^2
\]

This distance reflects the magnitude of disagreement between numerical ratings.

Krippendorff's alpha is computed separately for each politeness dimension.

### Input Dataset

`data/annotation/480email_politeness_scores_of_the_annotators.csv`

The notebook reads the following columns:

**Directness**

- `Directness_vs_Indirectness_admin`
- `Directness_vs_Indirectness_Brianna`
- `Directness_vs_Indirectness_Amber`

**Positive Face**

- `Structural_Politeness_and_Politeness_Markers_admin`
- `Structural_Politeness_and_Politeness_Markers_Brianna`
- `Structural_Politeness_and_Politeness_Markers_Amber`

**Negative Face**

- `Tone_and_Overall_Consideration_admin`
- `Tone_and_Overall_Consideration_Brianna`
- `Tone_and_Overall_Consideration_Amber`

### Reported Statistics

For each dimension, the notebook reports:

- Krippendorff's alpha with interval distance
- Number of items containing at least two valid annotations

Missing annotations are excluded from pairwise observed-disagreement calculations.

## Requirements

The notebooks require Python and compatible versions of:

- pandas
- NumPy
- Jupyter Notebook or JupyterLab

The Face Act reliability implementation also uses Python's standard-library modules.

## Running the Notebooks

The analyses can be executed independently.

### 1. Prepare the Repository

Clone the repository and ensure that the annotation files are available under:

`data/annotation/`

### 2. Start Jupyter

From the repository root, launch JupyterLab:

```bash
jupyter lab
```

Open the `reliability/` directory and select the notebook to execute.

### 3. Check Dataset Paths

The notebooks currently use dataset paths relative to the repository root:

- `data/annotation/face_acts_annotators.csv`
- `data/annotation/480email_politeness_scores_of_the_annotators.csv`

The Python kernel's working directory must be the repository root for these paths to resolve.

If the kernel starts inside `reliability/`, either change its working directory to the repository root or adjust the dataset paths to begin with `../data/annotation/`.

### 4. Execute

Run all cells in the selected notebook.

The reliability statistics are printed in the notebook output.

## Outputs

The notebooks report their results directly without requiring separate output files.

The computed statistics include:

- Face Act agreement using Jaccard set distance
- Document-level politeness agreement using interval distance
- Additional descriptive reliability statistics

Intermediate outputs are not included in the GitHub repository.

## Reproducibility

The notebooks provide the implementations used to assess human annotation reliability.

They use the original annotator-level data available in `data/annotation/` and compute agreement separately for sentence-level Face Acts and document-level politeness dimensions.

These analyses support the reliability assessment of the annotated corpus described in the journal manuscript.
