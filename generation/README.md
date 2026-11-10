# Synthetic Email Generation and Politeness Validation

This directory contains the scripts used for synthetic email generation and politeness-level validation in:

**A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**

The scripts support the construction of a controlled synthetic email corpus based on Brown and Levinson's politeness theory, together with an analysis of the relationship between intended politeness levels and human document-level judgments.

## Overview

The generation procedure uses GPT-4o to produce professional email exchanges containing Face-Threatening Acts (FTAs) and alternative replies.

The workflow includes three main components:

1. Generating face-threatening emails and four alternative replies.
2. Generating progressively more polite versions of the initial email conversations.
3. Evaluating the alignment between GPT-4o's intended politeness levels and human gold-standard document-level scores.

These scripts represent specific stages of the corpus construction and validation process. Additional data selection, preprocessing, and annotation procedures are described in the manuscript.

## Implementation

This directory contains three Python scripts:

| Script | Description |
|---|---|
| `generating_face_threatening_emails.py` | Generates initial face-threatening emails and four alternative reply types using GPT-4o |
| `generating_more_polite_version.py` | Generates four progressively more polite versions of initial email conversations |
| `Alignment_between_GPT4o_intended_politeness_levels_and_human_gold_document_level.py` | Evaluates alignment between intended politeness levels and human gold document-level scores |

## 1. Face-Threatening Email Generation

**Script:** `generating_face_threatening_emails.py`

### Purpose

The script generates initial professional emails containing unmitigated Face-Threatening Acts.

The generation prompts are based on Brown and Levinson's politeness theory.

The initial email is designed to challenge the recipient's positive or negative face without introducing mitigating politeness strategies.

The script uses topic-related keywords derived from the Enron email corpus to guide generation.

### Reply Generation

For each initial email, GPT-4o generates four alternative replies:

| Reply Type | Description |
|---|---|
| Accept-Polite | Accepts the request using face-saving language |
| Reject-Impolite | Rejects the request using face-threatening language |
| Accept-Impolite | Accepts the request using face-threatening language |
| Reject-Polite | Rejects the request using face-saving language |

These alternatives combine acceptance or rejection with polite or impolite responses.

### Generation Model

- Model: `gpt-4o`
- Domain: Professional workplace communication
- Language: English
- Initial style: Face-threatening, without politeness mitigation
- Reply variants: Four per generated initial email

The script requests one initial email and its four reply alternatives for each topic-keyword record.

### Input

The script expects a CSV containing topic-related keywords, including the column:

`Top_n_words`

The original script references:

`/data/raw/cleaned_top_n_words_non_spam_emails_Enron_46.9M_FINAL.csv`

This source file is a separate input dependency and is not part of the released annotated corpus.

### Output

The script generates a CSV containing:

- `Top_n_words`
- `Rude_Email_with_Four_Replies`

The generated email text contains the initial face-threatening email and its four alternative replies.

## 2. Generating Progressively More Polite Versions

**Script:** `generating_more_polite_version.py`

### Purpose

The script transforms initially impolite email conversations into four progressively more polite alternatives while requesting preservation of the original communicative intent.

The generation prompt encourages increasing use of politeness strategies, including:

- Positive politeness
- Negative politeness
- Indirectness
- Hedging
- Expressions of appreciation
- Mitigation of Face-Threatening Acts
- Off-record strategies

The script also instructs GPT-4o not to increase politeness through changes to salutations or complimentary closings.

### Politeness Levels

The intended politeness scale contains five levels:

| Level | Description |
|---|---|
| 1 | Initial face-threatening version |
| 2 | Email Version 1 (E1) |
| 3 | Email Version 2 (E2) |
| 4 | Email Version 3 (E3) |
| 5 | Email Version 4 (E4) |

The prompt requests that each successive version be more polite than its predecessor.

These are **intended generation levels**. Their relationship to human-perceived politeness is examined separately in the validation analysis.

### Generation Model

- Model: `gpt-4o`
- Input: Initially impolite email conversations
- Output: Four increasingly polite versions
- Goal: Preserve communicative intent while varying politeness

### Input

The script expects a CSV containing:

`Rude_emails`

The original script references:

`/data/raw/face_threatening_emails_July2025_ShowToAnnotators_stripped_2.csv`

This is a selected or preprocessed intermediate dataset. It is not directly produced under the same filename and column structure by the first generation script.

### Output

The script adds a column:

`polite_version`

This column contains GPT-4o's generated sequence of four increasingly polite versions.

## 3. Alignment Between Intended Politeness and Human Gold Scores

**Script:** `Alignment_between_GPT4o_intended_politeness_levels_and_human_gold_document_level.py`

### Purpose

This script examines whether the intended politeness levels assigned during GPT-4o generation correspond to document-level politeness scores assigned by human annotators.

The analysis compares intended politeness levels from 1 to 5 with human gold-standard scores.

### Politeness Dimensions

The validation evaluates three document-level dimensions:

1. **Directness**
2. **Positive Face**
3. **Negative Face**

### Politeness Dimensions and Dataset Columns

The corresponding dataset columns are:

```python
# "Directness": "Directness_vs_Indirectness__GOLD"
# "Positive Face": "Structural_Politeness_and_Politeness_Markers__GOLD"
# "Negative Face": "Tone_and_Overall_Consideration__GOLD"
```

### Input Dataset

The validation dataset is located at:

`data/validation/gpt4o_intended_politeness_vs_human_gold.csv`

The analysis requires:

- `level`: Intended politeness level (1–5)
- `Directness_vs_Indirectness__GOLD`
- `Structural_Politeness_and_Politeness_Markers__GOLD`
- `Tone_and_Overall_Consideration__GOLD`

The analysis excludes records with missing required values.

### Evaluation Methods

The script computes:

- Spearman's rank correlation coefficient (ρ)
- Correlation p-values
- 95% bootstrap confidence intervals for Spearman's ρ
- Mean human gold scores for each intended politeness level
- Difference between the mean gold scores at levels 5 and 1

Bootstrap settings:

- Bootstrap samples: 4,000
- Random seed: 13

### Visualization

The script generates a plot illustrating the relationship between GPT-4o's intended politeness levels and human gold document-level scores.

The plot includes:

- Mean scores for each intended politeness level
- Approximate 95% confidence intervals for the means
- Separate curves for Directness, Positive Face, and Negative Face

The script also produces a LaTeX table summarizing the alignment analysis.

## Requirements

The scripts require Python and the following libraries:

- `openai`
- `pandas`
- `numpy`
- `scipy`
- `matplotlib`

The generation scripts require access to the OpenAI API and the GPT-4o model.

API credentials should be configured through the `OPENAI_API_KEY` environment variable.

**Do not include API keys or other credentials in the repository.**

Running the generation scripts makes API requests that may incur charges.

## Running the Scripts

Execute the scripts from the repository root.

### 1. Generate Face-Threatening Emails

```bash
python generation/generating_face_threatening_emails.py
```

### 2. Generate More Polite Versions

```bash
python generation/generating_more_polite_version.py
```

### 3. Evaluate Politeness-Level Alignment

```bash
python generation/Alignment_between_GPT4o_intended_politeness_levels_and_human_gold_document_level.py
```

### Important: Input and Output Paths

The original generation scripts were exported from Google Colab and contain absolute paths under `/data/raw/` and `/data/validation/`.

Before running the scripts in a cloned repository, these paths must be adjusted to locations available on the local machine.

For the alignment analysis, the repository-relative input path is:

```python
PATH = "data/validation/gpt4o_intended_politeness_vs_human_gold.csv"
```

The topic-keyword CSV and selected intermediate email CSV must also be supplied to reproduce their corresponding generation stages.

Generated files can be saved locally under `outputs/generation/`. The output paths in the original scripts must be updated accordingly.

The root `.gitignore` excludes the `outputs/` directory.

## Reproducibility

The scripts document the generation prompts, model selection, transformations, and intended-versus-human-rated politeness analysis.

The following considerations apply:

- GPT-4o generation may produce different responses across runs.
- Reproducing the initial generation requires the original topic-keyword input.
- Reproducing the politeness transformation requires the selected intermediate email dataset.
- The generation scripts do not constitute a complete automated pipeline for all corpus construction and annotation stages.
- The released annotated corpus and gold-standard scores are available separately under `data/corpus/` and `data/annotation/`.
- The intended-level alignment dataset is available under `data/validation/`.

The validation analysis provides evidence about how intended politeness levels correspond to human judgments within the controlled synthetic corpus. It does not establish generalization to human-written workplace emails.

This implementation accompanies the extended journal manuscript and is separate from the earlier Findings of EMNLP 2026 codebase.
