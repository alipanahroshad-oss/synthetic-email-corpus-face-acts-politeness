# Synthetic Email Generation and Politeness Validation

This directory contains the scripts used for synthetic email generation and politeness-level validation in:

**A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**

The scripts document the generation prompts, procedures, and validation analyses used in constructing the synthetic email corpus.

## Overview

The corpus generation procedure uses GPT-4o to produce professional email exchanges containing Face-Threatening Acts (FTAs), alternative replies, and progressively more polite versions.

The workflow includes three main components:

1. Generating face-threatening emails and four alternative reply types.
2. Generating progressively more polite versions of the initial email conversations.
3. Evaluating the alignment between GPT-4o's intended politeness levels and human gold-standard document-level scores.

The generation scripts are released primarily for methodological transparency. The repository provides the final annotated corpus and validation dataset, but not all intermediate files used during generation.

## Implementation

This directory contains three Python scripts:

| Script | Description |
|---|---|
| `generating_face_threatening_emails.py` | Generates initial face-threatening emails and four alternative replies using GPT-4o |
| `generating_more_polite_version.py` | Generates four progressively more polite versions of initial email conversations |
| `Alignment_between_GPT4o_intended_politeness_levels_and_human_gold_document_level.py` | Evaluates the alignment between intended politeness levels and human gold document-level scores |

## 1. Face-Threatening Email Generation

**Script:** `generating_face_threatening_emails.py`

### Purpose

The script generates initial professional emails containing unmitigated Face-Threatening Acts.

The generation prompts draw on Brown and Levinson's politeness theory.

The initial email is designed to challenge the recipient's positive or negative face without introducing mitigating politeness strategies.

Topic-related keywords derived from the Enron email corpus are used to guide generation.

### Reply Generation

For each initial email, GPT-4o generates four alternative replies:

| Reply Type | Description |
|---|---|
| Accept-Polite | Accepts the sender's Face-Threatening Act using face-saving language |
| Reject-Impolite | Rejects the sender's Face-Threatening Act using face-threatening language |
| Accept-Impolite | Accepts the sender's Face-Threatening Act using face-threatening language |
| Reject-Polite | Rejects the sender's Face-Threatening Act using face-saving language |

These alternatives combine acceptance or rejection with polite or impolite language.

### Generation Configuration

- Model: `gpt-4o`
- Language: English
- Domain: Professional workplace communication
- Initial style: Face-threatening, without politeness mitigation
- Reply alternatives: Four per initial email

The script requests one initial email and its four reply alternatives for each topic-keyword record.

### Input and Output

The script expects a CSV containing topic-related keywords in a column named:

`Top_n_words`

It generates a CSV containing:

- `Top_n_words`
- `Rude_Email_with_Four_Replies`

The topic-keyword input file and intermediate generated outputs are not included in this repository.

The original generation prompt is preserved in the Python script.

## 2. Generating Progressively More Polite Versions

**Script:** `generating_more_polite_version.py`

### Purpose

The script transforms initially impolite email conversations into four progressively more polite alternatives while requesting preservation of the original communicative intent.

The generation prompt encourages the increasing use of politeness strategies, including:

- Positive politeness
- Negative politeness
- Indirectness
- Hedging
- Expressions of gratitude and appreciation
- Mitigation of Face-Threatening Acts
- Off-record strategies

The prompt also instructs GPT-4o not to modify salutations or complimentary closings to increase politeness.

### Intended Politeness Levels

The generation procedure is designed around five intended politeness levels:

| Level | Description |
|---|---|
| 1 | Initial face-threatening version |
| 2 | First more polite version (E1) |
| 3 | Second more polite version (E2) |
| 4 | Third more polite version (E3) |
| 5 | Fourth more polite version (E4) |

The prompt requests that each successive version be more polite than the preceding version while preserving the underlying communicative intent.

These are intended generation levels rather than independently verified politeness scores.

Their relationship to human-perceived politeness is examined separately in the validation analysis.

### Generation Configuration

- Model: `gpt-4o`
- Input: Initially impolite email conversations
- Output: Four progressively more polite versions
- Objective: Vary politeness while preserving communicative intent

### Input and Output

The script expects an intermediate CSV containing the column:

`Rude_emails`

It generates a new column:

`polite_version`

This column contains the sequence of four generated alternatives.

The intermediate input and output CSV files are not included in this repository.

The script preserves the original GPT-4o prompting instructions used for this generation stage.

## 3. Alignment Between Intended Politeness Levels and Human Gold Scores

**Script:** `Alignment_between_GPT4o_intended_politeness_levels_and_human_gold_document_level.py`

### Purpose

This script evaluates whether the intended politeness levels specified during GPT-4o generation correspond to document-level politeness scores assigned by human annotators.

The analysis compares intended levels from 1 to 5 with human gold-standard scores.

### Politeness Dimensions

The validation evaluates three document-level politeness dimensions:

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

### Input Dataset

The validation dataset is available in the repository:

`data/validation/gpt4o_intended_politeness_vs_human_gold.csv`

The analysis requires the following columns:

- `level`: GPT-4o intended politeness level (1–5)
- `Directness_vs_Indirectness__GOLD`
- `Structural_Politeness_and_Politeness_Markers__GOLD`
- `Tone_and_Overall_Consideration__GOLD`

Records with missing required values are excluded.

### Evaluation Methods

The script computes:

- Spearman's rank correlation coefficient (ρ)
- Correlation p-values
- 95% bootstrap confidence intervals for Spearman's ρ
- Mean human gold scores for each intended politeness level
- Difference between the mean gold scores at levels 5 and 1

Bootstrap configuration:

- Number of bootstrap samples: 4,000
- Random seed: 13

The analysis is performed separately for Directness, Positive Face, and Negative Face.

### Visualization and Outputs

The script generates a plot showing the relationship between intended politeness levels and human gold-standard scores.

The visualization includes:

- Mean scores for each intended politeness level
- Approximate 95% confidence intervals for the means
- Separate curves for the three politeness dimensions

The script also generates a LaTeX table containing the numerical results.

These generated outputs do not need to be included in the repository because they can be recreated using the validation dataset and analysis script.

## Requirements

The scripts require Python and compatible versions of the following libraries:

- `openai`
- `pandas`
- `numpy`
- `scipy`
- `matplotlib`

The generation scripts require access to the OpenAI API and the GPT-4o model.

API credentials should be configured using the `OPENAI_API_KEY` environment variable.

**API keys and other credentials must not be included in the repository.**

Executing the generation scripts sends API requests that may incur charges.

## Running the Scripts

All commands below are intended to be executed from the repository root.

### 1. Generate Face-Threatening Emails

```bash
python generation/generating_face_threatening_emails.py
```

This script requires a suitable topic-keyword input CSV and correctly configured local input/output paths.

### 2. Generate More Polite Versions

```bash
python generation/generating_more_polite_version.py
```

This script requires an intermediate dataset containing the initial email conversations and correctly configured local input/output paths.

### 3. Evaluate Politeness-Level Alignment

```bash
python generation/Alignment_between_GPT4o_intended_politeness_levels_and_human_gold_document_level.py
```

The validation script should use the released dataset at:

```python
PATH = "data/validation/gpt4o_intended_politeness_vs_human_gold.csv"
```

The original scripts were exported from Google Colab and may contain absolute paths referring to their original execution environment.

Before execution, users must configure input and output paths for their local environment.

The generation commands are therefore not directly executable against the released repository alone without supplying the required intermediate inputs.

## Data Availability

The repository releases the final research datasets and the code used for corpus generation and analysis.

### Available Data

**Final synthetic corpus**

`data/corpus/`

Contains the released sentence-level and document-level gold-standard datasets.

**Human annotations**

`data/annotation/`

Contains annotator-level data used for the annotation and reliability analyses.

**Politeness-level validation data**

`data/validation/`

Contains the dataset used to evaluate the relationship between GPT-4o intended politeness levels and human document-level scores.

**Generation and validation scripts**

`generation/`

Contains the original generation prompts and validation analysis code.

### Intermediate Generation Files

The original corpus construction procedure involved intermediate files for:

- Topic selection and keyword extraction
- Initial face-threatening email generation
- Selection and preprocessing of email conversations
- Generation of progressively more polite versions

These intermediate files are **not included in the public repository**.

They are not required to use the released final annotated corpus or to study the original generation prompts.

However, re-executing the complete generation procedure requires suitable intermediate inputs and corresponding path configuration.

## Reproducibility and Limitations

The scripts are released to support transparency regarding the synthetic corpus construction procedure and its theoretical foundations.

The following considerations apply:

- The generation scripts preserve the original prompts and model selection.
- The final annotated corpus is available independently of the generation scripts.
- Intermediate generation files are not distributed.
- The generation scripts do not constitute a fully automated end-to-end corpus construction pipeline.
- GPT-4o outputs may differ across executions.
- Reproducing the complete original generation process requires additional intermediate inputs and preprocessing steps.
- The intended-versus-human politeness alignment analysis uses the separately released validation dataset.

The alignment analysis provides evidence about the relationship between intended politeness levels and human judgments within the controlled synthetic corpus.

It does not establish generalization to human-written workplace emails.

## Research Context

This implementation accompanies the extended journal manuscript:

**A Synthetic Email Corpus Annotated with Brown and Levinson's Face Acts and Document-Level Politeness**

It documents the generation and validation components of the research and is separate from the earlier Findings of EMNLP 2026 codebase.
