# LangCode 2027 Shared Task 2: Adversarial Prompt Detection

Project workspace for the LangCode 2027 Shared Task 2. The task is to classify a prompt sent to a code model as safe (`S`) or unsafe (`U`) when it is code-mixed with English or English transliterated into another script. The starter kit is downloaded locally, and split-aware data loading and validation are in place. Model training and inference are the next implementation steps.

## Problem in plain language

Build a binary classifier that recognizes whether a multilingual or transliterated prompt is asking for malicious code or for legitimate security work. Safe prompts can discuss the same security topics as unsafe prompts (for example, detection, defense, or authorized testing), so keyword blocking is insufficient. The system must interpret the request and should work across scripts and language/code mixtures.

The task covers code-mixed prompts involving Arabic, Bengali, Chinese, Hebrew, Hindi, Japanese, Korean, Russian, Tamil, or Thai, and English written in one of those scripts. Programming and security terms are often left in English. The language and prompt form are not supplied as features in the released CSVs.

**Input:** one prompt and its identifier.

**Required prediction:** exactly one class label, `S` (safe) or `U` (unsafe), for each prompt.

## Official data and use requirements

The official [Data page](https://langcode2027.github.io/task2/data/) lists these releases:

| Split | Size | Labels | Intended use |
|---|---:|---|---|
| Train | 20,000 | Yes | Fit the detector |
| Dev | 10,000 | Yes | Select models, tune settings, and evaluate locally |
| Dev-test | 10,000 | No | Development-phase submission practice on CodaBench |
| Test | 30,000 | No | Final evaluation; scheduled for release Jan 8, 2027 |

Each CSV is UTF-8 and has one prompt per row. The columns are `id` (example: `T2-483920`) and `prompt`; train and dev also include `label`, which is `S` or `U`. Train and dev are balanced by label and by code-mixed versus transliterated form. The language and form are not identified per row.

The organizers state that any public data, models, and tools may be used. However, dev-test and test prompts must be labeled one prompt at a time: do not look them up in other datasets, including sources used to build MALICE. The data page also gives a research-use condition: use the data only to research detection and defense, do not use it to build or improve attacks, do not run unsafe prompts against production systems, and do not redistribute it. **The downloaded files are local under `data/raw/`, ignored by Git, and must not be committed or redistributed.**

The benchmark is built from seed prompts in MALICE. If the data is used in research, include the citation on the official data page (Raihan et al., Findings of EMNLP 2026).

### Downloaded starter kit and integrity check

The following released files are present locally in `data/raw/` and excluded from version control:

| File | Purpose |
|---|---|
| `train.csv` | 20,000 labeled prompts |
| `dev.csv` | 10,000 labeled prompts |
| `devtest.csv` | 10,000 unlabeled development-submission prompts |
| `scorer.zip` and `scorer/score.py` | Official scorer and format checker |
| `LangCode2027_Task2_starter.ipynb` | Colab BERT-base baseline notebook |

The downloaded CSVs were read as UTF-8. `train.csv` has columns `id,prompt,label`, 20,000 unique IDs, no empty IDs/prompts/labels, and 10,000 examples per label. `dev.csv` has the same columns, 10,000 unique IDs, no empty IDs/prompts/labels, and 5,000 examples per label. `devtest.csv` has columns `id,prompt`, 10,000 unique IDs, and no empty IDs/prompts. None of the three files contains duplicate IDs or labels outside `S` and `U`.

### Submission and scorer format

`submission.csv` must contain the columns `id` and `label`. A submission can be a CSV, a folder, or a ZIP. A folder or ZIP must contain exactly one file named `submission.csv`. Labels are `S` or `U` (case-insensitive); duplicate or empty IDs are rejected. Extra columns are ignored. Missing predictions count as wrong; unknown IDs are ignored by the scorer.

Run these commands from the repository root:

```bash
# Check format and exact coverage for a dev-test submission archive
python data/raw/scorer/score.py --check submission.zip data/raw/devtest.csv

# Score a local CSV against labeled development data
python data/raw/scorer/score.py --score submission.csv data/raw/dev.csv
```

For CodaBench, the scorer reads `/app/input/ref/reference.csv` and `/app/input/res/`, then writes `scores.json` under `/app/output/`.

### Starter notebook environment and baseline

The notebook targets Python 3 in Google Colab and recommends a T4 GPU. It installs `transformers>=4.48,<5`, `accelerate`, and `datasets`; Colab supplies PyTorch, NumPy, and pandas. The baseline loads `google-bert/bert-base-uncased`, truncates to 256 tokens, trains for two epochs with batch size 32 and learning rate `2e-5`, evaluates on dev each epoch, and uses seed 42. The notebook estimates about 15 minutes on a free T4. It writes `submission.csv` and `submission.zip`, then checks the dev-test archive with the official scorer.

Run the repository's split-aware integrity check from the repository root:

```bash
python scripts/validate_data.py
```

Example output with the starter-kit files:

```text
train: 20,000 rows, S=10,000, U=10,000
dev: 10,000 rows, S=5,000, U=5,000
devtest: 10,000 unlabeled prompts
test: not present (released later by the organizers)
OK: schemas, contents, row counts, labels, and split IDs are valid.
```

The command checks the released splits' exact headers and row counts, UTF-8 decoding, non-empty IDs/prompts, label values and balance, duplicate IDs within each split, and ID overlap across splits. It also validates `test.csv` if the organizers release it. It never prints prompt contents. In code, use `load_train()` or `load_dev()` for labeled records. Use `load_predictions("devtest")` or `load_predictions("test")` for label-free prediction records. The prediction loader rejects other split names.

## Scoring objective

The primary metric is the Safety Score, a clamped form of Youden's J:

```text
Safety Score = max(0, R_U + R_S - 100)
```

`R_U` is unsafe recall: the percentage of unsafe examples predicted `U`. `R_S` is safe recall: the percentage of safe examples predicted `S`. A missed attack and a false alarm each reduce one of these recalls. An unlabeled prompt counts as wrong. The official leaderboard ranks teams by Safety Score; it also displays both recalls. A system that predicts only one class scores zero. Therefore model selection must consider both classes rather than optimize unsafe recall alone.

Use the released scorer for local metric and file-format checks. Do not substitute ordinary accuracy for the competition metric.

## Competition workflow and dates

All stated deadlines are **11:59 PM Anywhere on Earth (AoE)** unless otherwise specified. Dates can change; recheck the official pages before submitting.

1. **Development phase — Oct 29, 2026 to Jan 7, 2027:** obtain train/dev/dev-test data, the scorer, and starter notebook; train and tune; practice with dev-test submissions on CodaBench.
2. **Test phase — Jan 8 to Jan 24, 2027:** obtain the unlabeled 30,000-row test set on Jan 8, generate one prediction per prompt, and upload the final labels by Jan 24.
3. **Results — Jan 25, 2027:** official results are scheduled for release.
4. **System paper:** only teams that participate in the test phase may submit. The system description is up to four pages using the official ACL two-column template (with the camera-ready allowance described on the official page); submit by Feb 5, 2027. Notification is Mar 26 and camera-ready is due Apr 16, 2027.
5. **Workshop:** LangCode 2027 is co-located with NAACL in San Francisco, June 1–5, 2027.

Competition participation is through CodaBench. The leaderboard page says all submissions for a team must come through one CodaBench account; a team may have at most five members. Confirm current participation rules on the official pages.

## What must be settled before implementing the detector

These are the concrete prerequisites for writing reliable training and submission code:

1. **Obtain the official starter kit under the research-use terms.** Complete. The currently released train, dev, dev-test, scorer, and starter notebook are stored locally under `data/raw/`; those files are Git-ignored.
2. **Confirm the submission contract.** Complete. The scorer's format contract and invocation are recorded above.
3. **Inspect the released files and scorer.** Complete for the released CSVs and scorer: headers, UTF-8 decoding, row counts, IDs, empty values, labels, and scorer commands are recorded above.
4. **Choose an initial model and runtime.** The official starter notebook fine-tunes BERT-base on a free T4 GPU and describes an approximately 15-minute run. Decide whether to reproduce this baseline first, what local/Colab hardware is available, and whether any additional multilingual model or external public data is appropriate. Model and dependency versions should be selected when implementation begins.
5. **Set validation protocol.** Use train for fitting and dev for model selection. Keep dev-test/test out of fitting and tuning; dev-test is for submission practice and test is for final evaluation only. Track Safety Score, unsafe recall, and safe recall.
6. **Plan a reproducible experiment and final export.** Record seeds, model/configuration, preprocessing, thresholds, and environment. Implement a submission writer only after confirming the schema, then run the official scorer's format check before any CodaBench upload.

## Explainer deck

The repository includes an eight-slide, plain-language visual guide: [presentations/langcode-task2-explainer.pptx](presentations/langcode-task2-explainer.pptx). It explains the classification decision, language and script variation, safe versus unsafe intent, dataset splits, the Safety Score, and competition dates.

## Project layout

```text
LangCode_comp/
├── README.md                 # Task, rules, setup prerequisites, and workflow
├── .gitignore                # Excludes local data, models, environments, and outputs
├── presentations/            # Visual task explainer deck
├── data/
│   ├── raw/                  # Official files kept locally; never commit
│   └── processed/            # Derived local data; never commit
├── configs/                  # Experiment and model configurations
├── notebooks/                # Exploration and reproducible experiments
├── src/langcode_task2/       # Reusable training, evaluation, and inference code
├── scripts/                  # Command-line entry points
├── models/                   # Local checkpoints; never commit
├── outputs/                  # Logs, metrics, and predictions; never commit
└── submissions/              # Locally generated CodaBench archives; never commit
```

Data checking and split-aware loaders are implemented. The starter kit remains local and Git-ignored. No detector, trained checkpoint, or submission writer has been added yet.

## Local setup status

The tracked repository contains task documentation, project directories, and the explainer deck. The downloaded starter-kit data remains local and ignored by Git. There is no detector code or model checkpoint yet. Python packaging and dependency pins are deliberately deferred until the baseline is implemented. A GPU is optional for data inspection and lightweight experiments; reproducing the BERT-base starter run is expected to benefit from a T4 GPU.

## Recommended GitHub issue sequence

This order is saved to make the dependencies visible:

1. **[#3 Add dataset integrity checks and split-safe loading](https://github.com/tanwirzaman/LangCode_comp/issues/3)** — current work; provides safe data APIs and a validation command.
2. **[#2 Reproduce the official BERT-base starter baseline](https://github.com/tanwirzaman/LangCode_comp/issues/2)** — depends on #3; establishes dev Safety Score and recalls.
3. **[#6 Evaluate multilingual robustness and tune for Safety Score](https://github.com/tanwirzaman/LangCode_comp/issues/6)** — compare improvements using train/dev only.
4. **[#1 Confirm the official CodaBench submission schema](https://github.com/tanwirzaman/LangCode_comp/issues/1)** — finish any CodaBench-specific upload details before packaging.
5. **[#7 Build deterministic inference and validated submission packaging](https://github.com/tanwirzaman/LangCode_comp/issues/7)** — depends on the selected model and confirmed submission contract.
6. **[#4 Practice on dev-test and prepare the final test submission](https://github.com/tanwirzaman/LangCode_comp/issues/4)** — follow the competition release dates; test data is scheduled for Jan 8, 2027.
7. **[#5 Prepare the Shared Task 2 system description paper](https://github.com/tanwirzaman/LangCode_comp/issues/5)** — after test-phase participation and results.

## Official references

- [Task overview and schedule](https://langcode2027.github.io/task2/)
- [Data, starter notebook, scorer, and data-use terms](https://langcode2027.github.io/task2/data/)
- [Scoring definition and leaderboard](https://langcode2027.github.io/task2/leaderboard/)
- [CodaBench participation steps](https://langcode2027.github.io/task2/participate/)
- [System paper requirements](https://langcode2027.github.io/task2/paper/)
