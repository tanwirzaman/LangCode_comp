# LangCode 2027 Shared Task 2: Adversarial Prompt Detection

Project workspace for the LangCode 2027 Shared Task 2. The task is to classify a prompt sent to a code model as safe (`S`) or unsafe (`U`) when it is code-mixed with English or English transliterated into another script. This repository is currently a documented, code-free scaffold: the requirements and workflow are recorded here before implementing a model.

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

The organizers state that any public data, models, and tools may be used. However, dev-test and test prompts must be labeled one prompt at a time: do not look them up in other datasets, including sources used to build MALICE. The data page also gives a research-use condition: use the data only to research detection and defense, do not use it to build or improve attacks, do not run unsafe prompts against production systems, and do not redistribute it. **This repository does not download or include the dataset.** Read and accept the official terms yourself before downloading it.

The benchmark is built from seed prompts in MALICE. If the data is used in research, include the citation on the official data page (Raihan et al., Findings of EMNLP 2026).

## Scoring objective

The primary metric is the Safety Score, a clamped form of Youden's J:

```text
Safety Score = max(0, R_U + R_S - 100)
```

`R_U` is unsafe recall: the percentage of unsafe examples predicted `U`. `R_S` is safe recall: the percentage of safe examples predicted `S`. A missed attack and a false alarm each reduce one of these recalls. An unlabeled prompt counts as wrong. The official leaderboard ranks teams by Safety Score; it also displays both recalls. A system that predicts only one class scores zero. Therefore model selection must consider both classes rather than optimize unsafe recall alone.

Use the released scorer for local metric and file-format checks once it has been obtained from the official page. Do not substitute ordinary accuracy for the competition metric.

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

1. **Obtain the official starter kit under the research-use terms.** Download train, dev, dev-test, scorer, and the starter notebook from the [Data page](https://langcode2027.github.io/task2/data/). The dataset itself is intentionally absent from this workspace.
2. **Confirm the submission contract.** The data page says the exact upload format is on the CodaBench Submission page. Confirm the required filename/archive structure, column names/order, row ordering/IDs, and any metadata. The CodaBench link returned unrelated page content during this inspection, so the schema could not be independently verified here. The official scorer should be treated as the executable authority for format validation.
3. **Inspect the released files and scorer.** Verify CSV headers, encodings, nulls, duplicate IDs, allowed labels, scorer invocation, and submission constraints before building the data loader or exporter.
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

The directories are prepared for the next implementation step. No detector, data, model checkpoint, or unverified submission writer has been added yet.

## Local setup status

The repository contains task documentation and empty project directories, plus the explainer deck. It has no detector code, dataset, or model checkpoint. Python packaging and dependency pins are deliberately deferred until the official files and chosen baseline are inspected. Once implementation starts, use a project-local virtual environment and record exact dependencies so the baseline can be reproduced. A GPU is optional for data inspection and lightweight experiments; reproducing the BERT-base starter run is expected to benefit from a GPU such as the T4 mentioned by the organizers.

## Planned implementation sequence

1. Inspect the official starter notebook, CSVs, and scorer; record the exact schema and commands.
2. Add environment/package configuration and a data-integrity check.
3. Reproduce the official baseline on train/dev and report the official Safety Score and both recalls.
4. Add experiment tracking and compare improvements on dev, including error analysis across scripts and safe/unsafe intents where feasible.
5. Implement deterministic inference and a schema-checked submission archive; validate with the official scorer.
6. Practice with dev-test during the development phase, then freeze the chosen system before producing test predictions.

## Official references

- [Task overview and schedule](https://langcode2027.github.io/task2/)
- [Data, starter notebook, scorer, and data-use terms](https://langcode2027.github.io/task2/data/)
- [Scoring definition and leaderboard](https://langcode2027.github.io/task2/leaderboard/)
- [CodaBench participation steps](https://langcode2027.github.io/task2/participate/)
- [System paper requirements](https://langcode2027.github.io/task2/paper/)
