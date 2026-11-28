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

### Verified CodaBench submission contract

This contract is taken from the released starter notebook and the official `scorer/score.py` bundled in the starter kit. The [official data page](https://langcode2027.github.io/task2/data/) identifies that scorer as the program used on CodaBench; the [participation instructions](https://langcode2027.github.io/task2/participate/) say to use its format checker before uploading.

**Archive.** The starter notebook creates and uploads `submission.zip`. Use this filename as the team's standard. The outer ZIP name is not checked by the scorer, which accepts a ZIP, folder, or CSV. The canonical ZIP contains one top-level `submission.csv`:

```text
submission.zip
└── submission.csv
```

The scorer requires exactly one archive member named `submission.csv` (it locates that basename); the starter notebook writes the CSV at the ZIP root. Keep the root-level layout to match the official example.

**CSV.** Use UTF-8 text with a header and one prediction row per prompt. The official example writes columns in this order:

```csv
id,label
T2-483920,U
T2-105377,S
```

The required fields are named `id` and `label`. The scorer finds them by header name, so their order is not enforced; it also ignores extra columns. It strips surrounding header/field whitespace, accepts either case for header and labels, and requires labels to be `S` or `U`. IDs must be non-empty and unique. The submission should include exactly the IDs from the phase input, one row per prompt. The scorer does not enforce row order: it matches predictions by ID. Keeping input order is a useful convention, not a scorer requirement. Missing predictions count as wrong; unknown IDs are ignored. A format check with the phase's public CSV reports missing and unknown IDs.

Run the official format checker from the repository root before upload:

```powershell
python data/raw/scorer/score.py --check submission.zip data/raw/devtest.csv
```

Replace `devtest.csv` with the public input CSV for the phase being checked. A complete check reports zero missing and zero unknown IDs and says the file is ready to upload. To score a local prediction CSV against labeled dev data, use:

```powershell
python data/raw/scorer/score.py --score submission.csv data/raw/dev.csv
```

For the CodaBench scorer container itself, the official scorer reads `/app/input/ref/reference.csv` and `/app/input/res/`, then writes `scores.json` under `/app/output/`. Those are the scorer's runtime paths; the participant ZIP uses the layout shown above.

### Starter notebook environment and baseline

The notebook targets Python 3 in Google Colab and recommends a T4 GPU. It installs `transformers>=4.48,<5`, `accelerate`, and `datasets`; Colab supplies PyTorch, NumPy, and pandas. The baseline loads `google-bert/bert-base-uncased`, truncates to 256 tokens, trains for two epochs with batch size 32 and learning rate `2e-5`, evaluates on dev each epoch, and uses seed 42. The notebook estimates about 15 minutes on a free T4. It writes `submission.csv` and `submission.zip`, then checks the dev-test archive with the official scorer.

The repository reproduction uses pinned dependencies in `requirements-baseline.txt` and `scripts/train_baseline.py`. Its dev results, run settings, and local rerun instructions are documented in [reports/baseline-bert-base.md](reports/baseline-bert-base.md). On the RTX 5070 Ti used for that run, training took about 2.33 minutes.

Issue #6 compared multilingual models, context lengths, thresholds, and two seeds for the leading multilingual models. The current dev-selected candidate is XLM-R base, seed 43, with a 256-token input and `p(U) >= 0.52`; it scored 75.10 Safety Score (R_U 89.50, R_S 85.60) on dev. See [the robustness report](reports/robustness-experiments.md) and [candidate config](configs/candidate-xlm-roberta.yaml). The score is dev-selected and should not be read as final test performance.

On Windows with `uv`, create the environment and run the baseline from the repository root:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-baseline.txt
.venv\Scripts\python.exe scripts\validate_data.py
.venv\Scripts\python.exe scripts\train_baseline.py
```

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
4. **Choose an initial model and runtime.** Complete: the official BERT-base starter has been reproduced locally using a GeForce RTX 5070 Ti. Its exact model settings, dependency versions, runtime, and dev metrics are in [the baseline report](reports/baseline-bert-base.md). The baseline is a starting point; multilingual models and additional public data can be considered in follow-up experiments.
5. **Set validation protocol.** Use train for fitting and dev for model selection. Keep dev-test/test out of fitting and tuning; dev-test is for submission practice and test is for final evaluation only. Track Safety Score, unsafe recall, and safe recall.
6. **Plan a reproducible experiment and final export.** Record seeds, model/configuration, preprocessing, thresholds, and environment. Implement a submission writer only after confirming the schema, then run the official scorer's format check before any CodaBench upload.

## Explainer deck

The repository includes an eight-slide, plain-language visual guide: [presentations/langcode-task2-explainer.pptx](presentations/langcode-task2-explainer.pptx). It explains the classification decision, language and script variation, safe versus unsafe intent, dataset splits, the Safety Score, and competition dates.

## Project layout

```text
LangCode_comp/
├── README.md                 # Task, rules, setup prerequisites, and workflow
├── .gitignore                # Excludes local data, models, environments, and outputs
├── requirements-baseline.txt # Pinned dependencies for the BERT-base reproduction
├── presentations/            # Visual task explainer deck
├── reports/                  # Tracked experiment and baseline reports
├── data/
│   ├── raw/                  # Official files kept locally; never commit
│   └── processed/            # Derived local data; never commit
├── configs/                  # Experiment and model configurations
├── notebooks/                # Exploration and reproducible experiments
├── src/langcode_task2/       # Split-aware dataset loading and validation
├── scripts/                  # Data validation and baseline training commands
├── models/                   # Local checkpoints; never commit
├── outputs/                  # Logs, metrics, and predictions; never commit
└── submissions/              # Locally generated CodaBench archives; never commit
```

Data checking, split-aware loaders, and the official BERT-base baseline are implemented. The starter kit and trained checkpoint remain local and Git-ignored. The baseline checkpoint is saved under `models/bert-base-baseline/`; run details and dev predictions are saved under `outputs/bert-base-baseline/`.

## Local setup status

The tracked repository contains task documentation, split-safe data utilities, the baseline training script, pinned baseline dependencies, the baseline report, and the explainer deck. The downloaded starter-kit data, trained checkpoint, dev predictions, and machine-readable run output remain local and ignored by Git. The baseline has been run on a GeForce RTX 5070 Ti with CUDA 13.2; a CUDA-capable GPU is recommended for practical training speed.

## Recommended GitHub issue sequence

This order is saved to make the dependencies visible:

1. **[x #3 Add dataset integrity checks and split-safe loading](https://github.com/tanwirzaman/LangCode_comp/issues/3)** — complete; provides safe data APIs and a validation command.
2. **[x #2 Reproduce the official BERT-base starter baseline](https://github.com/tanwirzaman/LangCode_comp/issues/2)** — complete; dev metrics are recorded in the baseline report.
3. **[x #6 Evaluate multilingual robustness and tune for Safety Score](https://github.com/tanwirzaman/LangCode_comp/issues/6)** — complete; variants, thresholds, seed runs, script slices, and selected candidate are documented.
4. **[x #1 Confirm the official CodaBench submission schema](https://github.com/tanwirzaman/LangCode_comp/issues/1)** — complete; the official starter archive, CSV fields, coverage check, and ID semantics are documented above.
5. **[#7 Build deterministic inference and validated submission packaging](https://github.com/tanwirzaman/LangCode_comp/issues/7)** — depends on the selected model and confirmed submission contract.
6. **[#4 Practice on dev-test and prepare the final test submission](https://github.com/tanwirzaman/LangCode_comp/issues/4)** — follow the competition release dates; test data is scheduled for Jan 8, 2027.
7. **[#5 Prepare the Shared Task 2 system description paper](https://github.com/tanwirzaman/LangCode_comp/issues/5)** — after test-phase participation and results.

## Official references

- [Task overview and schedule](https://langcode2027.github.io/task2/)
- [Data, starter notebook, scorer, and data-use terms](https://langcode2027.github.io/task2/data/)
- [Scoring definition and leaderboard](https://langcode2027.github.io/task2/leaderboard/)
- [CodaBench participation steps](https://langcode2027.github.io/task2/participate/)
- [System paper requirements](https://langcode2027.github.io/task2/paper/)
