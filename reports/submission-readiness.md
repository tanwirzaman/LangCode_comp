# Submission Readiness Log

This file tracks the development practice submission and final-test handoff for LangCode 2027 Shared Task 2. It must not contain prompt text or labels from a held-out split.

## Development phase status — November 28, 2026

- **Selected system:** XLM-R base, seed 43; pinned base revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`; fine-tuned checkpoint stored locally at `models/xlm-roberta-base-256-seed43/`.
- **Tracked configuration:** [`configs/candidate-xlm-roberta.yaml`](../configs/candidate-xlm-roberta.yaml); max length 256; unsafe threshold 0.52.
- **Model selection boundary:** fitting and threshold selection used train/dev only. The experiment report states that no dev-test/test rows were loaded for tuning. Dev-test is used here only for the competition's practice submission.
- **Practice artifact:** `outputs/submissions/devtest/submission.zip` (local and Git-ignored); CSV SHA-256 `6162ab2ad52948bdcbbc7a3cd498bd5d4980c241d279185d688f3611bdaad56d`.
- **Local official format/coverage check:** 10,000 predictions; 0 missing IDs; 0 unknown IDs; scorer said “The file is ready to upload.” Two local inference runs produced byte-identical CSV files and the source row order was verified.
- **CodaBench practice result:** pending. Do not record a leaderboard score until the development archive has actually been submitted through the team's CodaBench account.

## Final test phase handoff

The official schedule currently lists test release on **January 8, 2027**, and the test-submission deadline as **January 24, 2027 at 11:59 PM AoE**. Recheck the official [schedule](https://langcode2027.github.io/task2/) and [participation instructions](https://langcode2027.github.io/task2/participate/) before test-phase actions because dates can change.

When `data/raw/test.csv` is released:

1. Confirm it is the official 30,000-row, label-free test file using `python scripts/validate_data.py`.
2. Keep the selected model/configuration frozen unless an allowed development-phase result justifies a change before the development phase closes. Record any change in this file and the experiment report.
3. Generate the final archive with `.venv/Scripts/python.exe scripts/create_submission.py --split test`.
4. Confirm the official scorer reports 30,000 predictions, zero missing IDs, zero unknown IDs, and “The file is ready to upload.” The script gates the final ZIP on this check.
5. Upload the ZIP through the team's single CodaBench account before the deadline; verify the completed submission and save its score/result below. The latest finished test submission is the ranked one, so avoid replacing a better completed submission without a reason.

Follow the official data restriction: label the held-out prompts with the system, do not look them up in external datasets (including MALICE source datasets), do not use them to improve attacks, and do not redistribute the source data. The inference command only reads the local label-free CSV and local saved model; it does not perform prompt retrieval.

## Submission result ledger

| Phase | Submission date/time (AoE) | Model/config version | Artifact SHA-256 | Safety Score | Unsafe recall | Safe recall | CodaBench status/result |
|---|---|---|---|---:|---:|---:|---|
| Dev-test | Pending upload | XLM-R base seed 43, threshold 0.52 | `6162ab2ad52948bdcbbc7a3cd498bd5d4980c241d279185d688f3611bdaad56d` (CSV) | Pending | Pending | Pending | Local check passed; CodaBench score not yet recorded |
| Test | Pending release | Pending | Pending | Pending | Pending | Pending | Pending |
