# Multilingual robustness experiments

## Scope and data handling

All experiments used the released 20,000-row train set for fitting and the 10,000-row dev set for evaluation and threshold selection. No dev-test or test rows were loaded. Every run used two epochs, learning rate `2e-5`, 6% warmup, weight decay `0.01`, effective train batch size 32, seed 42 or 43, and fp16 on the RTX 5070 Ti. Model revisions are pinned in the run commands below.

The official task page lists ten target languages/scripts and says the language and code-mixed versus transliterated form are not provided as annotations. It also says the two prompt forms are equally represented. Therefore this report does not claim form-specific or language-specific metrics. Instead, it reports descriptive Unicode-script slices: each dev prompt is grouped by its dominant character script among the ten target script families. Han without kana is grouped as Chinese/Han; Japanese kana and Korean Hangul are recognized separately. A 34-row Latin-only/other group and two rows with tied target scripts are shown separately and should not be interpreted as task language labels. No prompt text is printed or saved by the analysis. [Official task data description](https://langcode2027.github.io/task2/data/)

## Experiments and scores

All metrics are on the 10,000-row dev set. “At 0.50” and “Best dev threshold” report Safety Score, unsafe recall (R_U), and safe recall (R_S). The threshold sweep evaluates `p(U)` from 0.20 through 0.80 in increments of 0.01 using the official scorer implementation. The final selected submission-shaped CSV was also checked with the official scorer CLI.

| Candidate | Seed | Effective batch | Max tokens | Train time (s) | At 0.50: Safety / R_U / R_S | Best threshold | Best dev: Safety / R_U / R_S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BERT-base, `bert-base-uncased` | 42 | 32 | 256 | 140.02 | 51.18 / 85.28 / 65.90 | 0.44 | 52.26 / 90.14 / 62.12 |
| BERT-base, longer context | 42 | 32 (16 × 2 accumulation) | 512 | 260.98 | 60.12 / 85.90 / 74.22 | 0.39 | 60.60 / 92.50 / 68.10 |
| mBERT cased | 42 | 32 | 256 | 145.12 | 74.42 / 89.02 / 85.40 | 0.41 | 74.92 / 93.18 / 81.74 |
| mBERT cased | 43 | 32 | 256 | 145.77 | 74.12 / 89.96 / 84.16 | 0.25 | 74.68 / 94.94 / 79.74 |
| XLM-R base | 42 | 32 (16 × 2 accumulation) | 256 | 154.30 | 73.68 / 87.72 / 85.96 | 0.26 | 74.26 / 93.46 / 80.80 |
| XLM-R base | 43 | 32 (16 × 2 accumulation) | 256 | 154.30 | 75.06 / 90.16 / 84.90 | **0.52** | **75.10 / 89.50 / 85.60** |

The multilingual encoders outperform English-only BERT on dev by a wide margin. Extending BERT's context from 256 to 512 tokens improves its score by 8.94 points at the default threshold, but still trails the multilingual candidates. Model choice matters substantially here: the mBERT and XLM-R runs land close to each other, so the per-seed results and threshold are retained in the output files instead of hiding run variation behind a single number.

### Model and preprocessing choices

- The baseline is the official English-only `google-bert/bert-base-uncased` model (109.5M parameters).
- `google-bert/bert-base-multilingual-cased` (177.9M parameters) is pretrained across 104 languages; it is a useful same-family multilingual comparison. [Model card](https://huggingface.co/google-bert/bert-base-multilingual-cased)
- `FacebookAI/xlm-roberta-base` (278.0M parameters) is a multilingual RoBERTa model pretrained across 100 languages. [Model card](https://huggingface.co/FacebookAI/xlm-roberta-base)
- The 512-token experiment changes only the maximum input length and physical batch size; gradient accumulation preserves an effective train batch of 32. All other runs use max length 256.
- Inputs are the raw `prompt` field passed through each model's own tokenizer. No translation, transliteration, keyword filtering, or other hand-built lexical rule is used.

Each run uses two epochs, learning rate `2e-5`, linear decay, 6% warmup, weight decay `0.01`, labels `S=0` and `U=1`, and fp16. The mBERT/XLM-R runs use the same seed settings for the first comparison; seed 43 was added for a second run of both multilingual encoders. Training is not bit-identical across GPU runs even with a seed: the BERT-base reference run in [the baseline report](baseline-bert-base.md) scored 53.56, while the later same-settings rerun scored 51.18. Treat differences below a few points as provisional; the dev set is also used for model and threshold selection.

## Selected candidate

The current candidate for the next packaging step is **XLM-R base, seed 43, max length 256, threshold 0.52**. It produced the best observed dev Safety Score in this comparison: 75.10, with 89.50 unsafe recall and 85.60 safe recall. This is a dev-selected result, not an unbiased estimate of final test performance. mBERT is a close and smaller alternative: its best observed dev score was 74.92, and its mean threshold-tuned score across seeds 42 and 43 was 74.80 versus 74.68 for XLM-R. The single XLM-R seed-43 result wins the selection criterion by only 0.18 points, so the two should be treated as practically close.

The chosen checkpoint is kept locally at `models/xlm-roberta-base-256-seed43/` and ignored by Git. Its tracked configuration, including model revision, seed, threshold, and label map, is [configs/candidate-xlm-roberta.yaml](../configs/candidate-xlm-roberta.yaml). The threshold sweep shifts the seed-43 XLM-R trade-off from 90.16/84.90 recalls at 0.50 to 89.50/85.60 at 0.52, a slightly more balanced pair.

## Script-slice error review

These metrics are for the selected XLM-R seed-43 candidate at threshold 0.52. R_U and R_S use the official formulas within each group; false-unsafe counts are safe prompts predicted U, and missed-unsafe counts are unsafe prompts predicted S.

| Unicode-script proxy | n | Safe / unsafe | Safety Score | R_U | R_S | False unsafe | Missed unsafe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arabic | 987 | 471 / 516 | 72.23 | 86.24 | 85.99 | 66 | 71 |
| Bengali | 1,039 | 542 / 497 | 75.80 | 90.74 | 85.06 | 81 | 46 |
| Chinese/Han | 943 | 489 / 454 | 77.14 | 89.21 | 87.93 | 59 | 49 |
| Cyrillic | 1,042 | 519 / 523 | 76.56 | 91.20 | 85.36 | 76 | 46 |
| Devanagari | 962 | 469 / 493 | 75.57 | 89.86 | 85.71 | 67 | 50 |
| Hebrew | 953 | 489 / 464 | 72.90 | 85.99 | 86.91 | 64 | 65 |
| Japanese kana | 967 | 472 / 495 | 77.52 | 91.72 | 85.81 | 67 | 41 |
| Korean Hangul | 1,040 | 542 / 498 | 76.40 | 93.37 | 83.03 | 92 | 33 |
| Tamil | 1,043 | 519 / 524 | 72.37 | 87.79 | 84.59 | 80 | 64 |
| Thai | 988 | 473 / 515 | 76.73 | 90.68 | 86.05 | 66 | 48 |
| No target non-Latin script | 34 | 15 / 19 | 23.51 | 36.84 | 86.67 | 2 | 12 |
| Multiple target scripts | 2 | 0 / 2 | 0.00 | 100.00 | 0.00 | 0 | 0 |

The ten main script groups are close to 1,000 examples each. Within this single dev split, Arabic, Hebrew, and Tamil have the lowest Safety Scores among the target-script groups, while Chinese/Han and Japanese kana have the highest. These are descriptive slices from one selected model and one dev set; they are not proof of language-level differences. The two tiny miscellaneous groups are especially unreliable. Form-specific comparisons cannot be performed because individual code-mixed/transliterated labels are absent.

## Reproduce

The environment is the same one recorded in the baseline report: Python 3.12.13, PyTorch 2.15.0+cu132, Transformers 4.57.6, Datasets 5.0.1, Accelerate 1.16.0, NumPy 2.5.5, Pandas 3.1.1, Windows 11, and an NVIDIA GeForce RTX 5070 Ti. Direct package pins are in `requirements-baseline.txt`. Start with the baseline command, then run model variants from the repository root:

```powershell
.venv\Scripts\python.exe scripts\train_baseline.py --run-name bert-base-baseline --model-id google-bert/bert-base-uncased --model-revision 86b5e0934494bd15c9632b12f734a8a67f723594 --max-length 256 --epochs 2 --train-batch-size 32 --eval-batch-size 128 --seed 42
.venv\Scripts\python.exe scripts\train_baseline.py --run-name bert-base-512 --model-id google-bert/bert-base-uncased --model-revision 86b5e0934494bd15c9632b12f734a8a67f723594 --max-length 512 --epochs 2 --train-batch-size 16 --gradient-accumulation-steps 2 --eval-batch-size 64 --seed 42
.venv\Scripts\python.exe scripts\train_baseline.py --run-name mbert-cased-256 --model-id google-bert/bert-base-multilingual-cased --model-revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8 --max-length 256 --epochs 2 --train-batch-size 32 --eval-batch-size 128 --seed 42
.venv\Scripts\python.exe scripts\train_baseline.py --run-name mbert-cased-256-seed43 --model-id google-bert/bert-base-multilingual-cased --model-revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8 --max-length 256 --epochs 2 --train-batch-size 32 --eval-batch-size 128 --seed 43
.venv\Scripts\python.exe scripts\train_baseline.py --run-name xlm-roberta-base-256 --model-id FacebookAI/xlm-roberta-base --model-revision e73636d4f797dec63c3081bb6ed5c7b0bb3f2089 --max-length 256 --epochs 2 --train-batch-size 16 --gradient-accumulation-steps 2 --eval-batch-size 128 --seed 42
.venv\Scripts\python.exe scripts\train_baseline.py --run-name xlm-roberta-base-256-seed43 --model-id FacebookAI/xlm-roberta-base --model-revision e73636d4f797dec63c3081bb6ed5c7b0bb3f2089 --max-length 256 --epochs 2 --train-batch-size 16 --gradient-accumulation-steps 2 --eval-batch-size 128 --seed 43
.venv\Scripts\python.exe scripts\analyze_experiments.py --experiment bert-base-256=outputs/bert-base-baseline/dev_probabilities.csv --experiment bert-base-512=outputs/bert-base-512/dev_probabilities.csv --experiment mbert-seed42=outputs/mbert-cased-256/dev_probabilities.csv --experiment mbert-seed43=outputs/mbert-cased-256-seed43/dev_probabilities.csv --experiment xlm-roberta-seed42=outputs/xlm-roberta-base-256/dev_probabilities.csv --experiment xlm-roberta-seed43=outputs/xlm-roberta-base-256-seed43/dev_probabilities.csv --output-dir outputs/robustness-analysis
```

The analysis command writes the full threshold sweep, model summary, selected dev predictions, script slices, and selection metadata under `outputs/robustness-analysis/`. These artifacts and the trained checkpoints are local and Git-ignored. All scripts use only train/dev; no keyword-only rejection rule is introduced.
