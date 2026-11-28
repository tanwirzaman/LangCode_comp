# BERT-base starter baseline

This run reproduces the official Task 2 starter notebook on the released train and dev splits. Training uses only `train.csv`; `dev.csv` is used for evaluation after each epoch and final scoring. Dev-test and test are not loaded.

## Results

The final dev predictions were scored with the official `data/raw/scorer/score.py`:

| Epoch | Safety Score | Unsafe recall (R_U) | Safe recall (R_S) |
| ---: | ---: | ---: | ---: |
| 1 | 49.84 | 88.18 | 61.66 |
| 2 | **53.56** | **86.58** | **66.98** |

Official scorer output:

```text
Safety Score 53.56 | R_U (unsafe recall) 86.58 | R_S (safe recall) 66.98 | missing 0, unknown ids 0
```

## Configuration

| Setting | Value |
| --- | --- |
| Model | `google-bert/bert-base-uncased` (109,483,778 parameters) |
| Inputs | `prompt` column only; no additional text normalization |
| Truncation | 256 tokens |
| Train/dev rows | 20,000 / 10,000 |
| Epochs | 2 |
| Train/eval batch size | 32 / 128 |
| Learning rate | `2e-5`, linear decay |
| Warmup | 6% of steps |
| Weight decay | 0.01 |
| Output classes | `S=0`, `U=1` |
| Decision threshold | Predict `U` when its probability is at least 0.5 |
| Seed | 42 |
| Precision | fp16 |
| Training time | 140.00 seconds (2.33 minutes) |

The released notebook sets seed 42 through `TrainingArguments`; that seed is applied when training starts, after the classification head is initialized. The repository script also sets seed 42 before model construction so that the randomly initialized head is seeded. This preserves the notebook's model and training settings while making the initialization repeatable.

## Environment

| Component | Version / device |
| --- | --- |
| OS | Windows 11 (10.0.26200) |
| Python | 3.12.13 |
| GPU | NVIDIA GeForce RTX 5070 Ti, compute capability 12.0 |
| PyTorch CUDA build | CUDA 13.2 |
| `torch` | `2.15.0+cu132` |
| `transformers` | `4.57.6` |
| `datasets` | `5.0.1` |
| `accelerate` | `1.16.0` |
| `numpy` | `2.5.5` |
| `pandas` | `3.1.1` |

Direct dependency pins are in [`requirements-baseline.txt`](../requirements-baseline.txt). The machine-readable output of the most recent run is saved locally at `outputs/bert-base-baseline/run.json`; model weights and run output are ignored by Git.

## Reproduce

From the repository root in PowerShell:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-baseline.txt
.venv\Scripts\python.exe scripts\validate_data.py
.venv\Scripts\python.exe scripts\train_baseline.py
```

The script saves the model and tokenizer under `models/bert-base-baseline/`, official dev predictions under `outputs/bert-base-baseline/dev_predictions.csv`, and run details under `outputs/bert-base-baseline/run.json`. These local artifacts are not committed. The model weights can vary slightly with hardware, CUDA kernels, and library versions; use the pinned dependencies and seed for comparable reruns.
