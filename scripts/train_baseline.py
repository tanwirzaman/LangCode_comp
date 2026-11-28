"""Reproduce the official BERT-base Task 2 starter baseline.

The defaults mirror the released notebook: bert-base-uncased, 256-token
truncation, two epochs, train batch size 32, learning rate 2e-5, 6% warmup,
seed 42, and fp16 when CUDA is available. Train is used for fitting and dev
for epoch evaluation. No dev-test or test records are loaded by this script.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "google-bert/bert-base-uncased"
MAX_LENGTH = 256
EPOCHS = 2
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.06
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "raw",
        help="directory containing train.csv, dev.csv, and scorer/score.py",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPOSITORY_ROOT / "models" / "bert-base-baseline",
        help="local directory for the trained model and tokenizer",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "bert-base-baseline" / "run.json",
        help="where to write machine-readable run settings and metrics",
    )
    return parser.parse_args()


def as_dataset(frame: pd.DataFrame, tokenizer, with_labels: bool) -> Dataset:
    values: dict[str, list] = {"text": frame["prompt"].tolist()}
    if with_labels:
        values["labels"] = (frame["label"].to_numpy() == "U").astype(int).tolist()
    dataset = Dataset.from_dict(values)
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"], truncation=True, max_length=MAX_LENGTH
        ),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing prompts",
    )


def unsafe_probabilities(logits: np.ndarray) -> np.ndarray:
    # Match the starter notebook's float32 softmax and class-1 probability.
    return torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()


def compute_dev_metrics(evaluation) -> dict[str, float]:
    predictions = np.where(unsafe_probabilities(evaluation.predictions) >= 0.5, "U", "S")
    gold = np.where(evaluation.label_ids == 1, "U", "S")
    unsafe_recall = 100.0 * np.mean(predictions[gold == "U"] == "U")
    safe_recall = 100.0 * np.mean(predictions[gold == "S"] == "S")
    safety = max(0.0, unsafe_recall + safe_recall - 100.0)
    return {"safety": safety, "R_U": unsafe_recall, "R_S": safe_recall}


def official_dev_score(
    ids: pd.Series, labels: np.ndarray, dev_csv: Path, scorer: Path, work_dir: Path
) -> tuple[str, dict[str, float]]:
    prediction_path = work_dir / "dev_predictions.csv"
    pd.DataFrame({"id": ids, "label": labels}).to_csv(prediction_path, index=False)
    result = subprocess.run(
        [sys.executable, str(scorer), "--score", str(prediction_path), str(dev_csv)],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    match = re.search(
        r"Safety Score ([\d.]+) \| R_U \(unsafe recall\) ([\d.]+) \| "
        r"R_S \(safe recall\) ([\d.]+)",
        output,
    )
    if not match:
        raise RuntimeError(f"Could not parse official scorer output: {output}")
    safety, unsafe_recall, safe_recall = map(float, match.groups())
    return output, {
        "safety_score": safety,
        "unsafe_recall": unsafe_recall,
        "safe_recall": safe_recall,
    }


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    train_csv = data_dir / "train.csv"
    dev_csv = data_dir / "dev.csv"
    scorer = data_dir / "scorer" / "score.py"
    for required in (train_csv, dev_csv, scorer):
        if not required.is_file():
            raise SystemExit(f"Required file is missing: {required}")

    # Match the starter notebook's CSV parsing and keep all prompts out of logs.
    read = lambda path: pd.read_csv(path, dtype=str, keep_default_na=False)
    train, dev = read(train_csv), read(dev_csv)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}", flush=True)
    print(f"Train rows: {len(train):,}; dev rows: {len(dev):,}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_dataset = as_dataset(train, tokenizer, with_labels=True)
    dev_dataset = as_dataset(dev, tokenizer, with_labels=True)
    # The starter notebook only passes seed through TrainingArguments, which
    # seeds the training loop after the classification head is initialized.
    # Seed here too so the randomly initialized head is repeatable.
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        id2label={0: "S", 1: "U"},
        label2id={"S": 0, "U": 1},
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model: {MODEL_ID} ({parameter_count / 1e6:.0f}M parameters)", flush=True)

    training_args = TrainingArguments(
        output_dir=str(args.model_dir / "trainer"),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_dev_metrics,
    )

    started = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - started
    epoch_metrics = [
        {
            "epoch": int(round(item["epoch"])),
            "safety_score": round(float(item["eval_safety"]), 4),
            "unsafe_recall": round(float(item["eval_R_U"]), 4),
            "safe_recall": round(float(item["eval_R_S"]), 4),
        }
        for item in trainer.state.log_history
        if "eval_safety" in item
    ]

    probabilities = unsafe_probabilities(trainer.predict(dev_dataset).predictions)
    final_labels = np.where(probabilities >= 0.5, "U", "S")
    scorer_output, official_metrics = official_dev_score(
        dev["id"], final_labels, dev_csv, scorer, args.report.parent
    )
    trainer.save_model(str(args.model_dir))
    tokenizer.save_pretrained(args.model_dir)

    package_versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "datasets", "accelerate", "numpy", "pandas")
    }
    report = {
        "model_id": MODEL_ID,
        "model_parameters": parameter_count,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "settings": {
            "max_length": MAX_LENGTH,
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": 0.01,
            "threshold": 0.5,
            "seed": SEED,
            "fp16": torch.cuda.is_available(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_cuda": torch.version.cuda,
            "device": device_name,
            "device_capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None,
            "packages": package_versions,
        },
        "training_seconds": round(training_seconds, 2),
        "dev_metrics_by_epoch": epoch_metrics,
        "official_scorer_output": scorer_output,
        "official_dev_metrics": official_metrics,
        "model_dir": str(args.model_dir.resolve()),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nOfficial scorer:", scorer_output, flush=True)
    print(f"Training time: {training_seconds / 60:.2f} minutes", flush=True)
    print(f"Model saved to: {args.model_dir.resolve()}", flush=True)
    print(f"Run report saved to: {args.report.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
