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
from functools import partial
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
MODEL_REVISIONS = {
    "google-bert/bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
    "google-bert/bert-base-multilingual-cased": "3f076fdb1ab68d5b2880cb87a0886f315b8146f8",
    "FacebookAI/xlm-roberta-base": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
}


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
        help="local directory for the trained model and tokenizer",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="where to write machine-readable run settings and metrics",
    )
    parser.add_argument("--run-name", default="bert-base-baseline", help="name for local model/output folders")
    parser.add_argument("--model-id", default=MODEL_ID, help="Hugging Face model id or local checkpoint")
    parser.add_argument("--model-revision", help="optional pinned Hugging Face revision")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.model_revision is None:
        args.model_revision = MODEL_REVISIONS.get(args.model_id)
    args.model_dir = args.model_dir or REPOSITORY_ROOT / "models" / args.run_name
    args.report = args.report or REPOSITORY_ROOT / "outputs" / args.run_name / "run.json"
    if args.max_length <= 0 or args.epochs <= 0:
        parser.error("--max-length and --epochs must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be at least 1")
    return args


def as_dataset(frame: pd.DataFrame, tokenizer, with_labels: bool, max_length: int) -> Dataset:
    values: dict[str, list] = {"text": frame["prompt"].tolist()}
    if with_labels:
        values["labels"] = (frame["label"].to_numpy() == "U").astype(int).tolist()
    dataset = Dataset.from_dict(values)
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"], truncation=True, max_length=max_length
        ),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing prompts",
    )


def unsafe_probabilities(logits: np.ndarray) -> np.ndarray:
    # Match the starter notebook's float32 softmax and class-1 probability.
    return torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()


def compute_dev_metrics(evaluation, threshold: float = 0.5) -> dict[str, float]:
    predictions = np.where(unsafe_probabilities(evaluation.predictions) >= threshold, "U", "S")
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

    load_kwargs = {"revision": args.model_revision} if args.model_revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **load_kwargs)
    train_dataset = as_dataset(train, tokenizer, with_labels=True, max_length=args.max_length)
    dev_dataset = as_dataset(dev, tokenizer, with_labels=True, max_length=args.max_length)
    # The starter notebook only passes seed through TrainingArguments, which
    # seeds the training loop after the classification head is initialized.
    # Seed here too so the randomly initialized head is repeatable.
    set_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        **load_kwargs,
        num_labels=2,
        id2label={0: "S", 1: "U"},
        label2id={"S": 0, "U": 1},
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model: {args.model_id} ({parameter_count / 1e6:.0f}M parameters)", flush=True)

    training_args = TrainingArguments(
        output_dir=str(args.model_dir / "trainer"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=partial(compute_dev_metrics, threshold=args.threshold),
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
    final_labels = np.where(probabilities >= args.threshold, "U", "S")
    scorer_output, official_metrics = official_dev_score(
        dev["id"], final_labels, dev_csv, scorer, args.report.parent
    )
    pd.DataFrame(
        {"id": dev["id"], "label": dev["label"], "p_unsafe": probabilities}
    ).to_csv(args.report.parent / "dev_probabilities.csv", index=False)
    trainer.save_model(str(args.model_dir))
    tokenizer.save_pretrained(args.model_dir)

    package_versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "datasets", "accelerate", "numpy", "pandas")
    }
    report = {
        "model_id": args.model_id,
        "model_revision": args.model_revision or getattr(model.config, "_commit_hash", None),
        "model_parameters": parameter_count,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "settings": {
            "max_length": args.max_length,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_train_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "threshold": args.threshold,
            "seed": args.seed,
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
