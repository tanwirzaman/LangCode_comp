"""Run the selected local classifier and package a CodaBench submission.

The script only loads label-free ``devtest`` or ``test`` records. It preserves
their ID order, applies the threshold recorded in the YAML candidate config,
and runs the official scorer's format and coverage check before publishing the
ZIP at its final path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Set this before importing torch so deterministic CUDA kernels can be selected.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.langcode_task2.data import EXPECTED_ROWS, load_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "candidate-xlm-roberta.yaml")
    parser.add_argument("--split", choices=("devtest", "test"), default="devtest")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-dir", type=Path, help="override the checkpoint path in the YAML config")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.output_dir is None:
        args.output_dir = REPOSITORY_ROOT / "outputs" / "submissions" / args.split
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    label_to_id = config["classification"]["label_to_id"]
    if label_to_id != {"S": 0, "U": 1}:
        raise ValueError(f"Expected label map {{'S': 0, 'U': 1}}, found {label_to_id!r}")
    threshold = float(config["classification"]["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The configured unsafe threshold must be between 0 and 1")
    max_length = int(config["model"]["max_length"])
    if max_length < 1:
        raise ValueError("Configured max_length must be positive")

    checkpoint = args.model_dir or resolve_path(Path(config["model"]["local_checkpoint"]))
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Local model checkpoint not found: {checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config["training"]["seed"])
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        local_files_only=True,
        attn_implementation="eager",
    )
    if model.config.num_labels != 2:
        raise ValueError(f"Expected a two-class checkpoint, found {model.config.num_labels} labels")
    model.to(device)
    model.eval()

    records = load_predictions(args.split, args.data_dir)
    expected_rows = EXPECTED_ROWS[args.split]
    if len(records) != expected_rows:
        raise ValueError(f"Expected {expected_rows} {args.split} rows, found {len(records)}")

    predictions: list[str] = []
    amp_enabled = device.type == "cuda" and bool(config["training"].get("fp16", False))
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            encoded = tokenizer(
                [record.prompt for record in batch],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(**encoded).logits
            unsafe_probabilities = logits.float().softmax(dim=-1)[:, label_to_id["U"]]
            predictions.extend("U" if p >= threshold else "S" for p in unsafe_probabilities.cpu().tolist())

    if len(predictions) != len(records):
        raise RuntimeError("Internal error: prediction count does not match input rows")

    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer, lineterminator="\n")
    writer.writerow(("id", "label"))
    writer.writerows((record.id, label) for record, label in zip(records, predictions, strict=True))
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "submission.csv"
    zip_path = output_dir / "submission.zip"
    csv_tmp = output_dir / ".submission.csv.tmp"
    zip_tmp = output_dir / ".submission.zip.tmp"
    csv_tmp.write_bytes(csv_bytes)
    try:
        with zipfile.ZipFile(zip_tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            info = zipfile.ZipInfo("submission.csv", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, csv_bytes, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

        scorer = args.data_dir / "scorer" / "score.py"
        if not scorer.is_file():
            raise FileNotFoundError(f"Official scorer not found: {scorer}")
        public_csv = args.data_dir / f"{args.split}.csv"
        check = subprocess.run(
            [sys.executable, str(scorer), "--check", str(zip_tmp), str(public_csv)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        print(check.stdout.rstrip())
        if "The file is ready to upload." not in check.stdout:
            raise RuntimeError("Official scorer did not confirm full ID coverage")
        os.replace(csv_tmp, csv_path)
        os.replace(zip_tmp, zip_path)
    finally:
        csv_tmp.unlink(missing_ok=True)
        zip_tmp.unlink(missing_ok=True)

    print(f"Split: {args.split} ({len(records)} prompts); device: {device}; seed: {seed}")
    print(f"Threshold: {threshold:.4f}; batch size: {args.batch_size}; max length: {max_length}")
    print(f"CSV: {csv_path}")
    print(f"ZIP: {zip_path}")
    print(f"CSV SHA-256: {hashlib.sha256(csv_bytes).hexdigest()}")
    print(f"Labels: S={predictions.count('S')}, U={predictions.count('U')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
