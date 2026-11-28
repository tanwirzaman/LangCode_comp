"""Split-aware CSV loading and integrity checks for Task 2.

Only train and dev are exposed as labeled records. Dev-test and test are
exposed as label-free prediction records so they cannot accidentally enter a
training or tuning path through this module.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


LabeledSplit = Literal["train", "dev"]
PredictionSplit = Literal["devtest", "test"]
SplitName = Literal["train", "dev", "devtest", "test"]

EXPECTED_HEADERS: dict[SplitName, tuple[str, ...]] = {
    "train": ("id", "prompt", "label"),
    "dev": ("id", "prompt", "label"),
    "devtest": ("id", "prompt"),
    "test": ("id", "prompt"),
}

# Official release sizes. The test split is optional until organizers publish it.
EXPECTED_ROWS: dict[SplitName, int] = {
    "train": 20_000,
    "dev": 10_000,
    "devtest": 10_000,
    "test": 30_000,
}

EXPECTED_LABEL_COUNTS: dict[LabeledSplit, dict[str, int]] = {
    "train": {"S": 10_000, "U": 10_000},
    "dev": {"S": 5_000, "U": 5_000},
}

REQUIRED_SPLITS: tuple[SplitName, ...] = ("train", "dev", "devtest")


@dataclass(frozen=True, slots=True)
class LabeledPrompt:
    """A prompt from train or dev, with its gold label."""

    id: str
    prompt: str
    label: Literal["S", "U"]


@dataclass(frozen=True, slots=True)
class PredictionPrompt:
    """A prompt from dev-test or test; this type intentionally has no label."""

    id: str
    prompt: str


@dataclass(frozen=True, slots=True)
class SplitSummary:
    split: SplitName
    rows: int
    safe: int = 0
    unsafe: int = 0


class DatasetValidationError(ValueError):
    """Raised when an official CSV does not match its required schema."""


def _read_rows(split: SplitName, data_dir: str | Path) -> list[dict[str, str]]:
    path = Path(data_dir) / f"{split}.csv"
    if not path.is_file():
        raise DatasetValidationError(f"Missing required file: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = tuple(reader.fieldnames or ())
            expected = EXPECTED_HEADERS[split]
            if headers != expected:
                raise DatasetValidationError(
                    f"{path}: expected columns {expected}, found {headers}"
                )

            rows: list[dict[str, str]] = []
            seen_ids: set[str] = set()
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise DatasetValidationError(
                        f"{path}:{line_number}: row has more fields than the header"
                    )

                prompt_id = row.get("id") or ""
                prompt = row.get("prompt") or ""
                if not prompt_id.strip():
                    raise DatasetValidationError(
                        f"{path}:{line_number}: id is empty"
                    )
                if prompt_id != prompt_id.strip():
                    raise DatasetValidationError(
                        f"{path}:{line_number}: id has leading or trailing whitespace"
                    )
                if prompt_id in seen_ids:
                    raise DatasetValidationError(
                        f"{path}:{line_number}: duplicate id"
                    )
                if not prompt.strip():
                    raise DatasetValidationError(
                        f"{path}:{line_number}: prompt is empty"
                    )

                if split in ("train", "dev"):
                    label = row.get("label") or ""
                    if label not in ("S", "U"):
                        raise DatasetValidationError(
                            f"{path}:{line_number}: label must be exactly S or U"
                        )
                    row["label"] = label

                seen_ids.add(prompt_id)
                rows.append(row)
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(f"{path}: file is not valid UTF-8") from exc
    except csv.Error as exc:
        raise DatasetValidationError(f"{path}: CSV parsing failed: {exc}") from exc

    return rows


def load_train(data_dir: str | Path = "data/raw") -> tuple[LabeledPrompt, ...]:
    """Load only the labeled training split."""

    rows = _read_rows("train", data_dir)
    return tuple(LabeledPrompt(row["id"], row["prompt"], row["label"]) for row in rows)


def load_dev(data_dir: str | Path = "data/raw") -> tuple[LabeledPrompt, ...]:
    """Load only the labeled development split for validation/model selection."""

    rows = _read_rows("dev", data_dir)
    return tuple(LabeledPrompt(row["id"], row["prompt"], row["label"]) for row in rows)


def load_predictions(
    split: PredictionSplit, data_dir: str | Path = "data/raw"
) -> tuple[PredictionPrompt, ...]:
    """Load dev-test or test records without exposing labels."""

    if split not in ("devtest", "test"):
        raise ValueError("Prediction data is limited to 'devtest' or 'test'.")
    rows = _read_rows(split, data_dir)
    return tuple(PredictionPrompt(row["id"], row["prompt"]) for row in rows)


def validate_data_directory(data_dir: str | Path = "data/raw") -> tuple[SplitSummary, ...]:
    """Validate every required split and an optional released test split.

    Checks exact public row counts, split-specific headers, non-empty fields,
    label values/balance, duplicate IDs within a split, and ID overlap across
    splits. Prompt contents are never included in errors or summaries.
    """

    root = Path(data_dir)
    missing = [name for name in REQUIRED_SPLITS if not (root / f"{name}.csv").is_file()]
    if missing:
        names = ", ".join(f"{name}.csv" for name in missing)
        raise DatasetValidationError(f"Missing required split file(s) in {root}: {names}")

    available: list[SplitName] = ["train", "dev", "devtest"]
    if (root / "test.csv").is_file():
        available.append("test")

    summaries: list[SplitSummary] = []
    ids_by_split: dict[SplitName, set[str]] = {}
    for split in available:
        if split == "train":
            records = load_train(root)
        elif split == "dev":
            records = load_dev(root)
        else:
            records = load_predictions(split, root)

        expected_rows = EXPECTED_ROWS[split]
        if len(records) != expected_rows:
            raise DatasetValidationError(
                f"{root / f'{split}.csv'}: expected {expected_rows} rows, found {len(records)}"
            )

        ids_by_split[split] = {record.id for record in records}
        if split in ("train", "dev"):
            safe = sum(record.label == "S" for record in records)
            unsafe = sum(record.label == "U" for record in records)
            expected_counts = EXPECTED_LABEL_COUNTS[split]
            if {"S": safe, "U": unsafe} != expected_counts:
                raise DatasetValidationError(
                    f"{root / f'{split}.csv'}: expected label counts "
                    f"{expected_counts}, found {{'S': {safe}, 'U': {unsafe}}}"
                )
            summaries.append(SplitSummary(split, len(records), safe, unsafe))
        else:
            summaries.append(SplitSummary(split, len(records)))

    for index, first in enumerate(available):
        for second in available[index + 1 :]:
            overlap = ids_by_split[first] & ids_by_split[second]
            if overlap:
                raise DatasetValidationError(
                    f"ID overlap between {first}.csv and {second}.csv: "
                    f"{len(overlap)} duplicate ID(s) across splits"
                )

    return tuple(summaries)
