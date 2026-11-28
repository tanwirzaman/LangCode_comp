"""Validate the released Task 2 CSV splits without printing prompt contents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from langcode_task2.data import DatasetValidationError, validate_data_directory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "raw",
        help="directory containing train.csv, dev.csv, devtest.csv, and optional test.csv",
    )
    args = parser.parse_args()

    try:
        summaries = validate_data_directory(args.data_dir)
    except DatasetValidationError as exc:
        print(f"Data validation failed: {exc}", file=sys.stderr)
        return 1

    for summary in summaries:
        if summary.split in ("train", "dev"):
            print(
                f"{summary.split}: {summary.rows:,} rows, "
                f"S={summary.safe:,}, U={summary.unsafe:,}"
            )
        else:
            print(f"{summary.split}: {summary.rows:,} unlabeled prompts")

    if not any(summary.split == "test" for summary in summaries):
        print("test: not present (released later by the organizers)")
    print("OK: schemas, contents, row counts, labels, and split IDs are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
