"""Compare dev probability files with the official scorer and script slices.

Example:
  python scripts/analyze_experiments.py \
    --experiment bert-base=outputs/bert-base-baseline/dev_probabilities.csv \
    --experiment mbert=outputs/mbert/dev_probabilities.csv

The script reads dev only. It does not read dev-test or test. Script groups
are Unicode-script proxies; it never attempts to infer language or prompt form.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "Bengali": ((0x0980, 0x09FF),),
    "Han": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0x20000, 0x2EBEF)),
    "Hebrew": ((0x0590, 0x05FF), (0xFB1D, 0xFB4F)),
    "Devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "Hiragana/Katakana": ((0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF65, 0xFF9F)),
    "Hangul": ((0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7AF), (0xD7B0, 0xD7FF)),
    "Cyrillic": ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
    "Tamil": ((0x0B80, 0x0BFF),),
    "Thai": ((0x0E00, 0x0E7F),),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="name and dev_probabilities.csv path; repeat for each candidate",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "robustness-analysis",
    )
    parser.add_argument("--low", type=float, default=0.20)
    parser.add_argument("--high", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()
    if args.step <= 0 or args.low < 0 or args.high > 1 or args.low >= args.high:
        parser.error("thresholds must satisfy 0 <= low < high <= 1 and step > 0")
    return args


def load_official_scorer(path: Path):
    spec = importlib.util.spec_from_file_location("langcode_official_scorer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official scorer at {path}")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    return scorer


def read_probability_file(path: Path, gold: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": str, "label": str})
    required = {"id", "label", "p_unsafe"}
    if set(frame.columns) != required:
        raise ValueError(f"{path}: columns must be exactly {sorted(required)}")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate IDs")
    frame["p_unsafe"] = pd.to_numeric(frame["p_unsafe"], errors="raise")
    if not frame["p_unsafe"].between(0, 1).all():
        raise ValueError(f"{path}: p_unsafe values must be in [0, 1]")
    by_id = frame.set_index("id")
    if set(by_id.index) != set(gold):
        raise ValueError(f"{path}: IDs do not exactly match dev.csv")
    if any(by_id.at[pid, "label"] != label for pid, label in gold.items()):
        raise ValueError(f"{path}: stored labels do not match dev.csv")
    return by_id


def metric_row(scorer, probs: pd.Series, gold: dict[str, str], threshold: float) -> dict:
    predictions = {
        prompt_id: ("U" if probability >= threshold else "S")
        for prompt_id, probability in probs.items()
    }
    scores = scorer.safety_scores(predictions, gold)
    return {
        "threshold": round(float(threshold), 4),
        "safety_score": scores["score"],
        "unsafe_recall": scores["unsafe_recall"],
        "safe_recall": scores["safe_recall"],
    }


def script_group(prompt: str) -> str:
    counts: Counter[str] = Counter()
    for character in prompt:
        codepoint = ord(character)
        for name, ranges in SCRIPT_RANGES.items():
            if any(start <= codepoint <= end for start, end in ranges):
                counts[name] += 1
                break
    if not counts:
        return "No target non-Latin script"
    if len(counts) > 1:
        largest = max(counts.values())
        leaders = [name for name, count in counts.items() if count == largest]
        if len(leaders) > 1:
            return "Multiple target scripts"
    else:
        leaders = [next(iter(counts))]
    selected = leaders[0]
    if selected == "Han":
        return "Chinese/Han"
    if selected == "Hiragana/Katakana":
        return "Japanese kana"
    if selected == "Hangul":
        return "Korean Hangul"
    return selected


def main() -> int:
    args = parse_args()
    scorer = load_official_scorer(args.data_dir / "scorer" / "score.py")
    gold = scorer.read_labeled(str(args.data_dir / "dev.csv"))
    dev = pd.read_csv(args.data_dir / "dev.csv", dtype=str, keep_default_na=False)
    groups = {row.id: script_group(row.prompt) for row in dev.itertuples(index=False)}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.round(
        np.arange(args.low, args.high + args.step / 2, args.step), 6
    )
    sweep_rows: list[dict] = []
    summary_rows: list[dict] = []
    loaded: dict[str, pd.DataFrame] = {}
    for experiment in args.experiment:
        if "=" not in experiment:
            raise SystemExit(f"Expected NAME=CSV for --experiment, got {experiment!r}")
        name, raw_path = experiment.split("=", 1)
        if not name or name in loaded:
            raise SystemExit(f"Experiment names must be non-empty and unique: {name!r}")
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        probabilities = read_probability_file(path, gold)
        loaded[name] = probabilities
        per_threshold = [
            metric_row(scorer, probabilities["p_unsafe"], gold, threshold)
            for threshold in thresholds
        ]
        sweep_rows.extend({"experiment": name, **row} for row in per_threshold)
        # Resolve ties by preferring the threshold closest to 0.5.
        best = max(
            per_threshold,
            key=lambda row: (
                row["safety_score"],
                -abs(row["threshold"] - 0.5),
                min(row["unsafe_recall"], row["safe_recall"]),
            ),
        )
        at_default = min(per_threshold, key=lambda row: abs(row["threshold"] - 0.5))
        summary_rows.append(
            {
                "experiment": name,
                "score_at_0_5": at_default["safety_score"],
                "R_U_at_0_5": at_default["unsafe_recall"],
                "R_S_at_0_5": at_default["safe_recall"],
                "best_threshold_on_dev": best["threshold"],
                "best_safety_score_on_dev": best["safety_score"],
                "best_unsafe_recall_on_dev": best["unsafe_recall"],
                "best_safe_recall_on_dev": best["safe_recall"],
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["best_safety_score_on_dev", "score_at_0_5"], ascending=False
    )
    sweep = pd.DataFrame(sweep_rows)
    summary.to_csv(args.output_dir / "experiment_summary.csv", index=False)
    sweep.to_csv(args.output_dir / "threshold_sweep.csv", index=False)

    selected = summary.iloc[0]
    selected_name = selected["experiment"]
    selected_threshold = float(selected["best_threshold_on_dev"])
    selected_probs = loaded[selected_name]["p_unsafe"]
    selected_preds = {
        prompt_id: ("U" if probability >= selected_threshold else "S")
        for prompt_id, probability in selected_probs.items()
    }
    pd.DataFrame(
        {"id": list(selected_preds), "label": list(selected_preds.values())}
    ).to_csv(args.output_dir / "selected_dev_predictions.csv", index=False)
    script_rows: list[dict] = []
    for group in sorted(set(groups.values())):
        group_ids = [prompt_id for prompt_id, value in groups.items() if value == group]
        group_gold = {prompt_id: gold[prompt_id] for prompt_id in group_ids}
        group_preds = {prompt_id: selected_preds[prompt_id] for prompt_id in group_ids}
        scores = scorer.safety_scores(group_preds, group_gold)
        false_positive = sum(
            label == "S" and group_preds[prompt_id] == "U"
            for prompt_id, label in group_gold.items()
        )
        false_negative = sum(
            label == "U" and group_preds[prompt_id] == "S"
            for prompt_id, label in group_gold.items()
        )
        script_rows.append(
            {
                "script_proxy": group,
                "n": len(group_ids),
                "safe_n": sum(label == "S" for label in group_gold.values()),
                "unsafe_n": sum(label == "U" for label in group_gold.values()),
                "safety_score": scores["score"],
                "unsafe_recall": scores["unsafe_recall"],
                "safe_recall": scores["safe_recall"],
                "false_unsafe": false_positive,
                "missed_unsafe": false_negative,
            }
        )
    pd.DataFrame(script_rows).to_csv(args.output_dir / "selected_script_slices.csv", index=False)

    payload = {
        "selection_criterion": "maximum dev Safety Score over the configured threshold grid; ties prefer a threshold closest to 0.5, then better minimum class recall",
        "threshold_grid": {"low": args.low, "high": args.high, "step": args.step},
        "selected_experiment": selected_name,
        "selected_threshold": selected_threshold,
        "dev_only": True,
        "form_analysis_note": "The official data has no language or code-mixed/transliterated-form annotations. Unicode script groups are descriptive script proxies; form-specific metrics are not reported.",
        "script_group_definition": "Dominant Unicode script among the ten task target script families; Han without kana is treated as Chinese/Han, Japanese kana as Japanese. No prompt text is written to these outputs.",
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nSelected on dev: {selected_name} at p(U) >= {selected_threshold:.2f}")
    print("Script slices for selected candidate:")
    print(pd.DataFrame(script_rows).to_string(index=False))
    print(f"\nAnalysis files: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
