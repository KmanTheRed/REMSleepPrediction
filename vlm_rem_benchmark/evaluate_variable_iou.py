#!/usr/bin/env python3
"""Evaluate Codex REM events with a duration-dependent IoU gate.

The gate is defined per reference event of duration L as:
    T(L) = max(0, 1 - 20 / L)

This corresponds to a conservative allowance of 10 seconds at each boundary.
All loading, eligibility exclusions, clipping, merging, duration filtering, and
ordered one-to-one matching follow evaluate.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Sequence

import evaluate


ROOT = Path(r"D:\downloads\Kethan\Kethan\vlm_rem_benchmark")
VAULT = ROOT / "generated" / "private_vault" / "frames.jsonl"
COMBINED_PREDICTIONS = ROOT / "generated" / "codex_test_01_test_02_predictions.jsonl"
CONFIRMATORY_PREDICTIONS = Path(
    r"D:\vlm_clean_runs\chatgpt_sol_ultra_test_01_fresh\predictions.jsonl"
)
OUTPUT = Path(r"D:\downloads\Kethan\Kethan\codex_variable_iou_metrics.json")


def variable_threshold(reference: tuple[float, float]) -> float:
    duration = reference[1] - reference[0]
    return max(0.0, 1.0 - 20.0 / duration) if duration > 0 else 1.0


def ordered_variable_matches(
    truth: Sequence[tuple[float, float]],
    prediction: Sequence[tuple[float, float]],
) -> list[tuple[int, int, float, float]]:
    """Maximize count then IoU using the threshold attached to each truth."""
    rows = len(truth) + 1
    cols = len(prediction) + 1
    cell_type = tuple[int, float, tuple[tuple[int, int, float, float], ...]]
    table: list[list[cell_type]] = [
        [(0, 0.0, ()) for _ in range(cols)] for _ in range(rows)
    ]

    def better(*candidates: cell_type) -> cell_type:
        return max(candidates, key=lambda value: (value[0], value[1]))

    for i in range(1, rows):
        threshold = variable_threshold(truth[i - 1])
        for j in range(1, cols):
            candidates = [table[i - 1][j], table[i][j - 1]]
            iou = evaluate.interval_iou(truth[i - 1], prediction[j - 1])
            if iou >= threshold:
                prior = table[i - 1][j - 1]
                match = (i - 1, j - 1, iou, threshold)
                candidates.append((prior[0] + 1, prior[1] + iou, prior[2] + (match,)))
            table[i][j] = better(*candidates)
    return list(table[-1][-1][2])


def score(frames: list[dict]) -> dict:
    matches: list[tuple[float, float, float]] = []
    reference_count = 0
    prediction_count = 0
    thresholds: list[float] = []
    reference_durations: list[float] = []

    for frame in frames:
        reference_count += len(frame["truth"])
        prediction_count += len(frame["prediction"])
        for reference in frame["truth"]:
            reference_durations.append(reference[1] - reference[0])
            thresholds.append(variable_threshold(reference))
        for truth_index, pred_index, iou, threshold in ordered_variable_matches(
            frame["truth"], frame["prediction"]
        ):
            duration = frame["truth"][truth_index][1] - frame["truth"][truth_index][0]
            matches.append((iou, threshold, duration))

    matched = len(matches)
    precision = evaluate.safe_div(matched, prediction_count)
    recall = evaluate.safe_div(matched, reference_count)
    return {
        "matches": matched,
        "reference_events": reference_count,
        "predicted_events": prediction_count,
        "precision": precision,
        "recall": recall,
        "f1": evaluate.f1_score(precision, recall),
        "mean_matched_iou": fmean(item[0] for item in matches) if matches else None,
        "mean_required_iou": fmean(thresholds) if thresholds else None,
        "median_required_iou": sorted(thresholds)[len(thresholds) // 2] if thresholds else None,
        "reference_duration_range_seconds": [
            min(reference_durations) if reference_durations else None,
            max(reference_durations) if reference_durations else None,
        ],
    }


def load_frames(predictions: Path, stages: list[str]) -> dict[str, list[dict]]:
    vault_rows = evaluate.load_json_or_jsonl(VAULT)
    prediction_rows = evaluate.load_json_or_jsonl(predictions)
    by_stage, _issues = evaluate.prepare(vault_rows, prediction_rows, stages)
    return by_stage


def main() -> None:
    combined = load_frames(COMBINED_PREDICTIONS, ["test_01", "test_02"])
    confirmatory = load_frames(CONFIRMATORY_PREDICTIONS, ["confirmatory"])

    combined_frames = combined["test_01"] + combined["test_02"]
    report = {
        "gate": {
            "formula": "max(0, 1 - 20 / reference_duration_seconds)",
            "boundary_ambiguity_seconds_per_edge": 10,
        },
        "test_01": score(combined["test_01"]),
        "test_02": score(combined["test_02"]),
        "combined": score(combined_frames),
        "confirmatory": score(confirmatory["confirmatory"]),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    OUTPUT.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
