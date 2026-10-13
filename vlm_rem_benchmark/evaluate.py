#!/usr/bin/env python3
"""Score blinded REM interval predictions using the frozen poster protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence


FRAME_SECONDS = 1800.0
MERGE_GAP_SECONDS = 5.0
MIN_DURATION_SECONDS = 30.0
IOU_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return payload
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_intervals(
    raw: Iterable[dict[str, Any]],
    issues: dict[str, dict[str, int]],
    source: str,
    issue_group: str = "prediction",
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, dict) or "start_sec" not in item or "end_sec" not in item:
            raise ValueError(f"Malformed interval in {source}: {item!r}")
        start = float(item["start_sec"])
        end = float(item["end_sec"])
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"Non-finite interval in {source}: {item!r}")
        clipped_start = max(0.0, min(FRAME_SECONDS, start))
        clipped_end = max(0.0, min(FRAME_SECONDS, end))
        if clipped_start != start or clipped_end != end:
            issues[issue_group]["clipped_intervals"] += 1
        if clipped_start >= clipped_end:
            issues[issue_group]["discarded_nonpositive_intervals"] += 1
            continue
        intervals.append((clipped_start, clipped_end))

    intervals.sort()
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + MERGE_GAP_SECONDS:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    output = []
    for start, end in merged:
        if end - start < MIN_DURATION_SECONDS:
            issues[issue_group]["discarded_short_after_merge"] += 1
        else:
            output.append((start, end))
    return output


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return safe_div(intersection, union)


def intersect_seconds(left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    i = j = 0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def ordered_one_to_one_matches(
    truth: Sequence[tuple[float, float]],
    prediction: Sequence[tuple[float, float]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Maximize match count, then total IoU, while preserving interval order."""
    rows = len(truth) + 1
    cols = len(prediction) + 1
    table: list[list[tuple[int, float, tuple[tuple[int, int, float], ...]]]] = [
        [(0, 0.0, ()) for _ in range(cols)] for _ in range(rows)
    ]

    def better(*candidates: tuple[int, float, tuple[tuple[int, int, float], ...]]):
        return max(candidates, key=lambda value: (value[0], value[1]))

    for i in range(1, rows):
        for j in range(1, cols):
            candidates = [table[i - 1][j], table[i][j - 1]]
            iou = interval_iou(truth[i - 1], prediction[j - 1])
            if iou >= threshold:
                prior = table[i - 1][j - 1]
                candidates.append((prior[0] + 1, prior[1] + iou, prior[2] + ((i - 1, j - 1, iou),)))
            table[i][j] = better(*candidates)
    return list(table[-1][-1][2])


def score_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    total_seconds = len(frames) * FRAME_SECONDS
    truth_seconds = sum(end - start for frame in frames for start, end in frame["truth"])
    prediction_seconds = sum(end - start for frame in frames for start, end in frame["prediction"])
    overlap_seconds = sum(intersect_seconds(frame["truth"], frame["prediction"]) for frame in frames)
    tp = overlap_seconds
    fp = prediction_seconds - overlap_seconds
    fn = truth_seconds - overlap_seconds
    tn = total_seconds - tp - fp - fn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    observed = safe_div(tp + tn, total_seconds)
    expected = safe_div(
        truth_seconds * prediction_seconds
        + (total_seconds - truth_seconds) * (total_seconds - prediction_seconds),
        total_seconds * total_seconds,
    )
    kappa = safe_div(observed - expected, 1 - expected)

    event_metrics: dict[str, Any] = {}
    anchor_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for threshold in IOU_THRESHOLDS:
        matches = []
        truth_count = prediction_count = 0
        for frame in frames:
            truth_count += len(frame["truth"])
            prediction_count += len(frame["prediction"])
            frame_matches = ordered_one_to_one_matches(frame["truth"], frame["prediction"], threshold)
            matches.extend(frame_matches)
            if threshold == 0.3:
                anchor_pairs.extend((frame["truth"][i], frame["prediction"][j]) for i, j, _ in frame_matches)
        matched = len(matches)
        event_precision = safe_div(matched, prediction_count)
        event_recall = safe_div(matched, truth_count)
        event_metrics[f"{threshold:.1f}"] = {
            "matches": matched,
            "reference_events": truth_count,
            "predicted_events": prediction_count,
            "precision": event_precision,
            "recall": event_recall,
            "f1": f1_score(event_precision, event_recall),
            "mean_matched_iou": fmean(x[2] for x in matches) if matches else None,
        }

    negative_frames = [frame for frame in frames if not frame["truth"]]
    false_alarm_frames = sum(bool(frame["prediction"]) for frame in negative_frames)
    overlapping_pieces = []
    missed_reference_events = 0
    for frame in frames:
        for truth_interval in frame["truth"]:
            pieces = sum(interval_iou(truth_interval, pred) > 0 for pred in frame["prediction"])
            if pieces:
                overlapping_pieces.append(pieces)
            else:
                missed_reference_events += 1

    return {
        "frames": len(frames),
        "postprocessed_reference_events": sum(len(frame["truth"]) for frame in frames),
        "postprocessed_predicted_events": sum(len(frame["prediction"]) for frame in frames),
        "per_second": {
            "tp_seconds": tp,
            "fp_seconds": fp,
            "fn_seconds": fn,
            "tn_seconds": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
            "cohens_kappa": kappa,
        },
        "events_by_iou": event_metrics,
        "boundary_error_at_iou_0.3": {
            "matched_pairs": len(anchor_pairs),
            "mean_absolute_onset_seconds": (
                fmean(abs(truth[0] - pred[0]) for truth, pred in anchor_pairs) if anchor_pairs else None
            ),
            "mean_absolute_offset_seconds": (
                fmean(abs(truth[1] - pred[1]) for truth, pred in anchor_pairs) if anchor_pairs else None
            ),
        },
        "rem_time": {
            "reference_seconds": truth_seconds,
            "predicted_seconds": prediction_seconds,
            "bias_seconds": prediction_seconds - truth_seconds,
            "relative_bias": safe_div(prediction_seconds - truth_seconds, truth_seconds),
        },
        "negative_frame_false_alarm": {
            "reference_negative_frames": len(negative_frames),
            "false_alarm_frames": false_alarm_frames,
            "rate": safe_div(false_alarm_frames, len(negative_frames)),
        },
        "fragmentation": {
            "detected_reference_events": len(overlapping_pieces),
            "missed_reference_events": missed_reference_events,
            "mean_overlapping_prediction_pieces": fmean(overlapping_pieces) if overlapping_pieces else None,
        },
    }


def prepare(
    vault_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]], stages: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    selected = [row for row in vault_rows if row.get("stage") in stages]
    if not selected:
        raise ValueError(f"No vault frames found for stage(s): {', '.join(stages)}")

    expected = {row["frame_id"]: row for row in selected}
    predictions: dict[str, dict[str, Any]] = {}
    for row in prediction_rows:
        frame_id = row.get("frame_id")
        if not isinstance(frame_id, str):
            raise ValueError("Every prediction must have a string frame_id")
        if frame_id in predictions:
            raise ValueError(f"Duplicate prediction frame_id: {frame_id}")
        predictions[frame_id] = row
    missing = sorted(set(expected) - set(predictions))
    extra = sorted(set(predictions) - set(expected))
    if missing or extra:
        raise ValueError(f"Frame ID mismatch: {len(missing)} missing, {len(extra)} unexpected")

    issues = {
        group: {
            "clipped_intervals": 0,
            "discarded_nonpositive_intervals": 0,
            "discarded_short_after_merge": 0,
        }
        for group in ("reference", "prediction")
    }
    issues["quality_exclusions"] = {stage: 0 for stage in stages}
    by_stage: dict[str, list[dict[str, Any]]] = {stage: [] for stage in stages}
    for frame_id, reference in expected.items():
        prediction = predictions[frame_id]
        episodes = prediction.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(f"episodes must be a list for {frame_id}")
        truth = normalize_intervals(
            reference["raw_episodes"], issues, f"reference {frame_id}", "reference"
        )
        predicted = normalize_intervals(
            episodes, issues, f"prediction {frame_id}", "prediction"
        )
        eligible = reference.get("render", {}).get("eligible_for_scoring", True)
        if not eligible:
            issues["quality_exclusions"][reference["stage"]] += 1
            continue
        by_stage[reference["stage"]].append(
            {"frame_id": frame_id, "truth": truth, "prediction": predicted}
        )
    return by_stage, issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--stage", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if len(args.stage) != len(set(args.stage)):
        raise ValueError("A stage was specified more than once")
    vault_rows = load_json_or_jsonl(args.vault)
    prediction_rows = load_json_or_jsonl(args.predictions)
    by_stage, issues = prepare(vault_rows, prediction_rows, args.stage)
    report = {
        "protocol": {
            "frame_seconds": FRAME_SECONDS,
            "merge_gap_seconds": MERGE_GAP_SECONDS,
            "minimum_duration_seconds": MIN_DURATION_SECONDS,
            "minimum_central_coverage_fraction": 0.9,
            "postprocess_order": ["clip", "sort", "merge", "duration_floor"],
            "iou_thresholds": IOU_THRESHOLDS,
        },
        "validation": issues,
        "by_stage": {stage: score_frames(frames) for stage, frames in by_stage.items()},
    }
    if len(args.stage) > 1:
        report["micro_aggregate"] = score_frames([frame for stage in args.stage for frame in by_stage[stage]])

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
