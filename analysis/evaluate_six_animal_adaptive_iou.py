#!/usr/bin/env python3
"""Evaluate REM episodes with the requested six-animal adaptive-IoU protocol.

Every reference and prediction stream is processed identically: merge gaps of
at most 5 seconds, then discard episodes shorter than 30 seconds. The
rule-label TCN is thresholded at 0.80 and the expert/manual-label TCN at 0.50.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(r"D:\downloads\Kethan\Kethan")
Interval = tuple[float, float]


@dataclass(frozen=True)
class Animal:
    name: str
    reference_txt: Path
    reference_base: str
    rule_txt: Path
    classical_csv: Path
    expert_csv: Path


ANIMALS = [
    Animal(
        "A1",
        ROOT / "newData/ManualScores/COMBINED_CAFFEINE/manualscore1.txt",
        "2022-04-08 00:00:00",
        ROOT / "newData/MLModelScores/COMBINED_CAFFEINE_A1_2022-04-08_to_2022-04-15/classicalModel/classicalModel_RAW_ranges_thr0.500A1.txt",
        ROOT / "rem_tcn_event_v9_classicalscores_stride120_fresh/trial_0000/test_predictions_COMBINED_CAFFEINE_t0000.csv",
        ROOT / "rem_tcn_event_v9_manualscores_stride120_fresh/trial_0064/test_predictions_COMBINED_CAFFEINE064.csv",
    ),
    Animal(
        "A3",
        ROOT / "newData/ManualScores/COMBINED_CAFFEINE3/manualscore 3.txt",
        "2022-07-20 00:00:00",
        ROOT / "newData/MLModelScores/COMBINED_CAFFEINE_A3_2022-07-20_to_2022-07-27/classicalModel/classicalModel_RAW_ranges_thr0.500A3.txt",
        ROOT / "rem_tcn_event_v9_classicalscores_stride120_fresh/trial_0000/A3_pred_t0000.csv",
        ROOT / "rem_tcn_event_v9_manualscores_stride120_fresh/trial_0064/A3_pred0064.csv",
    ),
    Animal(
        "A5",
        ROOT / "newData/ManualScores/COMBINED_CAFFEINE5/manualscore5.txt",
        # A5 starts before midnight; elapsed-time zero is the next date.
        "2022-10-04 00:00:00",
        ROOT / "newData/MLModelScores/animal5_20221003B_trimmed/classicalModel/classicalModel_RAW_ranges_thr0.500A5.txt",
        ROOT / "rem_tcn_event_v9_classicalscores_stride120_fresh/trial_0000/A5_pred_t0000.csv",
        ROOT / "rem_tcn_event_v9_manualscores_stride120_fresh/trial_0064/A5_pred0064.csv",
    ),
    Animal(
        "A6",
        ROOT / "newData/ManualScores/COMBINED_CAFFEINE6/manualscore6.txt",
        "2023-06-05 00:00:00",
        ROOT / "newData/MLModelScores/COMBINED_CAFFEINE_A6_2023-06-05_to_2023-06-08/classicalModel/classicalModel_RAW_ranges_thr0.500A6.txt",
        ROOT / "rem_tcn_event_v9_classicalscores_stride120_fresh/trial_0000/A6_pred_t0000.csv",
        ROOT / "rem_tcn_event_v9_manualscores_stride120_fresh/trial_0064/A6_pred0064.csv",
    ),
    Animal(
        "A8",
        ROOT / "newData/ManualScores/COMBINED_CAFFEINE8/wowCombinedCaffeine8.txt",
        "2023-06-08 00:00:00",
        ROOT / "newData/MLModelScores/COMBINED_CAFFEINE_A8_2023-06-08_to_2023-06-22/classicalModel/classicalModel_RAW_ranges_thr0.500A8.txt",
        ROOT / "rem_tcn_event_v9_classicalscores_stride120_fresh/trial_0000/A8_pred_t0000.csv",
        ROOT / "rem_tcn_event_v9_manualscores_stride120_fresh/trial_0064/A8_pred0064.csv",
    ),
    Animal(
        "A11",
        ROOT / "newData/ManualScores/COMBINED_CAFFEINE11/A11ManualScore.txt",
        "2023-06-20 00:00:00",
        ROOT / "newData/MLModelScores/20230616B/classicalModel/classicalModel_RAW_ranges_thr0.500A11.txt",
        ROOT / "rem_tcn_event_v9_classicalscores_stride120_fresh/trial_0000/A11_pred_t0000.csv",
        ROOT / "rem_tcn_event_v9_manualscores_stride120_fresh/trial_0064/A11_pred0064.csv",
    ),
]

MODELS = ("Hand-tuned rules", "TCN, rule labels", "TCN, expert labels")
HELD_OUT = ("A5", "A6", "A8", "A11")
MERGE_GAP_S = 5.0
MIN_DURATION_S = 30.0


def dt_seconds(value: str) -> float:
    """Convert a timezone-naive acquisition timestamp to epoch-like seconds."""
    return (datetime.fromisoformat(value) - datetime(1970, 1, 1)).total_seconds()


def clock_seconds(token: str) -> int:
    """Parse elapsed HH:MM:SS; HH may exceed 23."""
    hours, minutes, seconds = (int(part) for part in token.strip().split(":"))
    return hours * 3600 + minutes * 60 + seconds


def clip(intervals: list[Interval], lo: float, hi: float) -> list[Interval]:
    return [
        (max(start, lo), min(end, hi))
        for start, end in intervals
        if end > lo and start < hi and min(end, hi) > max(start, lo)
    ]


def merge_floor(
    intervals: list[Interval], gap: float = MERGE_GAP_S, floor: float = MIN_DURATION_S
) -> list[Interval]:
    """Merge first, then apply the duration floor, using half-open intervals."""
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged if end - start >= floor]


def load_elapsed_intervals(path: Path, base: str) -> list[Interval]:
    base_s = dt_seconds(base)
    intervals: list[Interval] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            parts = [part.strip() for part in line.strip().split(",")]
            if len(parts) != 2:
                continue
            try:
                start = base_s + clock_seconds(parts[0])
                end = base_s + clock_seconds(parts[1])
            except (ValueError, IndexError):
                continue
            if end > start:
                intervals.append((start, end))
    return intervals


def load_probability_runs(path: Path, threshold: float) -> tuple[list[Interval], float, float]:
    """Load thresholded runs while respecting real timestamp discontinuities."""
    raw: list[Interval] = []
    active_start: float | None = None
    previous: float | None = None
    first: float | None = None
    last: float | None = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            timestamp = dt_seconds(row["timestamp"])
            is_active = float(row["pred_prob_rem"]) >= threshold
            if first is None:
                first = timestamp

            # A missing second ends any active run; never bridge data gaps.
            if active_start is not None and previous is not None and timestamp - previous != 1.0:
                raw.append((active_start, previous + 1.0))
                active_start = None

            if is_active and active_start is None:
                active_start = timestamp
            elif not is_active and active_start is not None:
                raw.append((active_start, timestamp))
                active_start = None

            previous = timestamp
            last = timestamp

    if active_start is not None and last is not None:
        raw.append((active_start, last + 1.0))
    if first is None or last is None:
        raise ValueError(f"No predictions found in {path}")
    return raw, first, last + 1.0


def iou(a: Interval, b: Interval) -> float:
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if intersection <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    return intersection / union


def adaptive_gate(reference: Interval) -> float:
    """IoU gate allowing 10 seconds of ambiguity at each reference edge."""
    duration = reference[1] - reference[0]
    return max(0.0, 1.0 - 20.0 / duration) if duration > 0 else 1.0


def evaluate(reference: list[Interval], predicted: list[Interval], gate: float | str) -> dict:
    """Greedy one-to-one event matching in descending IoU order."""
    reference = sorted(reference)
    predicted = sorted(predicted)
    candidates: list[tuple[float, int, int]] = []
    scan_start = 0
    for ref_i, ref in enumerate(reference):
        threshold = adaptive_gate(ref) if gate == "adaptive" else float(gate)
        while scan_start < len(predicted) and predicted[scan_start][1] <= ref[0]:
            scan_start += 1
        pred_i = scan_start
        while pred_i < len(predicted) and predicted[pred_i][0] < ref[1]:
            score = iou(ref, predicted[pred_i])
            if score >= threshold:
                candidates.append((score, ref_i, pred_i))
            pred_i += 1

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_ref: set[int] = set()
    used_pred: set[int] = set()
    matched_ious: list[float] = []
    for score, ref_i, pred_i in candidates:
        if ref_i not in used_ref and pred_i not in used_pred:
            used_ref.add(ref_i)
            used_pred.add(pred_i)
            matched_ious.append(score)

    tp = len(used_ref)
    fp = len(predicted) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
    }


def macro(results: dict, names: tuple[str, ...]) -> dict:
    output: dict[str, dict[str, float]] = {}
    for model in MODELS:
        output[model] = {}
        for gate in ("fixed_0.5", "fixed_0.7", "adaptive"):
            for metric in ("precision", "recall", "f1"):
                values = [float(results[name][model][gate][metric]) for name in names]
                output[model][f"{gate}_{metric}"] = sum(values) / len(values)
    return output


def pooled(results: dict, names: tuple[str, ...]) -> dict:
    output: dict[str, dict[str, dict]] = {}
    for model in MODELS:
        output[model] = {}
        for gate in ("fixed_0.5", "fixed_0.7", "adaptive"):
            tp = sum(int(results[name][model][gate]["tp"]) for name in names)
            fp = sum(int(results[name][model][gate]["fp"]) for name in names)
            fn = sum(int(results[name][model][gate]["fn"]) for name in names)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            output[model][gate] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
            }
    return output


def main() -> None:
    missing = [str(path) for animal in ANIMALS for path in (
        animal.reference_txt, animal.rule_txt, animal.classical_csv, animal.expert_csv
    ) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    results: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}
    for animal in ANIMALS:
        classical_raw, classical_lo, classical_hi = load_probability_runs(animal.classical_csv, 0.80)
        expert_raw, expert_lo, expert_hi = load_probability_runs(animal.expert_csv, 0.50)
        lo, hi = max(classical_lo, expert_lo), min(classical_hi, expert_hi)
        if hi <= lo:
            raise ValueError(f"No common model coverage for {animal.name}")

        reference = merge_floor(clip(load_elapsed_intervals(
            animal.reference_txt, animal.reference_base
        ), lo, hi))
        predictions = {
            "Hand-tuned rules": merge_floor(clip(load_elapsed_intervals(
                animal.rule_txt, animal.reference_base
            ), lo, hi)),
            "TCN, rule labels": merge_floor(clip(classical_raw, lo, hi)),
            "TCN, expert labels": merge_floor(clip(expert_raw, lo, hi)),
        }

        diagnostics[animal.name] = {
            "coverage_start": (datetime(1970, 1, 1) + timedelta(seconds=lo)).isoformat(sep=" "),
            "coverage_end_exclusive": (datetime(1970, 1, 1) + timedelta(seconds=hi)).isoformat(sep=" "),
            "reference_events": len(reference),
            **{f"{model}_events": len(events) for model, events in predictions.items()},
        }
        results[animal.name] = {}
        for model, predicted in predictions.items():
            results[animal.name][model] = {
                "fixed_0.5": evaluate(reference, predicted, 0.5),
                "fixed_0.7": evaluate(reference, predicted, 0.7),
                "adaptive": evaluate(reference, predicted, "adaptive"),
            }

    all_names = tuple(animal.name for animal in ANIMALS)
    payload = {
        "protocol": {
            "interval_convention": "half-open [start, end)",
            "postprocessing_order": "merge gaps <= 5 s, then discard duration < 30 s",
            "applied_to": "reference and all three prediction streams",
            "tcn_rule_label_threshold": 0.80,
            "tcn_expert_label_threshold": 0.50,
            "adaptive_gate": "IoU >= max(0, 1 - 20/L), L = reference duration in seconds",
            "adaptive_interpretation": "10 seconds of boundary ambiguity at each edge",
            "matching": "greedy one-to-one matching, descending IoU",
        },
        "sources": {
            animal.name: {
                "reference": str(animal.reference_txt),
                "reference_base": animal.reference_base,
                "hand_tuned_rules": str(animal.rule_txt),
                "tcn_rule_labels": str(animal.classical_csv),
                "tcn_expert_labels": str(animal.expert_csv),
            } for animal in ANIMALS
        },
        "diagnostics": diagnostics,
        "animals": results,
        "macro_all_6": macro(results, all_names),
        "macro_held_out_4": macro(results, HELD_OUT),
        "pooled_all_6": pooled(results, all_names),
        "pooled_held_out_4": pooled(results, HELD_OUT),
    }

    json_path = ROOT / "six_animal_adaptive_iou_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = ROOT / "six_animal_adaptive_iou_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["cohort", "animal", "model", "gate", "precision", "recall", "f1", "tp", "fp", "fn"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for animal in ANIMALS:
            for model in MODELS:
                for gate in ("fixed_0.5", "fixed_0.7", "adaptive"):
                    writer.writerow({
                        "cohort": "animal", "animal": animal.name,
                        "model": model, "gate": gate,
                        **{key: results[animal.name][model][gate][key]
                           for key in ("precision", "recall", "f1", "tp", "fp", "fn")},
                    })
        for cohort_name, names in (("macro_all_6", all_names), ("macro_held_out_4", HELD_OUT)):
            aggregate = macro(results, names)
            for model in MODELS:
                for gate in ("fixed_0.5", "fixed_0.7", "adaptive"):
                    writer.writerow({
                        "cohort": cohort_name, "animal": "", "model": model, "gate": gate,
                        "precision": aggregate[model][f"{gate}_precision"],
                        "recall": aggregate[model][f"{gate}_recall"],
                        "f1": aggregate[model][f"{gate}_f1"],
                        "tp": "", "fp": "", "fn": "",
                    })

    for cohort in ("macro_all_6", "macro_held_out_4"):
        print(cohort)
        for model in MODELS:
            row = payload[cohort][model]
            print(
                f"  {model}: fixed@.5={row['fixed_0.5_f1']:.6f}, "
                f"fixed@.7={row['fixed_0.7_f1']:.6f}, adaptive={row['adaptive_f1']:.6f}"
            )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
