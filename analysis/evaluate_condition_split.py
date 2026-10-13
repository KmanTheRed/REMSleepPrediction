#!/usr/bin/env python3
"""Evaluate REM detectors separately before and during caffeine exposure.

Condition boundaries come from the first strictly positive value in each
recording's native ``caffeine`` channel.  The scoring protocol is otherwise
identical to ``evaluate_six_animal_adaptive_iou.py``: half-open intervals,
common TCN coverage, merge gaps <=5 s, discard episodes <30 s, thresholds
0.80/0.50, and greedy one-to-one event matching.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import evaluate_six_animal_adaptive_iou as base


ROOT = base.ROOT
SIGNALS = {
    "A1": ROOT / "newData/COMBINED_CAFFEINE_A1_2022-04-08_to_2022-04-15.pq",
    "A3": ROOT / "newData/COMBINED_CAFFEINE_A3_2022-07-20_to_2022-07-27.pq",
    "A5": ROOT / "newData/animal5_20221003B_trimmed.pq",
    "A6": ROOT / "newData/COMBINED_CAFFEINE_A6_2023-06-05_to_2023-06-08.pq",
    "A8": ROOT / "newData/COMBINED_CAFFEINE_A8_2023-06-08_to_2023-06-22.pq",
    "A11": ROOT / "newData/animal11_20230616B_trimmed.pq",
}


def first_caffeine_time(path: Path) -> float | None:
    """Return the first native timestamp with caffeine > 0, if one exists."""
    frame = pd.read_parquet(path, columns=["caffeine", "t_idx"])
    caffeine = pd.to_numeric(frame["caffeine"], errors="coerce").fillna(0.0)
    positive = caffeine.gt(0.0).to_numpy().nonzero()[0]
    if len(positive) == 0:
        return None
    row = int(positive[0])
    timestamp_value = frame.iloc[row]["t_idx"] if "t_idx" in frame.columns else frame.index[row]
    timestamp = pd.Timestamp(timestamp_value).to_pydatetime()
    return base.dt_seconds(timestamp.isoformat(sep=" "))


def duration(intervals: list[base.Interval]) -> float:
    return sum(end - start for start, end in intervals)


def intersection_duration(a: list[base.Interval], b: list[base.Interval]) -> float:
    total = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def second_metrics(
    reference: list[base.Interval], predicted: list[base.Interval], span: float
) -> dict[str, float]:
    tp = intersection_duration(reference, predicted)
    ref_seconds = duration(reference)
    pred_seconds = duration(predicted)
    fp = max(0.0, pred_seconds - tp)
    fn = max(0.0, ref_seconds - tp)
    tn = max(0.0, span - tp - fp - fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    observed = (tp + tn) / span if span else 0.0
    ref_pos = (tp + fn) / span if span else 0.0
    pred_pos = (tp + fp) / span if span else 0.0
    expected = ref_pos * pred_pos + (1.0 - ref_pos) * (1.0 - pred_pos)
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 0.0
    rem_bias = (pred_seconds - ref_seconds) / ref_seconds if ref_seconds else 0.0
    return {
        "tp_seconds": tp,
        "fp_seconds": fp,
        "fn_seconds": fn,
        "tn_seconds": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
        "reference_rem_seconds": ref_seconds,
        "predicted_rem_seconds": pred_seconds,
        "rem_time_bias": rem_bias,
        "absolute_rem_time_error": abs(rem_bias),
    }


def iso(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return (datetime(1970, 1, 1) + timedelta(seconds=seconds)).isoformat(sep=" ")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(
    animal_results: dict[str, dict], animal_names: tuple[str, ...], condition: str
) -> dict[str, dict]:
    eligible = [name for name in animal_names if condition in animal_results.get(name, {})]
    output: dict[str, dict] = {"animals": eligible, "n_animals": len(eligible), "models": {}}
    for model in base.MODELS:
        rows = [animal_results[name][condition][model] for name in eligible]
        output["models"][model] = {
            "per_second_precision": mean([row["per_second"]["precision"] for row in rows]),
            "per_second_recall": mean([row["per_second"]["recall"] for row in rows]),
            "per_second_f1": mean([row["per_second"]["f1"] for row in rows]),
            "kappa": mean([row["per_second"]["kappa"] for row in rows]),
            "event_f1_iou_0.5": mean([row["fixed_0.5"]["f1"] for row in rows]),
            "event_f1_iou_0.7": mean([row["fixed_0.7"]["f1"] for row in rows]),
            "adaptive_event_f1": mean([row["adaptive"]["f1"] for row in rows]),
            "volume_mad": mean([
                row["per_second"]["absolute_rem_time_error"] for row in rows
            ]),
        }
    return output


def main() -> None:
    boundaries = {animal.name: first_caffeine_time(SIGNALS[animal.name]) for animal in base.ANIMALS}
    results: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}

    for animal in base.ANIMALS:
        classical_raw, classical_lo, classical_hi = base.load_probability_runs(
            animal.classical_csv, 0.80
        )
        expert_raw, expert_lo, expert_hi = base.load_probability_runs(animal.expert_csv, 0.50)
        coverage_lo = max(classical_lo, expert_lo)
        coverage_hi = min(classical_hi, expert_hi)
        if coverage_hi <= coverage_lo:
            raise ValueError(f"No common model coverage for {animal.name}")

        reference_raw = base.load_elapsed_intervals(animal.reference_txt, animal.reference_base)
        rule_raw = base.load_elapsed_intervals(animal.rule_txt, animal.reference_base)
        caffeine_start = boundaries[animal.name]
        windows: dict[str, tuple[float, float]] = {}
        noncaffeine_hi = min(coverage_hi, caffeine_start) if caffeine_start else coverage_hi
        if noncaffeine_hi > coverage_lo:
            windows["non_caffeine"] = (coverage_lo, noncaffeine_hi)
        if caffeine_start is not None:
            caffeine_lo = max(coverage_lo, caffeine_start)
            if coverage_hi > caffeine_lo:
                windows["caffeine"] = (caffeine_lo, coverage_hi)

        diagnostics[animal.name] = {
            "source_signal": str(SIGNALS[animal.name]),
            "caffeine_start": iso(caffeine_start),
            "coverage_start": iso(coverage_lo),
            "coverage_end_exclusive": iso(coverage_hi),
            "conditions": {},
        }
        results[animal.name] = {}

        for condition, (lo, hi) in windows.items():
            reference = base.merge_floor(base.clip(reference_raw, lo, hi))
            predictions = {
                "Hand-tuned rules": base.merge_floor(base.clip(rule_raw, lo, hi)),
                "TCN, rule labels": base.merge_floor(base.clip(classical_raw, lo, hi)),
                "TCN, expert labels": base.merge_floor(base.clip(expert_raw, lo, hi)),
            }
            diagnostics[animal.name]["conditions"][condition] = {
                "start": iso(lo),
                "end_exclusive": iso(hi),
                "duration_seconds": hi - lo,
                "reference_events": len(reference),
                **{f"{model}_events": len(predicted) for model, predicted in predictions.items()},
            }
            results[animal.name][condition] = {}
            for model, predicted in predictions.items():
                results[animal.name][condition][model] = {
                    "per_second": second_metrics(reference, predicted, hi - lo),
                    "fixed_0.5": base.evaluate(reference, predicted, 0.5),
                    "fixed_0.7": base.evaluate(reference, predicted, 0.7),
                    "adaptive": base.evaluate(reference, predicted, "adaptive"),
                }

    all_animals = tuple(animal.name for animal in base.ANIMALS)
    aggregates = {
        "all_six": {
            condition: aggregate(results, all_animals, condition)
            for condition in ("non_caffeine", "caffeine")
        },
        "held_out_four": {
            condition: aggregate(results, base.HELD_OUT, condition)
            for condition in ("non_caffeine", "caffeine")
        },
    }
    payload = {
        "protocol": {
            "condition_boundary": "first native caffeine value > 0",
            "condition_window_rule": "clip raw intervals to condition, then merge/floor",
            "interval_convention": "half-open [start, end)",
            "postprocessing": "merge gaps <= 5 s, then discard duration < 30 s",
            "applied_to": "reference and all prediction streams",
            "tcn_rule_label_threshold": 0.80,
            "tcn_expert_label_threshold": 0.50,
            "matching": "greedy one-to-one matching, descending IoU",
            "adaptive_gate": "IoU >= max(0, 1 - 20/L)",
            "aggregation": "unweighted macro-average over eligible animals",
        },
        "diagnostics": diagnostics,
        "animals": results,
        "aggregates": aggregates,
    }

    json_path = ROOT / "condition_split_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = ROOT / "condition_split_metrics.csv"
    fields = [
        "cohort", "condition", "n_animals", "animals", "model",
        "per_second_precision", "per_second_recall", "per_second_f1", "kappa",
        "event_f1_iou_0.5", "event_f1_iou_0.7", "adaptive_event_f1", "volume_mad",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cohort, condition_rows in aggregates.items():
            for condition, block in condition_rows.items():
                for model, metrics in block["models"].items():
                    writer.writerow({
                        "cohort": cohort,
                        "condition": condition,
                        "n_animals": block["n_animals"],
                        "animals": ",".join(block["animals"]),
                        "model": model,
                        **metrics,
                    })

    print(json.dumps({"boundaries": {k: iso(v) for k, v in boundaries.items()}, "aggregates": aggregates}, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
