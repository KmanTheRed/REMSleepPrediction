#!/usr/bin/env python3
"""Recalculate continuous-recording REM event metrics with an adaptive IoU gate.

This script uses the four frozen held-out recordings and prediction streams used
in the poster.  It also reports fixed IoU 0.5/0.7 values so the reconstruction
can be checked against the poster before the adaptive result is used.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

SITE_PACKAGES = Path(r"D:\downloads\Kethan\Kethan\.venv\Lib\site-packages")
sys.path.insert(0, str(SITE_PACKAGES))

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(r"D:\downloads\Kethan\Kethan")


@dataclass(frozen=True)
class Animal:
    name: str
    parquet: Path
    reference_txt: Path
    reference_base: str
    rule_csv: Path
    classical_csv: Path
    expert_csv: Path


ANIMALS = [
    Animal(
        "A6",
        ROOT / "newData/COMBINED_CAFFEINE_A6_2023-06-05_to_2023-06-08.pq",
        ROOT / "newData/ManualScores/COMBINED_CAFFEINE6/manualscore6.txt",
        "2023-06-05 00:00:00",
        ROOT / "newData/MLModelScores/COMBINED_CAFFEINE_A6_2023-06-05_to_2023-06-08/classicalModel/classicalModel_RAW_ranges_thr0.500.csv",
        ROOT / "rem_tcn_event_v9_classicalscores_stride120_fresh/trial_0000/A6_pred_t0000.csv",
        ROOT / "rem_tcn_event_v9_manualscores_stride120_fresh/trial_0064/A6_pred0064.csv",
    ),
    Animal(
        "A8",
        ROOT / "newData/COMBINED_CAFFEINE_A8_2023-06-08_to_2023-06-22.pq",
        ROOT / "newData/ManualScores/COMBINED_CAFFEINE8/wow.txt",
        "2023-06-08 00:00:00",
        ROOT / "newData/MLModelScores/COMBINED_CAFFEINE_A8_2023-06-08_to_2023-06-22/classicalModel/classicalModel_RAW_ranges_thr0.500.csv",
        ROOT / "rem_tcn_event_v9_classicalscores_stride120_fresh/trial_0000/A8_pred_t0000.csv",
        ROOT / "rem_tcn_event_v9_manualscores_stride120_fresh/trial_0064/A8_pred0064.csv",
    ),
    Animal(
        "A11",
        ROOT / "newData/animal11_20230616B_trimmed.pq",
        ROOT / "newData/ManualScores/COMBINED_CAFFEINE11/manualscore.txt",
        "2023-06-20 00:00:00",
        ROOT / "newData/MLModelScores/20230616B/classicalModel/classicalModel_RAW_ranges_thr0.500.csv",
        ROOT / "rem_tcn_event_v9_classicalscores_stride120_fresh/trial_0000/A11_pred_t0000.csv",
        ROOT / "rem_tcn_event_v9_manualscores_stride120_fresh/trial_0064/A11_pred0064.csv",
    ),
    Animal(
        "A5",
        ROOT / "newData/animal5_20221003B_trimmed.pq",
        ROOT / "newData/ManualScores/COMBINED_CAFFEINE5/manualscore5.txt",
        "2022-10-04 00:00:00",
        ROOT / "newData/MLModelScores/animal5_20221003B_trimmed/classicalModel/classicalModel_RAW_ranges_thr0.500.csv",
        ROOT / "rem_tcn_event_v9_classicalscores_stride120_fresh/trial_0000/A5_pred_t0000.csv",
        ROOT / "rem_tcn_event_v9_manualscores_stride120_fresh/trial_0064/A5_pred0064.csv",
    ),
]

Interval = tuple[float, float]


def dt_seconds(value: str) -> float:
    # Treat all acquisition timestamps as timezone-naive wall-clock times.
    # datetime.timestamp() would apply the host's Pacific offset while Arrow's
    # timezone-naive datetime64 values do not, shifting streams by seven hours.
    return (datetime.fromisoformat(value) - datetime(1970, 1, 1)).total_seconds()


def merge_floor(intervals: list[Interval], gap: float, floor: float) -> list[Interval]:
    if not intervals:
        return []
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged if end - start >= floor]


def mask_to_intervals(times: np.ndarray, mask: np.ndarray) -> list[Interval]:
    """Convert one-second samples to half-open intervals [start, end)."""
    if len(times) == 0:
        return []
    times_s = times.astype("datetime64[s]").astype(np.int64)
    true_idx = np.flatnonzero(mask)
    if len(true_idx) == 0:
        return []
    breaks = np.flatnonzero((np.diff(true_idx) != 1) | (np.diff(times_s[true_idx]) != 1))
    starts_i = np.r_[0, breaks + 1]
    ends_i = np.r_[breaks, len(true_idx) - 1]
    return [
        (float(times_s[true_idx[s]]), float(times_s[true_idx[e]] + 1))
        for s, e in zip(starts_i, ends_i)
    ]


def _clock_seconds(token: str) -> int:
    hours, minutes, seconds = (int(part) for part in token.strip().split(":"))
    return hours * 3600 + minutes * 60 + seconds


def load_gt(path: Path, base: str, lo: float, hi: float) -> list[Interval]:
    base_s = dt_seconds(base)
    intervals: list[Interval] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            try:
                start = base_s + _clock_seconds(parts[0])
                end = base_s + _clock_seconds(parts[1])
            except (ValueError, IndexError):
                continue
            intervals.append((start, end))
    clipped = [(max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi]
    return [(s, e) for s, e in clipped if e > s]


def csv_coverage(path: Path) -> tuple[float, float]:
    first = last = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            t = dt_seconds(row["timestamp"])
            if first is None:
                first = t
            last = t
    assert first is not None and last is not None
    return first, last + 1.0


def load_prob_events(path: Path, threshold: float, gap: float, floor: float) -> list[Interval]:
    raw: list[Interval] = []
    active_start = None
    previous = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            t = dt_seconds(row["timestamp"])
            active = float(row["pred_prob_rem"]) >= threshold
            if active and active_start is None:
                active_start = t
            if active_start is not None and (not active or (previous is not None and t - previous != 1.0)):
                raw.append((active_start, (previous + 1.0) if previous is not None else t))
                active_start = t if active else None
            previous = t
    if active_start is not None and previous is not None:
        raw.append((active_start, previous + 1.0))
    return merge_floor(raw, gap, floor)


def load_rule_events(path: Path, lo: float, hi: float, gap: float, floor: float) -> list[Interval]:
    raw: list[Interval] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            start = dt_seconds(row["start_abs"])
            end = dt_seconds(row["end_abs"])
            if end > lo and start < hi:
                raw.append((max(start, lo), min(end, hi)))
    return merge_floor(raw, gap, floor)


def iou(a: Interval, b: Interval) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter <= 0:
        return 0.0
    return inter / ((a[1] - a[0]) + (b[1] - b[0]) - inter)


def adaptive_gate(gt: Interval) -> float:
    duration = gt[1] - gt[0]
    return max(0.0, 1.0 - 20.0 / duration) if duration > 0 else 1.0


def evaluate(gt: list[Interval], pred: list[Interval], gate: float | str) -> dict[str, float | int]:
    candidates: list[tuple[float, int, int]] = []
    pred = sorted(pred)
    scan_start = 0
    for gi, g in enumerate(sorted(gt)):
        threshold = adaptive_gate(g) if gate == "adaptive" else float(gate)
        while scan_start < len(pred) and pred[scan_start][1] <= g[0]:
            scan_start += 1
        pi = scan_start
        while pi < len(pred) and pred[pi][0] < g[1]:
            p = pred[pi]
            score = iou(g, p)
            if score > 0 and score >= threshold:
                candidates.append((score, gi, pi))
            pi += 1
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_g: set[int] = set()
    used_p: set[int] = set()
    for _, gi, pi in candidates:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
    tp = len(used_g)
    fp = len(pred) - tp
    fn = len(gt) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    # Poster operating points: rules return interval calls; the classical-label
    # TCN uses 0.80/merge 0 s; the expert-label TCN uses 0.50/merge 5 s.
    specs = {
        "Hand-tuned rules": (None, 0.0),
        "TCN, rule labels": (0.80, 0.0),
        "TCN, expert labels": (0.50, 5.0),
    }
    all_results: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    diagnostics: dict[str, dict[str, int | float]] = {}
    for animal in ANIMALS:
        lo1, hi1 = csv_coverage(animal.classical_csv)
        lo2, hi2 = csv_coverage(animal.expert_csv)
        lo, hi = max(lo1, lo2), min(hi1, hi2)
        # The poster's episode protocol applies the same merge-then-floor rule
        # to the reference intervals before event matching.
        gt = merge_floor(load_gt(animal.reference_txt, animal.reference_base, lo, hi), 5.0, 30.0)
        predictions = {
            "Hand-tuned rules": load_rule_events(animal.rule_csv, lo, hi, 0.0, 30.0),
            "TCN, rule labels": load_prob_events(animal.classical_csv, 0.80, 0.0, 30.0),
            "TCN, expert labels": load_prob_events(animal.expert_csv, 0.50, 5.0, 30.0),
        }
        diagnostics[animal.name] = {
            "coverage_start": lo,
            "coverage_end": hi,
            "reference_events": len(gt),
            **{f"{name}_events": len(iv) for name, iv in predictions.items()},
        }
        all_results[animal.name] = {}
        for model, pred in predictions.items():
            all_results[animal.name][model] = {
                "fixed_0.5": evaluate(gt, pred, 0.5),
                "fixed_0.7": evaluate(gt, pred, 0.7),
                "adaptive": evaluate(gt, pred, "adaptive"),
            }

    macro: dict[str, dict[str, float]] = {}
    for model in specs:
        macro[model] = {}
        for gate in ("fixed_0.5", "fixed_0.7", "adaptive"):
            for metric in ("precision", "recall", "f1"):
                macro[model][f"{gate}_{metric}"] = mean(
                    [float(all_results[a.name][model][gate][metric]) for a in ANIMALS]
                )

    payload = {
        "criterion": "IoU >= max(0, 1 - 20/L), where L is reference-event duration in seconds",
        "matching": "greedy one-to-one, descending IoU",
        "averaging": "unweighted macro-average over A6, A8, A11, and sealed A5",
        "sources": {
            animal.name: {
                "signal": str(animal.parquet),
                "reference": str(animal.reference_txt),
                "reference_base": animal.reference_base,
                "hand_tuned_rules": str(animal.rule_csv),
                "tcn_rule_labels": str(animal.classical_csv),
                "tcn_expert_labels": str(animal.expert_csv),
            }
            for animal in ANIMALS
        },
        "diagnostics": diagnostics,
        "animals": all_results,
        "macro": macro,
    }
    out = ROOT / "continuous_adaptive_iou_metrics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_out = ROOT / "continuous_adaptive_iou_metrics.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["animal", "model", "precision", "recall", "f1", "tp", "fp", "fn"],
        )
        writer.writeheader()
        for animal in ANIMALS:
            for model in specs:
                row = all_results[animal.name][model]["adaptive"]
                writer.writerow({"animal": animal.name, "model": model, **row})
        for model in specs:
            writer.writerow({
                "animal": "Macro average",
                "model": model,
                "precision": macro[model]["adaptive_precision"],
                "recall": macro[model]["adaptive_recall"],
                "f1": macro[model]["adaptive_f1"],
                "tp": "", "fp": "", "fn": "",
            })
    for model in specs:
        print(model, {
            "fixed_0.5_f1": round(macro[model]["fixed_0.5_f1"], 6),
            "fixed_0.7_f1": round(macro[model]["fixed_0.7_f1"], 6),
            "adaptive_f1": round(macro[model]["adaptive_f1"], 6),
        })
    print(f"Wrote {out}")
    print(f"Wrote {csv_out}")


if __name__ == "__main__":
    main()
