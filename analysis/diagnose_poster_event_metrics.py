#!/usr/bin/env python3
"""Sensitivity audit for the poster's held-out event-F1 values."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import evaluate_six_animal_adaptive_iou as ev


TARGETS = {
    "Hand-tuned rules": (0.843, 0.624),
    "TCN, rule labels": (0.854, 0.603),
    "TCN, expert labels": (0.919, 0.834),
}


def load_column_runs(path: Path, column: str, threshold: float) -> tuple[list[ev.Interval], float, float]:
    raw: list[ev.Interval] = []
    active_start = previous = first = last = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            timestamp = ev.dt_seconds(row["timestamp"])
            active = float(row[column]) >= threshold
            if first is None:
                first = timestamp
            if active_start is not None and previous is not None and timestamp - previous != 1.0:
                raw.append((active_start, previous + 1.0))
                active_start = None
            if active and active_start is None:
                active_start = timestamp
            elif not active and active_start is not None:
                raw.append((active_start, timestamp))
                active_start = None
            previous = last = timestamp
    if active_start is not None and last is not None:
        raw.append((active_start, last + 1.0))
    if first is None or last is None:
        raise ValueError(path)
    return raw, first, last + 1.0


def merge_only(intervals: list[ev.Interval], gap: float) -> list[ev.Interval]:
    return ev.merge_floor(intervals, gap, 0.0)


def postprocess(
    intervals: list[ev.Interval], gap: float, floor: float, order: str, endpoint: str
) -> list[ev.Interval]:
    if endpoint == "last-positive":
        intervals = [(start, end - 1.0) for start, end in intervals if end - 1.0 > start]
    if order == "merge-floor":
        return ev.merge_floor(intervals, gap, floor)
    kept = [(start, end) for start, end in intervals if end - start >= floor]
    return merge_only(kept, gap)


def aggregate(per_animal: list[dict], averaging: str) -> tuple[float, float]:
    if averaging == "macro":
        f05 = sum(row["f05"] for row in per_animal) / len(per_animal)
        f07 = sum(row["f07"] for row in per_animal) / len(per_animal)
        return f05, f07
    totals = {}
    for gate in ("r05", "r07"):
        tp = sum(row[gate]["tp"] for row in per_animal)
        fp = sum(row[gate]["fp"] for row in per_animal)
        fn = sum(row[gate]["fn"] for row in per_animal)
        totals[gate] = 2 * tp / (2 * tp + fp + fn)
    return totals["r05"], totals["r07"]


def main() -> None:
    held = [animal for animal in ev.ANIMALS if animal.name in ev.HELD_OUT]
    raw: dict[str, dict] = {}
    for animal in held:
        classical_prob, clo, chi = ev.load_probability_runs(animal.classical_csv, 0.80)
        expert_prob, elo, ehi = ev.load_probability_runs(animal.expert_csv, 0.50)
        classical_label, _, _ = load_column_runs(animal.classical_csv, "pred_label", 0.5)
        expert_label, _, _ = load_column_runs(animal.expert_csv, "pred_label", 0.5)
        lo, hi = max(clo, elo), min(chi, ehi)
        raw[animal.name] = {
            "lo": lo,
            "hi": hi,
            "reference": ev.clip(ev.load_elapsed_intervals(animal.reference_txt, animal.reference_base), lo, hi),
            "Hand-tuned rules": ev.clip(ev.load_elapsed_intervals(animal.rule_txt, animal.reference_base), lo, hi),
            "TCN, rule labels/prob": ev.clip(classical_prob, lo, hi),
            "TCN, rule labels/pred_label": ev.clip(classical_label, lo, hi),
            "TCN, expert labels/prob": ev.clip(expert_prob, lo, hi),
            "TCN, expert labels/pred_label": ev.clip(expert_label, lo, hi),
        }

    output: dict[str, list[dict]] = {}
    for model in TARGETS:
        variants: list[dict] = []
        sources = ["raw"] if model == "Hand-tuned rules" else ["prob", "pred_label"]
        for source, gt_gap, gt_floor, gt_order, pred_gap, pred_floor, pred_order, endpoint, averaging in itertools.product(
            sources, (0.0, 5.0), (0.0, 30.0), ("merge-floor", "floor-merge"),
            (0.0, 5.0, 10.0), (0.0, 20.0, 30.0), ("merge-floor", "floor-merge"),
            ("half-open", "last-positive"), ("macro", "pooled"),
        ):
            rows = []
            for animal in held:
                gt = postprocess(raw[animal.name]["reference"], gt_gap, gt_floor, gt_order, "half-open")
                key = model if model == "Hand-tuned rules" else f"{model}/{source}"
                pred = postprocess(raw[animal.name][key], pred_gap, pred_floor, pred_order, endpoint)
                r05 = ev.evaluate(gt, pred, 0.5)
                r07 = ev.evaluate(gt, pred, 0.7)
                rows.append({"f05": r05["f1"], "f07": r07["f1"], "r05": r05, "r07": r07})
            f05, f07 = aggregate(rows, averaging)
            target05, target07 = TARGETS[model]
            variants.append({
                "distance": abs(f05 - target05) + abs(f07 - target07),
                "f1_0.5": f05,
                "f1_0.7": f07,
                "source": source,
                "gt_gap": gt_gap,
                "gt_floor": gt_floor,
                "gt_order": gt_order,
                "pred_gap": pred_gap,
                "pred_floor": pred_floor,
                "pred_order": pred_order,
                "endpoint": endpoint,
                "averaging": averaging,
            })
        variants.sort(key=lambda row: row["distance"])
        output[model] = variants[:20]

    out = ev.ROOT / "poster_metric_sensitivity.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for model, variants in output.items():
        print(f"\n{model} target={TARGETS[model]}")
        for row in variants[:8]:
            print(row)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
