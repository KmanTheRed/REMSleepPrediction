#!/usr/bin/env python3
"""Compare prespecified duration-dependent IoU gates on pre-caffeine data.

All streams use the established protocol: half-open intervals, common TCN
coverage, clip to the pre-caffeine window, merge gaps <=5 seconds, discard
episodes <30 seconds, and maximum-cardinality ordered one-to-one matching.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable

import evaluate_condition_split as split
import evaluate_six_animal_adaptive_iou as base


GateFunction = Callable[[float], float]
ROOT = base.ROOT


def gate_from_error(duration: float, error: float) -> float:
    if duration <= 0:
        return 1.0
    return max(0.5, (duration - error) / (duration + error))


def constant_gate(duration: float) -> float:
    return gate_from_error(duration, 10.0)


def logarithmic_gate(duration: float) -> float:
    # Anchors: e(30)=10 seconds and e(150)=15 seconds.
    error = 10.0 + 5.0 * math.log(duration / 30.0) / math.log(5.0)
    return gate_from_error(duration, error)


def linear_gate(duration: float) -> float:
    # Anchors: e(30)=10 seconds and e(150)=15 seconds.
    error = 10.0 + (duration - 30.0) / 24.0
    return gate_from_error(duration, error)


def hybrid_gate(duration: float) -> float:
    # Absolute 10-second allowance or 10% of duration, whichever is larger.
    # This also gives e(30)=10 seconds and e(150)=15 seconds.
    error = max(10.0, 0.10 * duration)
    return gate_from_error(duration, error)


GATES: dict[str, GateFunction] = {
    "constant_e10": constant_gate,
    "log_e10_at_30_e15_at_150": logarithmic_gate,
    "linear_e10_at_30_e15_at_150": linear_gate,
    "hybrid_max_10s_10pct": hybrid_gate,
}


def evaluate_gate(
    reference: list[base.Interval],
    predicted: list[base.Interval],
    gate_function: GateFunction,
) -> dict[str, float | int]:
    """Globally maximize valid matches, then summed IoU, in temporal order."""
    reference = sorted(reference)
    predicted = sorted(predicted)
    pred_count = len(predicted)
    previous_counts = [0] * (pred_count + 1)
    previous_sums = [0.0] * (pred_count + 1)

    for ref in reference:
        threshold = gate_function(ref[1] - ref[0])
        current_counts = [0] * (pred_count + 1)
        current_sums = [0.0] * (pred_count + 1)
        for j, pred in enumerate(predicted, start=1):
            best_count = previous_counts[j]
            best_sum = previous_sums[j]

            left_count = current_counts[j - 1]
            left_sum = current_sums[j - 1]
            if left_count > best_count or (
                left_count == best_count and left_sum > best_sum
            ):
                best_count, best_sum = left_count, left_sum

            score = base.iou(ref, pred)
            if score >= threshold:
                match_count = previous_counts[j - 1] + 1
                match_sum = previous_sums[j - 1] + score
                if match_count > best_count or (
                    match_count == best_count and match_sum > best_sum
                ):
                    best_count, best_sum = match_count, match_sum

            current_counts[j] = best_count
            current_sums[j] = best_sum
        previous_counts, previous_sums = current_counts, current_sums

    tp = previous_counts[-1]
    fp = len(predicted) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": previous_sums[-1] / tp if tp else 0.0,
    }


def macro(results: dict, names: tuple[str, ...]) -> dict:
    output: dict[str, dict] = {}
    for model in base.MODELS:
        output[model] = {}
        for gate_name in GATES:
            rows = [results[name][model][gate_name] for name in names]
            output[model][gate_name] = {
                metric: sum(float(row[metric]) for row in rows) / len(rows)
                for metric in ("precision", "recall", "f1", "mean_matched_iou")
            }
    return output


def pooled(results: dict, names: tuple[str, ...]) -> dict:
    output: dict[str, dict] = {}
    for model in base.MODELS:
        output[model] = {}
        for gate_name in GATES:
            tp = sum(int(results[name][model][gate_name]["tp"]) for name in names)
            fp = sum(int(results[name][model][gate_name]["fp"]) for name in names)
            fn = sum(int(results[name][model][gate_name]["fn"]) for name in names)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            output[model][gate_name] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
    return output


def report_table(block: dict) -> list[str]:
    labels = {
        "constant_e10": "Constant e=10 s",
        "log_e10_at_30_e15_at_150": "Logarithmic",
        "linear_e10_at_30_e15_at_150": "Linear",
        "hybrid_max_10s_10pct": "Hybrid max(10 s, 10%)",
    }
    lines = [
        "| Detector | Constant e=10 s | Logarithmic | Linear | Hybrid |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in base.MODELS:
        values = block[model]
        lines.append(
            f"| {model} | "
            + " | ".join(f"{values[name]['f1']:.3f}" for name in labels)
            + " |"
        )
    return lines


def main() -> None:
    boundaries, boundary_source = split.caffeine_boundaries()
    results: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}

    for animal in base.ANIMALS:
        classical_raw, classical_lo, classical_hi = base.load_probability_runs(
            animal.classical_csv, 0.80
        )
        expert_raw, expert_lo, expert_hi = base.load_probability_runs(
            animal.expert_csv, 0.50
        )
        lo = max(classical_lo, expert_lo)
        hi = min(classical_hi, expert_hi)
        caffeine_start = boundaries[animal.name]
        if caffeine_start is not None:
            hi = min(hi, caffeine_start)
        if hi <= lo:
            raise ValueError(f"No pre-caffeine coverage for {animal.name}")

        reference = base.merge_floor(base.clip(base.load_elapsed_intervals(
            animal.reference_txt, animal.reference_base
        ), lo, hi))
        predictions = {
            "Hand-tuned rules": base.merge_floor(base.clip(base.load_elapsed_intervals(
                animal.rule_txt, animal.reference_base
            ), lo, hi)),
            "TCN, rule labels": base.merge_floor(base.clip(classical_raw, lo, hi)),
            "TCN, expert labels": base.merge_floor(base.clip(expert_raw, lo, hi)),
        }

        diagnostics[animal.name] = {
            "pre_caffeine_start": split.iso(lo),
            "pre_caffeine_end_exclusive": split.iso(hi),
            "reference_events": len(reference),
            **{f"{model}_events": len(events) for model, events in predictions.items()},
        }
        results[animal.name] = {
            model: {
                gate_name: evaluate_gate(reference, events, gate_function)
                for gate_name, gate_function in GATES.items()
            }
            for model, events in predictions.items()
        }

    all_names = tuple(animal.name for animal in base.ANIMALS)
    payload = {
        "protocol": {
            "condition": "pre-caffeine; boundary is first native caffeine value > 0",
            "condition_boundary_source": boundary_source,
            "postprocessing": "clip raw intervals, merge gaps <= 5 s, discard duration < 30 s",
            "applied_to": "reference and all prediction streams",
            "tcn_rule_label_threshold": 0.80,
            "tcn_expert_label_threshold": 0.50,
            "matching": "ordered maximum-cardinality one-to-one; maximum summed IoU tie-break",
            "gate_floor": 0.50,
            "gate_models": {
                "constant_e10": "e(d)=10",
                "log_e10_at_30_e15_at_150": "e(d)=10 + 5*ln(d/30)/ln(5)",
                "linear_e10_at_30_e15_at_150": "e(d)=10 + (d-30)/24",
                "hybrid_max_10s_10pct": "e(d)=max(10, 0.10*d)",
            },
            "aggregation": "unweighted macro-average over animals",
        },
        "gate_examples": {
            str(duration): {
                name: function(float(duration)) for name, function in GATES.items()
            }
            for duration in (30, 60, 90, 120, 150, 240)
        },
        "diagnostics": diagnostics,
        "animals": results,
        "macro_all_6": macro(results, all_names),
        "macro_held_out_4": macro(results, base.HELD_OUT),
        "pooled_all_6": pooled(results, all_names),
        "pooled_held_out_4": pooled(results, base.HELD_OUT),
    }

    json_path = ROOT / "precaffeine_gate_model_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = ROOT / "precaffeine_gate_model_metrics.csv"
    fields = [
        "cohort", "animal", "model", "gate_model", "precision", "recall", "f1",
        "mean_matched_iou", "tp", "fp", "fn",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for animal in base.ANIMALS:
            for model in base.MODELS:
                for gate_name in GATES:
                    writer.writerow({
                        "cohort": "animal",
                        "animal": animal.name,
                        "model": model,
                        "gate_model": gate_name,
                        **results[animal.name][model][gate_name],
                    })
        for cohort_name, names in (
            ("macro_all_6", all_names),
            ("macro_held_out_4", base.HELD_OUT),
        ):
            block = payload[cohort_name]
            for model in base.MODELS:
                for gate_name in GATES:
                    writer.writerow({
                        "cohort": cohort_name,
                        "animal": "",
                        "model": model,
                        "gate_model": gate_name,
                        **block[model][gate_name],
                        "tp": "",
                        "fp": "",
                        "fn": "",
                    })

    report_path = ROOT / "precaffeine_gate_model_report.md"
    lines = [
        "# Pre-caffeine event F1 under alternative uncertainty models",
        "",
        "All values are unweighted macro-averages across animals.",
        "",
        "## Held-out animals A5, A6, A8, A11",
        "",
        *report_table(payload["macro_held_out_4"]),
        "",
        "## All six animals",
        "",
        *report_table(payload["macro_all_6"]),
        "",
        "The constant model is the primary analysis. The logarithmic, linear, and hybrid models are sensitivity analyses. All gates have a minimum IoU of 0.5. The logarithmic and linear models satisfy e(30)=10 seconds and e(150)=15 seconds. The hybrid model uses e(d)=max(10 seconds, 10% of duration), which has the same two anchor values.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Held-out pre-caffeine macro event F1")
    for model in base.MODELS:
        print(model)
        for gate_name in GATES:
            print(f"  {gate_name}: {payload['macro_held_out_4'][model][gate_name]['f1']:.6f}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
