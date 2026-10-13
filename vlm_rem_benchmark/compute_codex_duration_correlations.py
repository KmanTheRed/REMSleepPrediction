from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from scipy.stats import norm, pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import load_json_or_jsonl, prepare


ROOT = Path(r"D:\downloads\Kethan\Kethan\vlm_rem_benchmark")
VAULT = ROOT / "generated" / "private_vault" / "frames.jsonl"


def duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def correlation_report(predictions: Path, stages: list[str]) -> dict[str, object]:
    by_stage, issues = prepare(
        load_json_or_jsonl(VAULT), load_json_or_jsonl(predictions), stages
    )
    frames = [frame for stage in stages for frame in by_stage[stage]]
    reference = [duration(frame["truth"]) for frame in frames]
    predicted = [duration(frame["prediction"]) for frame in frames]

    pearson = pearsonr(reference, predicted)
    spearman = spearmanr(reference, predicted)
    r = float(pearson.statistic)
    n = len(frames)
    z = math.atanh(r)
    delta = float(norm.ppf(0.975)) / math.sqrt(n - 3)
    ci = [math.tanh(z - delta), math.tanh(z + delta)]
    return {
        "stages": stages,
        "n_pairs": n,
        "pearson_r": r,
        "pearson_95_ci_fisher_z": ci,
        "pearson_p_two_tailed": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_two_tailed": float(spearman.pvalue),
        "quality_exclusions": issues["quality_exclusions"],
        "pair_definition": "postprocessed REM seconds per eligible 30-minute frame",
    }


report = {
    "combined": correlation_report(
        ROOT / "generated" / "codex_test_01_test_02_predictions.jsonl",
        ["test_01", "test_02"],
    ),
    "confirmatory": correlation_report(
        Path(r"D:\vlm_clean_runs\chatgpt_sol_ultra_test_01_fresh\predictions.jsonl"),
        ["confirmatory"],
    ),
}

output = ROOT / "generated" / "codex_duration_correlations.json"
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
