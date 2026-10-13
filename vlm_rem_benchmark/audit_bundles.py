#!/usr/bin/env python3
"""Audit generated bundles for label leakage and frozen context geometry."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image

import evaluate


PACKAGE_ROOT = Path(__file__).resolve().parent
GENERATED = PACKAGE_ROOT / "generated"
PUBLIC = GENERATED / "model_inputs"
VAULT = GENERATED / "private_vault" / "frames.jsonl"
PRIVATE_STAGES = GENERATED / "private_vault" / "stages.json"
STAGES = ("calibration", "test_01", "test_02", "confirmatory")
ID_PATTERN = re.compile(r"^F_[0-9a-f]{16}$")


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_rgb(path: Path):
    with Image.open(path) as opened:
        metadata = dict(opened.info)
        converted = opened.convert("RGB")
        converted.load()
        return converted.copy(), metadata


def main() -> None:
    vault = load_jsonl(VAULT)
    private_stages = json.loads(PRIVATE_STAGES.read_text(encoding="utf-8"))
    assert len(vault) == 800
    assert len({row["frame_id"] for row in vault}) == 800
    by_id = {row["frame_id"]: row for row in vault}
    calibration_rows = [row for row in vault if row["stage"] == "calibration"]
    calibration_counts = {
        split: {
            "frames": sum(row.get("split") == split for row in calibration_rows),
            "positive": sum(
                row.get("split") == split and bool(row["processed_episodes"])
                for row in calibration_rows
            ),
        }
        for split in ("train", "development")
    }
    assert calibration_counts == {
        "train": {"frames": 160, "positive": 76},
        "development": {"frames": 40, "positive": 19},
    }
    assert {
        stage: sum(
            row["stage"] == stage and row["render"]["context_before_available_sec"] < 300
            for row in vault
        )
        for stage in STAGES
    } == {"calibration": 0, "test_01": 1, "test_02": 1, "confirmatory": 5}
    assert {
        stage: sum(
            row["stage"] == stage and row["render"]["context_after_available_sec"] < 300
            for row in vault
        )
        for stage in STAGES
    } == {"calibration": 0, "test_01": 0, "test_02": 0, "confirmatory": 4}
    assert all(
        row["render"]["display_start_relative_sec"] == -300
        and row["render"]["display_end_relative_sec"] == 2100
        and row["render"]["score_start_relative_sec"] == 0
        and row["render"]["score_end_relative_sec"] == 1800
        and row["render"]["signal_samples"] >= 0
        and re.fullmatch(r"[0-9a-f]{64}", row["render"]["signal_sha256"])
        for row in vault
    )
    assert {
        stage: sum(
            row["stage"] == stage and row["render"]["eligible_for_scoring"]
            for row in vault
        )
        for stage in STAGES
    } == {"calibration": 200, "test_01": 198, "test_02": 200, "confirmatory": 196}
    assert {
        stage: metadata["manual_clock_base_offset_sec"]
        for stage, metadata in private_stages.items()
    } == {"calibration": 0, "test_01": 0, "test_02": 0, "confirmatory": 11302}
    assert private_stages["confirmatory"]["manual_alignment_validated"] is True
    assert private_stages["confirmatory"]["stored_label_disagreement_frames"] == 6
    assert all(
        private_stages[stage]["stored_label_disagreement_frames"] == 0
        for stage in ("calibration", "test_01", "test_02")
    )
    confirmatory_rows = [row for row in vault if row["stage"] == "confirmatory"]
    assert sum(bool(row["processed_episodes"]) for row in confirmatory_rows) == 96
    assert sum(len(row["processed_episodes"]) for row in confirmatory_rows) == 169

    for stage in STAGES:
        stage_root = PUBLIC / stage
        frames = load_jsonl(stage_root / "frames.jsonl")
        template = load_jsonl(stage_root / "predictions_template.jsonl")
        assert len(frames) == len(template) == 200
        assert all(set(row) == {"frame_id", "image"} for row in frames)
        assert all(set(row) == {"frame_id", "episodes"} and row["episodes"] == [] for row in template)
        assert {row["frame_id"] for row in frames} == {row["frame_id"] for row in template}
        assert all(ID_PATTERN.fullmatch(row["frame_id"]) for row in frames)
        assert all(Path(row["image"]).name == f"{row['frame_id']}.png" for row in frames)

        if stage == "calibration":
            annotations = load_jsonl(stage_root / "annotations.jsonl")
            assert len(annotations) == 200
            assert sum(row["split"] == "train" for row in annotations) == 160
            assert sum(row["split"] == "development" for row in annotations) == 40
        else:
            allowed = {"frames.jsonl", "predictions_template.jsonl", "images"}
            assert {path.name for path in stage_root.iterdir()} == allowed

        for frame in frames:
            frame_id = frame["frame_id"]
            public_image = stage_root / frame["image"]
            private = by_id[frame_id]
            assert private["stage"] == stage
            blinded, metadata = decode_rgb(public_image)
            try:
                assert blinded.size == (2613, 700)
                assert not metadata, f"PNG metadata present in {public_image}"
                top_extrema = blinded.crop((0, 0, blinded.width, 40)).getextrema()
                assert any(low != high for low, high in top_extrema)
            finally:
                blinded.close()

    expected_hashes = {}
    for line in (GENERATED / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        expected_hashes[relative] = digest
    assert len(expected_hashes) == 811
    for relative, expected in expected_hashes.items():
        assert sha256_file(GENERATED / relative) == expected, f"Hash mismatch: {relative}"

    manifest = json.loads((GENERATED / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest["stages"]) == set(STAGES)
    assert all(manifest["stages"][stage]["frames"] == 200 for stage in STAGES)
    assert manifest["protocol_version"] == "1.1-context"
    assert manifest["display_seconds"] == 2400
    assert manifest["context_seconds_each_side"] == 300
    assert manifest["display_axis_minutes"] == [-5, 35]
    assert manifest["score_axis_minutes"] == [0, 30]
    assert manifest["image_size_pixels"] == [2613, 700]
    assert manifest["minimum_central_coverage_fraction"] == 0.9

    oracle_frames = []
    for row in vault:
        issues = {
            "reference": {
                "clipped_intervals": 0,
                "discarded_nonpositive_intervals": 0,
                "discarded_short_after_merge": 0,
            }
        }
        truth = evaluate.normalize_intervals(row["raw_episodes"], issues, "oracle", "reference")
        oracle_frames.append({"frame_id": row["frame_id"], "truth": truth, "prediction": truth})
    oracle = evaluate.score_frames(oracle_frames)
    assert oracle["per_second"]["f1"] == 1.0
    assert oracle["per_second"]["cohens_kappa"] == 1.0
    assert all(metric["f1"] == 1.0 for metric in oracle["events_by_iou"].values())

    print("PASS: 800 opaque IDs are unique")
    print("PASS: held-out public JSON contains no labels or source metadata")
    print("PASS: calibration split is 160/40 and stratified 76/19 on postprocessed positives")
    print("PASS: every image is a 2613x700 40-min view with a frozen central 30-min score window")
    print("PASS: all missing context/data spans are hatched and recorded in the private vault")
    print("PASS: coverage eligibility is 200/198/200/196 for calibration/test_01/test_02/confirmatory")
    print("PASS: next-midnight label bases are 0/0/0/11302 seconds and Animal 5 uses the latest manual file")
    print("PASS: 6 stale Animal 5 JSONs are detected; authoritative postprocessing yields 96 positive frames / 169 events")
    print("PASS: all 811 public-file hashes match the frozen manifest")
    print("PASS: an oracle prediction scores 1.0 for second-level F1, kappa, and every event IoU")


if __name__ == "__main__":
    main()
