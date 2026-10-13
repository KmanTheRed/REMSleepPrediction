#!/usr/bin/env python3
"""Create label-safe public bundles and a separate private scoring vault."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import re
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


FRAME_SECONDS = 1800
CONTEXT_SECONDS = 300
DISPLAY_SECONDS = FRAME_SECONDS + 2 * CONTEXT_SECONDS
DPI = 140
FIGURE_WIDTH_INCHES = 56 / 3
FIGURE_HEIGHT_INCHES = 5
EXPECTED_IMAGE_SIZE = (2613, 700)
SPLIT_SEED = "rem-vlm-calibration-split-v1"


@dataclass(frozen=True)
class Stage:
    name: str
    relative_source: str
    labels_dir: str
    relative_signal: str
    exposes_labels: bool
    expected_manual_base_offset_sec: int
    relative_manual: str | None = None


STAGES = (
    Stage(
        "calibration",
        "qwen_rem_dataset",
        "labels",
        "newData/COMBINED_CAFFEINE_A8_2023-06-08_to_2023-06-22.pq",
        True,
        0,
    ),
    Stage(
        "test_01",
        "animal11_qwen_demo",
        "labels",
        "newData/animal11_20230616B_trimmed.pq",
        False,
        0,
    ),
    Stage(
        "test_02",
        "newData/animal6_qwen_dataset",
        "labels",
        "newData/COMBINED_CAFFEINE_A6_2023-06-05_to_2023-06-08.pq",
        False,
        0,
    ),
    Stage(
        "confirmatory",
        "newData/animal5_qwen_dataset",
        "labels_corrected",
        "newData/animal5_20221003B_trimmed.pq",
        False,
        11302,
        "newData/ManualScores/COMBINED_CAFFEINE5/manualscore5.txt",
    ),
)


def json_line(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opaque_id(salt: bytes, stage: str, source_image: str) -> str:
    digest = hashlib.sha256(salt + b"\0" + stage.encode() + b"\0" + source_image.encode()).hexdigest()
    return f"F_{digest[:16]}"


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 200:
        raise ValueError(f"Expected 200 frames in {path}, found {len(rows)}")
    return rows


def load_episodes(label_path: Path) -> list[dict[str, int]]:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    episodes: list[dict[str, int]] = []
    for item in payload.get("rem_episodes", []):
        start = int(item["start_offset_sec"])
        end = int(item["end_offset_sec"])
        if not (0 <= start < end <= FRAME_SECONDS):
            raise ValueError(f"Out-of-frame reference interval in {label_path}: {start}, {end}")
        episodes.append({"start_sec": start, "end_sec": end})
    return episodes


def postprocess(episodes: Iterable[dict[str, int]]) -> list[dict[str, int]]:
    clipped = sorted(
        (max(0, int(x["start_sec"])), min(FRAME_SECONDS, int(x["end_sec"])))
        for x in episodes
    )
    merged: list[list[int]] = []
    for start, end in clipped:
        if start >= end:
            continue
        if merged and start <= merged[-1][1] + 5:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [
        {"start_sec": start, "end_sec": end}
        for start, end in merged
        if end - start >= 30
    ]


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    low, high = np.nanpercentile(finite, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
        if low == high:
            low -= 1.0
            high += 1.0
    padding = 0.08 * (high - low)
    return float(low - padding), float(high + padding)


def load_signals(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp]:
    if not path.is_file():
        raise FileNotFoundError(f"Raw signal source is missing: {path}")
    frame = pd.read_parquet(path, columns=["t_idx", "CBV", "Motion Ang. Vector"])
    if "t_idx" in frame.columns:
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame["t_idx"]))
    elif frame.index.name == "t_idx":
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame.index))
    else:
        raise ValueError(f"No t_idx time axis found in {path}")

    cbv = -pd.to_numeric(frame["CBV"], errors="coerce").to_numpy(dtype=np.float64)
    motion = pd.to_numeric(frame["Motion Ang. Vector"], errors="coerce").to_numpy(dtype=np.float64)
    valid_time = ~timestamps.isna()
    timestamps = timestamps[valid_time]
    cbv = cbv[valid_time]
    motion = motion[valid_time]
    if not timestamps.is_monotonic_increasing:
        order = np.argsort(timestamps.asi8, kind="stable")
        timestamps = timestamps[order]
        cbv = cbv[order]
        motion = motion[order]
    if timestamps.size == 0:
        raise ValueError(f"No usable timestamps in {path}")
    recording_start = pd.Timestamp(timestamps[0])
    elapsed = (timestamps - recording_start).total_seconds().to_numpy(dtype=np.float64)
    return elapsed, cbv, motion, recording_start


def next_midnight_base(recording_start: pd.Timestamp) -> tuple[pd.Timestamp, int]:
    midnight = recording_start.normalize()
    base = midnight if recording_start == midnight else midnight + pd.Timedelta(days=1)
    return base, int(round((base - recording_start).total_seconds()))


def parse_manual_seconds(path: Path, base_offset_sec: int) -> list[tuple[int, int]]:
    pattern = re.compile(r"(\d+:\d{2}:\d{2})\s*,\s*(\d+:\d{2}:\d{2})")

    def seconds(value: str) -> int:
        hours, minutes, secs = map(int, value.split(":"))
        return hours * 3600 + minutes * 60 + secs

    intervals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        start = seconds(match.group(1)) + base_offset_sec
        end = seconds(match.group(2)) + base_offset_sec
        if end > start:
            intervals.append((start, end))
    return sorted(intervals)


def episodes_for_window(
    intervals: list[tuple[int, int]], score_start_sec: int
) -> list[dict[str, int]]:
    score_end_sec = score_start_sec + FRAME_SECONDS
    return [
        {
            "start_sec": max(start, score_start_sec) - score_start_sec,
            "end_sec": min(end, score_end_sec) - score_start_sec,
        }
        for start, end in intervals
        if min(end, score_end_sec) > max(start, score_start_sec)
    ]


def signal_digest(times: np.ndarray, cbv: np.ndarray, motion: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in (times, cbv, motion):
        digest.update(np.asarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def render_context_png(
    elapsed: np.ndarray,
    cbv: np.ndarray,
    motion: np.ndarray,
    score_start_sec: int,
    destination: Path,
) -> dict[str, Any]:
    display_start_sec = score_start_sec - CONTEXT_SECONDS
    display_end_sec = score_start_sec + FRAME_SECONDS + CONTEXT_SECONDS
    left = int(np.searchsorted(elapsed, display_start_sec, side="left"))
    right = int(np.searchsorted(elapsed, display_end_sec, side="left"))
    times = elapsed[left:right]
    relative_minutes = (times - score_start_sec) / 60.0
    window_cbv = cbv[left:right]
    window_motion = motion[left:right]
    central_left = int(np.searchsorted(elapsed, score_start_sec, side="left"))
    central_right = int(
        np.searchsorted(elapsed, score_start_sec + FRAME_SECONDS, side="left")
    )
    central_signal_samples = central_right - central_left

    missing_spans_sec: list[tuple[float, float]] = []
    if times.size == 0:
        missing_spans_sec.append((display_start_sec, display_end_sec))
    else:
        if times[0] - display_start_sec > 1.5:
            missing_spans_sec.append((display_start_sec, float(times[0])))
        for previous, following in zip(times[:-1], times[1:]):
            if following - previous > 1.5:
                missing_spans_sec.append((float(previous), float(following)))
        if display_end_sec - times[-1] > 1.5:
            missing_spans_sec.append((float(times[-1]), display_end_sec))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
        sharex=True,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#111111")
    series = ((window_cbv, "CBV (IOS inverted)"), (window_motion, "Motion Ang. Vector"))
    gap_at_next_sample = (
        np.concatenate(([False], np.diff(times) > 1.5)) if times.size else np.array([], dtype=bool)
    )

    for axis, (values, label) in zip(axes, series):
        axis.set_facecolor("#151515")
        axis.axvspan(-5, 0, color="#8d99ae", alpha=0.20, zorder=0)
        axis.axvspan(30, 35, color="#8d99ae", alpha=0.20, zorder=0)
        for missing_start, missing_end in missing_spans_sec:
            axis.axvspan(
                (missing_start - score_start_sec) / 60.0,
                (missing_end - score_start_sec) / 60.0,
                facecolor="#050505",
                alpha=0.95,
                hatch="///",
                zorder=1,
            )
        plotted_values = values.copy()
        plotted_values[gap_at_next_sample] = np.nan
        axis.plot(relative_minutes, plotted_values, linewidth=0.9, color="#1f77b4", zorder=2)
        axis.axvline(0, color="#f2f2f2", linestyle="--", linewidth=1.2, zorder=3)
        axis.axvline(30, color="#f2f2f2", linestyle="--", linewidth=1.2, zorder=3)
        axis.set_ylabel(label, color="white", fontsize=9)
        axis.tick_params(colors="white", labelsize=8)
        axis.grid(True, color="#333333", alpha=0.35, linewidth=0.5)
        axis.set_ylim(*robust_limits(values))
        for spine in axis.spines.values():
            spine.set_color("#555555")

    axes[0].text(
        15,
        0.98,
        "SCORE ONLY 0–30 MIN",
        transform=axes[0].get_xaxis_transform(),
        color="#ffffff",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="top",
    )
    axes[0].text(-2.5, 0.98, "CONTEXT", transform=axes[0].get_xaxis_transform(), color="#c9ced6", fontsize=8, ha="center", va="top")
    axes[0].text(32.5, 0.98, "CONTEXT", transform=axes[0].get_xaxis_transform(), color="#c9ced6", fontsize=8, ha="center", va="top")
    for missing_start, missing_end in missing_spans_sec:
        if missing_end - missing_start < 60:
            continue
        axes[0].text(
            ((missing_start + missing_end) / 2 - score_start_sec) / 60.0,
            0.50,
            "NO RECORDING",
            transform=axes[0].get_xaxis_transform(),
            color="#dddddd",
            fontsize=7,
            ha="center",
            va="center",
            rotation=90,
        )

    axes[-1].set_xlim(-5, 35)
    axes[-1].set_xticks(np.arange(-5, 36, 5))
    axes[-1].set_xlabel(
        "Minutes relative to scored interval (report only 0–30 min)", color="white"
    )
    fig.suptitle(
        "40-min view: 5-min context | SCORE ONLY central 30 min (0–30) | 5-min context",
        color="white",
        fontsize=12,
    )

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as opened:
        image = opened.convert("RGB")
    if image.size != EXPECTED_IMAGE_SIZE:
        raise ValueError(f"Unexpected rendered image size {image.size}; expected {EXPECTED_IMAGE_SIZE}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, "PNG", optimize=True)

    before_available = min(
        CONTEXT_SECONDS,
        int(np.searchsorted(elapsed, score_start_sec, side="left"))
        - int(np.searchsorted(elapsed, score_start_sec - CONTEXT_SECONDS, side="left")),
    )
    after_available = min(
        CONTEXT_SECONDS,
        int(
            np.searchsorted(
                elapsed, score_start_sec + FRAME_SECONDS + CONTEXT_SECONDS, side="left"
            )
        )
        - int(np.searchsorted(elapsed, score_start_sec + FRAME_SECONDS, side="left")),
    )
    return {
        "display_start_relative_sec": -CONTEXT_SECONDS,
        "display_end_relative_sec": FRAME_SECONDS + CONTEXT_SECONDS,
        "score_start_relative_sec": 0,
        "score_end_relative_sec": FRAME_SECONDS,
        "context_before_available_sec": int(round(before_available)),
        "context_after_available_sec": int(round(after_available)),
        "central_signal_samples": int(central_signal_samples),
        "central_coverage_fraction": min(1.0, central_signal_samples / FRAME_SECONDS),
        "eligible_for_scoring": central_signal_samples >= int(0.9 * FRAME_SECONDS),
        "missing_display_seconds": int(
            round(sum(end - start for start, end in missing_spans_sec))
        ),
        "signal_samples": int(times.size),
        "signal_sha256": signal_digest(times - score_start_sec, window_cbv, window_motion),
    }


def calibration_splits(records: list[dict[str, Any]]) -> dict[str, str]:
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        outcome = "positive" if record["processed_episodes"] else "negative"
        by_kind.setdefault(outcome, []).append(record)
    result: dict[str, str] = {}
    for kind, items in by_kind.items():
        ranked = sorted(
            items,
            key=lambda x: hashlib.sha256(
                f"{SPLIT_SEED}|{kind}|{x['frame_id']}".encode()
            ).hexdigest(),
        )
        development_count = round(len(ranked) * 0.2)
        for index, item in enumerate(ranked):
            result[item["frame_id"]] = "development" if index < development_count else "train"
    counts = {name: sum(value == name for value in result.values()) for name in ("train", "development")}
    if counts != {"train": 160, "development": 40}:
        raise ValueError(f"Unexpected calibration split: {counts}")
    return result


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json_line(row) + "\n")


def build(root: Path, output: Path, salt_path: Path, rebuild: bool) -> None:
    if output.exists():
        if not rebuild:
            raise FileExistsError(f"{output} exists; pass --rebuild only before scored runs")
        shutil.rmtree(output)

    public_root = output / "model_inputs"
    private_root = output / "private_vault"
    private_root.mkdir(parents=True)
    if salt_path.exists():
        salt = salt_path.read_bytes()
        if len(salt) != 32:
            raise ValueError(f"Expected a 32-byte ID salt in {salt_path}")
    else:
        salt = secrets.token_bytes(32)
        salt_path.parent.mkdir(parents=True, exist_ok=True)
        salt_path.write_bytes(salt)

    vault: list[dict[str, Any]] = []
    private_stage_metadata: dict[str, dict[str, Any]] = {}
    public_files: list[Path] = []

    for stage in STAGES:
        source_root = root / stage.relative_source
        signal_path = root / stage.relative_signal
        rows = load_manifest(source_root / "manifest.csv")
        stage_root = public_root / stage.name
        image_root = stage_root / "images"
        stage_records: list[dict[str, Any]] = []
        print(f"Loading raw signals for {stage.name}: {signal_path}")
        elapsed, cbv, motion, recording_start = load_signals(signal_path)
        manual_base, manual_base_offset_sec = next_midnight_base(recording_start)
        if manual_base_offset_sec != stage.expected_manual_base_offset_sec:
            raise ValueError(
                f"Unexpected manual-label base offset for {stage.name}: "
                f"computed {manual_base_offset_sec}, expected {stage.expected_manual_base_offset_sec}"
            )
        manual_intervals = None
        if stage.relative_manual:
            manual_path = root / stage.relative_manual
            manual_intervals = parse_manual_seconds(manual_path, manual_base_offset_sec)
        private_stage_metadata[stage.name] = {
            "source_signal": str(signal_path.resolve()),
            "recording_start": recording_start.isoformat(),
            "manual_clock_base": manual_base.isoformat(),
            "manual_clock_base_rule": (
                "recording date when start is 00:00:00; otherwise next calendar date at 00:00:00"
            ),
            "manual_clock_base_offset_sec": manual_base_offset_sec,
            "manual_alignment_validated": bool(manual_intervals is not None),
            "reference_source": (
                str((root / stage.relative_manual).resolve())
                if stage.relative_manual
                else str((source_root / stage.labels_dir).resolve())
            ),
        }
        stored_label_disagreement_frames = 0

        for index, row in enumerate(rows, start=1):
            source_image_rel = row["image"].replace("\\", "/")
            source_image = source_root / Path(source_image_rel)
            label_name = Path(row["label_json"].replace("\\", "/")).name
            source_label = source_root / stage.labels_dir / label_name
            if not source_image.is_file() or not source_label.is_file():
                raise FileNotFoundError(f"Missing pair: {source_image} / {source_label}")

            frame_id = opaque_id(salt, stage.name, source_image_rel)
            destination = image_root / f"{frame_id}.png"
            stored_episodes = load_episodes(source_label)
            episodes = stored_episodes
            frame_start_sec = int(row["frame_start_sec"])
            if manual_intervals is not None:
                expected_episodes = episodes_for_window(manual_intervals, frame_start_sec)
                if stored_episodes != expected_episodes:
                    stored_label_disagreement_frames += 1
                episodes = expected_episodes
            render_metadata = render_context_png(
                elapsed,
                cbv,
                motion,
                frame_start_sec,
                destination,
            )
            public_files.append(destination)

            record = {
                "frame_id": frame_id,
                "stage": stage.name,
                "source_image": str(source_image.resolve()),
                "source_label": str(source_label.resolve()),
                "source_signal": str(signal_path.resolve()),
                "source_kind": row.get("kind", "unknown"),
                "stored_label_disagrees_with_reference": stored_episodes != episodes,
                "frame_start_sec": frame_start_sec,
                "raw_episodes": episodes,
                "processed_episodes": postprocess(episodes),
                "render": render_metadata,
            }
            stage_records.append(record)
            if index % 25 == 0 or index == len(rows):
                print(f"  rendered {index}/{len(rows)} for {stage.name}", flush=True)

        del elapsed, cbv, motion
        gc.collect()
        private_stage_metadata[stage.name][
            "stored_label_disagreement_frames"
        ] = stored_label_disagreement_frames

        if len({x["frame_id"] for x in stage_records}) != len(stage_records):
            raise ValueError(f"Opaque ID collision in {stage.name}")

        if stage.exposes_labels:
            splits = calibration_splits(stage_records)
            for record in stage_records:
                record["split"] = splits[record["frame_id"]]
            annotations = [
                {
                    "frame_id": x["frame_id"],
                    "image": f"images/{x['frame_id']}.png",
                    "split": x["split"],
                    "episodes": x["raw_episodes"],
                }
                for x in sorted(stage_records, key=lambda x: (x["split"], x["frame_id"]))
            ]
            write_jsonl(stage_root / "annotations.jsonl", annotations)
            write_jsonl(stage_root / "train.jsonl", (x for x in annotations if x["split"] == "train"))
            write_jsonl(
                stage_root / "development.jsonl",
                (x for x in annotations if x["split"] == "development"),
            )
            public_files.extend(
                [stage_root / "annotations.jsonl", stage_root / "train.jsonl", stage_root / "development.jsonl"]
            )

        frames = [
            {"frame_id": x["frame_id"], "image": f"images/{x['frame_id']}.png"}
            for x in sorted(stage_records, key=lambda x: x["frame_id"])
        ]
        write_jsonl(stage_root / "frames.jsonl", frames)
        write_jsonl(
            stage_root / "predictions_template.jsonl",
            ({"frame_id": x["frame_id"], "episodes": []} for x in frames),
        )
        public_files.extend([stage_root / "frames.jsonl", stage_root / "predictions_template.jsonl"])
        vault.extend(stage_records)

    write_jsonl(private_root / "frames.jsonl", sorted(vault, key=lambda x: (x["stage"], x["frame_id"])))
    (private_root / "stages.json").write_text(
        json.dumps(private_stage_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    hashes_path = output / "SHA256SUMS.txt"
    stage_hash_material: dict[str, list[str]] = {stage.name: [] for stage in STAGES}
    with hashes_path.open("w", encoding="utf-8", newline="\n") as handle:
        for path in sorted(public_files):
            digest = sha256_file(path)
            relative = path.relative_to(output).as_posix()
            handle.write(f"{digest}  {relative}\n")
            stage_name = Path(relative).parts[1]
            stage_hash_material[stage_name].append(f"{digest}  {relative}")

    bundle_manifest = {
        "protocol_version": "1.1-context",
        "frame_seconds": FRAME_SECONDS,
        "display_seconds": DISPLAY_SECONDS,
        "context_seconds_each_side": CONTEXT_SECONDS,
        "display_axis_minutes": [-5, 35],
        "score_axis_minutes": [0, 30],
        "image_size_pixels": list(EXPECTED_IMAGE_SIZE),
        "minimum_central_coverage_fraction": 0.9,
        "stages": {
            stage.name: {
                "frames": sum(x["stage"] == stage.name for x in vault),
                "bundle_sha256": hashlib.sha256(
                    ("\n".join(sorted(stage_hash_material[stage.name])) + "\n").encode()
                ).hexdigest(),
            }
            for stage in STAGES
        },
    }
    (output / "BUNDLE_MANIFEST.json").write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Built {len(vault)} frames at {output}")
    for stage in STAGES:
        count = sum(x["stage"] == stage.name for x in vault)
        positive = sum(bool(x["processed_episodes"]) for x in vault if x["stage"] == stage.name)
        events = sum(len(x["processed_episodes"]) for x in vault if x["stage"] == stage.name)
        print(f"  {stage.name}: {count} frames, {positive} positive after rules, {events} events")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "generated")
    parser.add_argument(
        "--salt-file",
        type=Path,
        default=Path(__file__).resolve().parent / "private_id_salt.bin",
    )
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    build(args.root.resolve(), args.output.resolve(), args.salt_file.resolve(), args.rebuild)


if __name__ == "__main__":
    main()
