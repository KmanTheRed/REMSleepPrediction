#!/usr/bin/env python3
"""Export calibration plus one blinded stage outside the contaminated workspace."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


STAGES = ("calibration", "test_01", "test_02", "confirmatory")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parent
    workspace_root = package_root.parent.resolve()
    destination = args.destination.resolve()
    source = package_root / "generated" / "model_inputs" / args.stage
    calibration = package_root / "generated" / "model_inputs" / "calibration"

    if not source.is_dir():
        raise FileNotFoundError(f"Stage has not been generated: {source}")
    if destination == workspace_root or destination.is_relative_to(workspace_root):
        raise ValueError(
            "Destination must be outside the source workspace so its .claude state cannot be inherited"
        )
    if destination.exists():
        raise FileExistsError(f"Destination already exists; refusing to overwrite: {destination}")

    destination.mkdir(parents=True)
    if args.stage != "calibration":
        shutil.copytree(calibration, destination / "calibration")
    shutil.copytree(source, destination / args.stage)
    shutil.copy2(package_root / "prompt.txt", destination / "prompt.txt")
    shutil.copy2(package_root / "batch_task.txt", destination / "batch_task.txt")
    shutil.copy2(source / "predictions_template.jsonl", destination / "predictions_template.jsonl")

    print(f"Clean run exported to {destination}")
    print("Do not add source manifests, labels, scorers, or previous predictions.")


if __name__ == "__main__":
    main()
