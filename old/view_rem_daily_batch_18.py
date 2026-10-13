#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
view_rem_daily_batch_18_multi.py

Batch exporter: for each dataset (parquet + CSV) and for
each day covered by that dataset, build the
view_rem_panel_anyrange2 app clipped to that day and
save it as a standalone Bokeh HTML file.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import panel as pn

from panel.io.save import save as save_panel  # correct saver for Panel

# Import your existing viewer utilities
from view_rem_panel_anyrange2 import build_app, _ensure_dtindex

pn.extension("bokeh")


def get_day_range(parquet_path: Path) -> list[pd.Timestamp]:
    """Return list of midnights (one per day with data) for a parquet file."""
    raw = pd.read_parquet(parquet_path)
    idx = _ensure_dtindex(raw).index

    data_start = idx.min().normalize()
    # last instant with data → corresponding day
    data_end = (idx.max() - pd.Timedelta(seconds=1)).normalize()

    if data_end < data_start:
        raise ValueError(f"End day before start day in {parquet_path}")

    days = list(pd.date_range(data_start, data_end, freq="D"))
    return days


def main():
    ap = argparse.ArgumentParser(
        description="Export daily REM viewer HTML files for up to two datasets."
    )
    ap.add_argument("--data1", required=True, help="Parquet path for dataset 1")
    ap.add_argument("--pred_csv1", required=True, help="Prediction CSV for dataset 1")
    ap.add_argument("--data2", required=False, help="Parquet path for dataset 2")
    ap.add_argument("--pred_csv2", required=False, help="Prediction CSV for dataset 2")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used inside the viewer (for metrics).",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directory where daily HTML files will be written.",
    )
    ap.add_argument(
        "--skip-classical-rem",
        action="store_true",
        help="Skip classical REM computation to speed up.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[tuple[str, Path, Path]] = []

    datasets.append(
        (
            Path(args.pred_csv1).stem,  # label
            Path(args.data1),
            Path(args.pred_csv1),
        )
    )

    if args.data2 and args.pred_csv2:
        datasets.append(
            (
                Path(args.pred_csv2).stem,
                Path(args.data2),
                Path(args.pred_csv2),
            )
        )

    if not datasets:
        raise ValueError("No datasets configured.")

    for label, data_path, pred_path in datasets:
        print(f"\n=== Dataset: {label} ===")
        days = get_day_range(data_path)
        print(f"  Days: {days[0].date()} → {days[-1].date()}  ({len(days)} day(s))")

        for day in days:
            start_ts = pd.Timestamp(day)
            end_ts = start_ts + pd.Timedelta(days=1)

            print(f"  - Building app for {start_ts.date()} ...")
            app = build_app(
                data_path,
                pred_path,
                threshold=float(args.threshold),
                skip_classical_rem=args.skip_classical_rem,
                start_ts=start_ts,
                end_ts=end_ts,
                clip=True,  # clip to that day for speed
            )

            out_file = out_dir / f"{label}_REM_{start_ts:%Y-%m-%d}.html"
            print(f"    → Saving to {out_file}")
            save_panel(
                app,
                filename=str(out_file),
                title=f"REM viewer — {label} — {start_ts:%Y-%m-%d}",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
