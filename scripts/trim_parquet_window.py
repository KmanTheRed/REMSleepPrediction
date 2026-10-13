#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def normalize_id_value(x):
    if pd.isna(x):
        return ""
    return str(x).strip().replace(".0", "")


def get_time_series(df: pd.DataFrame) -> pd.Series:
    if "t_idx" in df.columns:
        return pd.to_datetime(df["t_idx"], errors="coerce")

    if df.index.name == "t_idx":
        return pd.Series(pd.to_datetime(df.index, errors="coerce"), index=df.index)

    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(df.index, errors="coerce"), index=df.index)

    raise RuntimeError(
        f"Could not recover t_idx. Columns={list(df.columns)}, index name={df.index.name}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--animal_id", required=True)
    ap.add_argument("--inclusive_end", action="store_true")
    args = ap.parse_args()

    input_path = str(Path(args.input))
    output_path = str(Path(args.output))
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    animal_id = str(args.animal_id).strip()

    print("Opening dataset...", flush=True)
    dataset = ds.dataset(input_path, format="parquet")

    print("\nSchema:", flush=True)
    print(dataset.schema, flush=True)

    print("\nArrow time filter...", flush=True)
    if args.inclusive_end:
        filt = (ds.field("t_idx") >= pa.scalar(start_ts.to_pydatetime())) & (
            ds.field("t_idx") <= pa.scalar(end_ts.to_pydatetime())
        )
    else:
        filt = (ds.field("t_idx") >= pa.scalar(start_ts.to_pydatetime())) & (
            ds.field("t_idx") < pa.scalar(end_ts.to_pydatetime())
        )

    table = dataset.to_table(filter=filt)
    print("Rows after Arrow time filter:", table.num_rows, flush=True)

    if table.num_rows == 0:
        raise RuntimeError("No rows found in requested time range.")

    print("\nConverting to pandas...", flush=True)
    df = table.to_pandas()

    print("Rows before pandas filtering:", len(df), flush=True)
    print("Original columns:", list(df.columns), flush=True)
    print("Original index name:", df.index.name, flush=True)

    df["__trim_time__"] = get_time_series(df)
    df = df[df["__trim_time__"].notna()].copy()

    if args.inclusive_end:
        df = df[(df["__trim_time__"] >= start_ts) & (df["__trim_time__"] <= end_ts)].copy()
    else:
        df = df[(df["__trim_time__"] >= start_ts) & (df["__trim_time__"] < end_ts)].copy()

    print("Rows after pandas time filter:", len(df), flush=True)

    if "id" not in df.columns:
        raise RuntimeError(f"'id' column not found. Columns={list(df.columns)}")

    df["id"] = df["id"].map(normalize_id_value)
    print("Unique ids in window (first 20):", sorted(df["id"].dropna().unique().tolist())[:20], flush=True)

    before_id = len(df)
    df = df[df["id"] == animal_id].copy()
    print(f"Rows after animal filter {animal_id}: {len(df)} / {before_id}", flush=True)

    if df.empty:
        raise RuntimeError(f"No rows remain for animal_id={animal_id}")

    df = df.reset_index(drop=True)

    if "t_idx" in df.columns:
        df = df.drop(columns=["t_idx"])

    df = df.rename(columns={"__trim_time__": "t_idx"})
    df = df.sort_values("t_idx").reset_index(drop=True)

    print("\nFinal rows:", len(df), flush=True)
    print("Final range:", df["t_idx"].min(), "->", df["t_idx"].max(), flush=True)
    print("Final unique ids:", sorted(df["id"].dropna().unique().tolist()), flush=True)

    print("\nSaving...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), output_path, compression="snappy")

    print("Done.", flush=True)
    print("Saved to:", output_path, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFATAL ERROR:", e, flush=True)
        traceback.print_exc()
        raise