#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_inference.py

Runs a trained REM TCN v9 model on a new sensor parquet and saves predictions
to a CSV that is fully compatible with view_pred_vs_manual_4panel.py.

Required files in --best_dir:
  model.pt         – model weights
  scaler.pkl       – fitted StandardScaler
  postprocess.json – temperature, event_mix, thresholds, gap/min rules
  config.json      – architecture hyperparameters + feature list

Output CSV columns:
  timestamp        – datetime of the *last* step in each inference window
  pred_prob_rem    – combined (frame + event_mix * event) probability, temperature-scaled
  pred_label       – 0/1 after hysteresis + merge + min-length post-processing
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import load
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def clean_cols(cols):
    return [re.sub(r"[^\w]+", "_", str(c)).strip("_") for c in cols]


def ensure_finite_df(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    if df.isna().any().any():
        df = df.ffill().bfill()
        num = df.select_dtypes(include=[np.number, "bool"]).columns
        df[num] = df[num].fillna(0.0)
    return df


def ensure_finite_np(x, name=""):
    if not np.isfinite(x).all():
        print(f"  WARN {name}: fixing {(~np.isfinite(x)).sum()} non-finite values")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Parquet loading
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_dt_index(df: pd.DataFrame, sort: bool = False) -> pd.DataFrame:
    """Ensure the dataframe has a tz-naive DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["t", "time", "timestamp", "datetime", "date", "index",
                    "Timestamp", "t_idx", "__index_level_0__"]:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col)
                    break
                except Exception:
                    pass
        else:
            raise RuntimeError(
                "Cannot find a datetime index/column. "
                f"Available columns: {list(df.columns)}"
            )

    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        try:
            df.index = df.index.tz_convert(None)
        except Exception:
            df.index = pd.to_datetime(df.index).tz_localize(None)

    if sort:
        df = df.sort_index()
    return df


# IMPORTANT:
# We match against CLEANED names, not raw parquet names.
# Example: raw 'Motion Ang. X' -> cleaned 'Motion_Ang_X'
_NEEDED_COLS = {
    # time candidates
    "t", "time", "timestamp", "datetime", "date", "index",
    "Timestamp", "t_idx", "__index_level_0__",

    # signal candidates
    "IOS", "OIS", "OIS_norm",

    # motion vector or components
    "Motion_Ang_Vector",
    "Motion_Ang_X", "Motion_Ang_Y", "Motion_Ang_Z",
    "Motion_Lin_X", "Motion_Lin_Y", "Motion_Lin_Z",

    # lowercase/alternate cleaned variants
    "motion_ang_vector",
    "motion_ang_x", "motion_ang_y", "motion_ang_z",
    "motion_lin_x", "motion_lin_y", "motion_lin_z",

    # optional aux
    "caffeine", "Caffeine",
}


def _read_one_part(path: Path,
                   start_ts: pd.Timestamp | None,
                   end_ts: pd.Timestamp | None) -> pd.DataFrame | None:
    """
    Read a single parquet part, keeping only useful columns and the requested
    date window. Returns None if the part contributes no rows.

    FIX:
    Match candidate columns using CLEANED names, because raw parquet names may
    be things like 'Motion Ang. X' and only become 'Motion_Ang_X' after
    clean_cols(...).
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    schema_names = list(pf.schema.names)

    # raw_name -> cleaned_name
    cleaned_map = {raw: clean_cols([raw])[0] for raw in schema_names}
    needed_clean_lower = {c.lower() for c in _NEEDED_COLS}

    # keep raw columns whose CLEANED version matches something we need
    keep = [
        raw for raw, cleaned in cleaned_map.items()
        if cleaned.lower() in needed_clean_lower
    ]

    # If nothing matched, read everything as a fallback
    read_cols = keep if keep else None

    try:
        df = pd.read_parquet(str(path), columns=read_cols)
    except Exception:
        df = pd.read_parquet(str(path))

    if len(df) == 0:
        return None

    df = _normalise_dt_index(df, sort=False)

    # Clean names immediately so downstream logic sees canonical names
    df.columns = clean_cols(df.columns)

    # clip early for RAM savings
    if start_ts is not None or end_ts is not None:
        idx = df.index
        mask = np.ones(len(idx), dtype=bool)
        if start_ts is not None:
            mask &= (idx >= start_ts)
        if end_ts is not None:
            mask &= (idx <= end_ts)
        df = df.loc[mask]

    return df if len(df) > 0 else None


def read_parquet(path: Path,
                 start_ts: pd.Timestamp | None = None,
                 end_ts: pd.Timestamp | None = None) -> pd.DataFrame:
    path = Path(path)

    # ── partitioned directory ─────────────────────────────────────────────────
    if path.is_dir():
        parts = sorted(path.glob("*.parquet")) + sorted(path.glob("*.pq"))
        if not parts:
            raise RuntimeError(f"No .parquet/.pq files found in directory: {path}")

        print(f"  Partitioned parquet: {len(parts)} part files in {path.name}")
        if start_ts or end_ts:
            print(f"  Clipping per-part to {start_ts} → {end_ts}  (saves RAM)")

        BATCH = 10
        batched = []
        current_batch = []
        kept = 0

        def _flush_batch(batch):
            if not batch:
                return None
            if len(batch) == 1:
                return batch[0]
            return pd.concat(batch, ignore_index=False)

        for i, p in enumerate(parts):
            chunk = _read_one_part(p, start_ts, end_ts)
            if chunk is not None:
                current_batch.append(chunk)
                kept += len(chunk)

            if len(current_batch) >= BATCH or (i + 1) == len(parts):
                merged = _flush_batch(current_batch)
                if merged is not None:
                    batched.append(merged)
                current_batch = []

            if (i + 1) % 10 == 0 or (i + 1) == len(parts):
                print(f"    [{i+1:3d}/{len(parts)}] parts read  rows so far: {kept:,}")

        if not batched:
            raise RuntimeError(
                f"No rows found in {path} for the requested date range "
                f"({start_ts} → {end_ts}). Check --start / --end."
            )

        if len(batched) == 1:
            df = batched[0]
        else:
            df = pd.concat(batched, ignore_index=False)

        del batched, current_batch
        print(f"  Concatenated: {len(df):,} rows")

        df = _normalise_dt_index(df, sort=True)
        df.columns = clean_cols(df.columns)
        return df

    # ── single parquet file ───────────────────────────────────────────────────
    df = pd.read_parquet(str(path))
    df = _normalise_dt_index(df, sort=True)
    df.columns = clean_cols(df.columns)

    if start_ts is not None or end_ts is not None:
        idx = df.index
        mask = np.ones(len(idx), dtype=bool)
        if start_ts is not None:
            mask &= (idx >= start_ts)
        if end_ts is not None:
            mask &= (idx <= end_ts)
        df = df.loc[mask]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering (mirrors training script exactly)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_motion_vector(df: pd.DataFrame) -> pd.DataFrame:
    lmap = {c.lower(): c for c in df.columns}

    # case-insensitive exact match first
    if "motion_ang_vector" in lmap:
        actual_col = lmap["motion_ang_vector"]
        out = df.copy() if actual_col != "Motion_Ang_Vector" else df
        out["Motion_Ang_Vector"] = pd.to_numeric(out[actual_col], errors="coerce").astype("float32")
        return out

    # compute from x/y/z components
    for trip in [
        ("motion_ang_x", "motion_ang_y", "motion_ang_z"),
        ("motion_lin_x", "motion_lin_y", "motion_lin_z"),
    ]:
        cols = [lmap[k] for k in trip if k in lmap]
        if len(cols) == 3:
            out = df.copy()
            vals = out[cols].apply(pd.to_numeric, errors="coerce").astype("float32")
            out["Motion_Ang_Vector"] = np.sqrt((vals ** 2).sum(axis=1)).astype("float32")
            return out

    raise RuntimeError(
        f"Cannot construct Motion_Ang_Vector. Available columns: {list(df.columns)}"
    )


def ensure_ois_norm(df: pd.DataFrame) -> pd.DataFrame:
    lmap = {c.lower(): c for c in df.columns}

    if "ois_norm" in lmap:
        actual_col = lmap["ois_norm"]
        out = df.copy() if actual_col != "OIS_norm" else df
        out["OIS_norm"] = pd.to_numeric(out[actual_col], errors="coerce").astype("float32")
        return out

    src = None
    for candidate in ("ios", "ois"):
        if candidate in lmap:
            src = lmap[candidate]
            break

    if not src:
        raise RuntimeError(f"Need OIS_norm, IOS, or OIS column. Available: {list(df.columns)}")

    s = pd.to_numeric(df[src], errors="coerce").astype("float32").replace([np.inf, -np.inf], np.nan)

    lo = np.nanpercentile(s.dropna(), 0.5) if s.notna().any() else 0.0
    hi = np.nanpercentile(s.dropna(), 99.5) if s.notna().any() else 1.0

    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        lo = float(np.nanmin(s)) if s.notna().any() else 0.0
        hi = float(np.nanmax(s)) if s.notna().any() else 1.0

    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        lo, hi = 0.0, 1.0

    out = df.copy()
    out["OIS_norm"] = ((s - lo) / max(1e-9, hi - lo)).clip(0, 1).astype("float32")
    return out


def make_feature_frame(
    df: pd.DataFrame,
    shared_base: list[str],
    use_time_features: bool = False,
    use_derivatives: bool = True,
) -> pd.DataFrame:
    feat = df[shared_base].astype("float32").copy()

    if use_derivatives:
        for c in list(shared_base):
            feat[f"d_{c}"] = feat[c].diff().fillna(0).astype("float32")

    if use_time_features:
        secs = (
            df.index.hour.to_numpy(dtype=np.float32) * 3600.0
            + df.index.minute.to_numpy(dtype=np.float32) * 60.0
            + df.index.second.to_numpy(dtype=np.float32)
        )
        feat["time_sin"] = np.sin(2 * np.pi * secs / 86400.0).astype("float32")
        feat["time_cos"] = np.cos(2 * np.pi * secs / 86400.0).astype("float32")

    return ensure_finite_df(feat.ffill().bfill().fillna(0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture (must match training script exactly)
# ─────────────────────────────────────────────────────────────────────────────

def GroupNorm1d(ch, groups=8):
    groups = min(groups, ch)
    while ch % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, ch)


class MultiKernelStem(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(5, 15, 31)):
        super().__init__()
        self.branches = nn.ModuleList()
        klist = list(kernels)
        base = max(8, out_ch // len(klist))
        for k in klist:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_ch, base, kernel_size=k, padding=k // 2, bias=False),
                GroupNorm1d(base),
                nn.SiLU(),
            ))
        self.proj = nn.Linear(base * len(klist), out_ch)

    def forward(self, x):
        xc = x.transpose(1, 2)
        outs = [b(xc).transpose(1, 2) for b in self.branches]
        return self.proj(torch.cat(outs, dim=-1))


class ResidualTCNBlock(nn.Module):
    def __init__(self, d_model, kernel_size=7, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.ln1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=pad, dilation=dilation, bias=False)
        self.gn1 = GroupNorm1d(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=pad, dilation=dilation, bias=False)
        self.gn2 = GroupNorm1d(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln1(x).transpose(1, 2)
        h = self.conv1(h)
        h = F.silu(self.gn1(h))
        h = self.drop(h)
        h = h.transpose(1, 2)

        h = self.ln2(x + h).transpose(1, 2)
        h = self.conv2(h)
        h = F.silu(self.gn2(h))
        h = self.drop(h)
        h = h.transpose(1, 2)
        return x + h


class EventSupportHead(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                d_model, d_model, kernel_size,
                padding=kernel_size // 2,
                groups=max(1, d_model // 8),
                bias=False
            ),
            GroupNorm1d(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
            GroupNorm1d(d_model),
            nn.SiLU(),
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.head(h).squeeze(-1)


class DualBranchEventTCN(nn.Module):
    def __init__(self, feature_names, d_model=128, stem_dim=48, blocks=8,
                 kernel_size=7, dropout=0.12):
        super().__init__()
        self.feature_names = list(feature_names)

        motion_idx, ois_idx, aux_idx = [], [], []
        for i, nm in enumerate(self.feature_names):
            nl = nm.lower()
            if "motion_ang_vector" in nl:
                motion_idx.append(i)
            elif "ois_norm" in nl:
                ois_idx.append(i)
            else:
                aux_idx.append(i)

        if not motion_idx or not ois_idx:
            raise RuntimeError(f"Could not partition features into motion/ois/aux: {self.feature_names}")

        self.register_buffer("motion_idx", torch.tensor(motion_idx, dtype=torch.long), persistent=False)
        self.register_buffer("ois_idx",    torch.tensor(ois_idx,    dtype=torch.long), persistent=False)
        self.register_buffer("aux_idx",    torch.tensor(aux_idx,    dtype=torch.long), persistent=False)

        self.motion_stem = MultiKernelStem(len(motion_idx), stem_dim)
        self.ois_stem    = MultiKernelStem(len(ois_idx), stem_dim)

        fuse_in = stem_dim * 2 + len(aux_idx)
        self.in_proj = nn.Sequential(
            nn.LayerNorm(fuse_in),
            nn.Linear(fuse_in, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        dilations = [1, 2, 4, 8, 16, 32]
        self.blocks = nn.ModuleList([
            ResidualTCNBlock(d_model, kernel_size, dilations[i % len(dilations)], dropout)
            for i in range(blocks)
        ])

        self.frame_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.event_head = EventSupportHead(
            d_model,
            kernel_size=max(15, kernel_size * 4 + 3),
            dropout=dropout,
        )

    def forward(self, x):
        xm = x.index_select(dim=2, index=self.motion_idx)
        xo = x.index_select(dim=2, index=self.ois_idx)

        hm = self.motion_stem(xm)
        ho = self.ois_stem(xo)

        parts = [hm, ho]
        if self.aux_idx.numel() > 0:
            parts.append(x.index_select(dim=2, index=self.aux_idx))

        h = torch.cat(parts, dim=-1)
        h = self.in_proj(h)

        for blk in self.blocks:
            h = blk(h)

        zf = self.frame_head(h).squeeze(-1)
        ze = self.event_head(h)
        return {"frame": zf, "event": ze}


# ─────────────────────────────────────────────────────────────────────────────
# Sequence helpers
# ─────────────────────────────────────────────────────────────────────────────

def contiguous_segments(index: pd.DatetimeIndex, step: float = 1.0):
    if len(index) == 0:
        return []
    diffs = index.to_series().diff().dt.total_seconds().fillna(step).to_numpy()
    breaks = np.where(np.abs(diffs - step) > 1e-6)[0]
    ss = np.r_[0, breaks]
    es = np.r_[breaks, len(index)]
    return [(int(s), int(e)) for s, e in zip(ss, es) if e > s]


def starts_from_segs(segs, L: int, stride: int = 1):
    out = []
    for s, e in segs:
        if e - s >= L:
            out.extend(range(s, e - L + 1, stride))
    return np.asarray(out, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class InferenceDS(Dataset):
    def __init__(self, X: np.ndarray, starts: np.ndarray, win: int):
        self.X = X
        self.starts = starts.astype(np.int64)
        self.win = int(win)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        return self.X[s:s + self.win].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def merge_runs(runs, gap: int):
    if not runs:
        return []
    gap = int(gap)
    out = [list(runs[0])]
    for s, e in runs[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def filter_runs_min_len(runs, min_len: int):
    return [(s, e) for s, e in runs if (e - s) >= int(min_len)]


def runs_to_mask(n: int, runs):
    m = np.zeros(int(n), dtype=np.uint8)
    for s, e in runs:
        m[int(s):int(e)] = 1
    return m


def hysteresis_decode_runs(probs, start_thr: float, stop_thr: float):
    probs = np.asarray(probs, dtype=np.float32)
    runs, on, s = [], False, None
    for i, p in enumerate(probs):
        if not on:
            if p >= start_thr:
                on, s = True, i
        else:
            if p < stop_thr:
                runs.append((s, i))
                on, s = False, None
    if on and s is not None:
        runs.append((s, len(probs)))
    return runs


def decode_postprocessed_runs(probs, start_thr, stop_thr, gap_steps, min_steps):
    pred_runs = hysteresis_decode_runs(probs, start_thr, stop_thr)
    pred_runs = merge_runs(pred_runs, gap_steps)
    pred_runs = filter_runs_min_len(pred_runs, min_steps)
    return pred_runs


# ─────────────────────────────────────────────────────────────────────────────
# Main inference routine
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: DualBranchEventTCN,
    X: np.ndarray,
    idx: pd.DatetimeIndex,
    starts: np.ndarray,
    win: int,
    batch_size: int,
    device: torch.device,
    event_mix: float,
    temperature: float,
    start_thr: float,
    stop_thr: float,
    gap_steps: int,
    min_steps: int,
    num_workers: int = 0,
) -> pd.DataFrame:
    model.eval()
    use_amp = device.type == "cuda"

    ds = InferenceDS(X, starts, win)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_probs_combined = []
    all_probs_frame = []
    all_probs_event = []
    all_timestamps = []

    cursor = 0
    n_batches = len(loader)

    for bi, xb in enumerate(loader):
        xb = xb.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(xb)
            zf = torch.nan_to_num(out["frame"], nan=0.0, posinf=0.0, neginf=0.0)
            ze = torch.nan_to_num(out["event"], nan=0.0, posinf=0.0, neginf=0.0)
            z = zf + float(event_mix) * ze

        pf = torch.sigmoid(zf[:, -1] / temperature).float().cpu().numpy()
        pe = torch.sigmoid(ze[:, -1] / temperature).float().cpu().numpy()
        pc = torch.sigmoid(z[:, -1] / temperature).float().cpu().numpy()

        bs = len(pc)
        batch_starts = starts[cursor:cursor + bs]
        cursor += bs

        for s, pcf, pff, pef in zip(batch_starts, pc, pf, pe):
            all_timestamps.append(idx[int(s) + win - 1])
            all_probs_combined.append(float(pcf))
            all_probs_frame.append(float(pff))
            all_probs_event.append(float(pef))

        if (bi + 1) % max(1, n_batches // 10) == 0 or (bi + 1) == n_batches:
            pct = 100.0 * (bi + 1) / n_batches
            print(f"  [{pct:5.1f}%] batch {bi+1}/{n_batches}  windows processed so far: {cursor:,}")

    print(f"  Total windows: {cursor:,}")

    probs_arr = np.asarray(all_probs_combined, dtype=np.float32)
    pred_runs = decode_postprocessed_runs(probs_arr, start_thr, stop_thr, gap_steps, min_steps)
    pred_mask = runs_to_mask(len(probs_arr), pred_runs)

    df = pd.DataFrame({
        "timestamp": all_timestamps,
        "pred_prob_rem": all_probs_combined,
        "pred_prob_frame_head": all_probs_frame,
        "pred_prob_event_head": all_probs_event,
        "pred_label": pred_mask.astype(np.int8),
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Run REM TCN v9 inference on a new parquet file."
    )
    ap.add_argument(
        "--best_dir", required=True,
        help="Path to the 'best' folder produced by train_rem_tcn_event_v9.py "
             "(must contain model.pt, scaler.pkl, postprocess.json, config.json)"
    )
    ap.add_argument(
        "--parquet", required=True,
        help="Sensor parquet file (.pq / .parquet)"
    )
    ap.add_argument(
        "--start", default=None,
        help="Clip start datetime, e.g. '2023-06-05 00:00:00'"
    )
    ap.add_argument(
        "--end", default=None,
        help="Clip end datetime, e.g. '2023-06-08 00:00:00'"
    )
    ap.add_argument(
        "--out_csv", required=True,
        help="Output CSV file path"
    )
    ap.add_argument(
        "--win", type=int, default=None,
        help="Inference window length in steps (overrides config.json 'win')"
    )
    ap.add_argument(
        "--stride", type=int, default=1,
        help="Stride between windows (default: 1 = every step, dense)"
    )
    ap.add_argument(
        "--batch_size", type=int, default=256,
        help="Inference batch size (default: 256)"
    )
    ap.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader worker processes (default: 0)"
    )
    ap.add_argument(
        "--device", default=None,
        help="'cuda' or 'cpu' (auto-detected if omitted)"
    )
    ap.add_argument(
        "--expected_step_seconds", type=float, default=1.0,
        help="Expected seconds between consecutive rows (default: 1.0)"
    )

    args = ap.parse_args()

    best_dir = Path(args.best_dir)
    for fname in ("model.pt", "scaler.pkl", "postprocess.json", "config.json"):
        if not (best_dir / fname).exists():
            raise FileNotFoundError(f"Missing required file: {best_dir / fname}")

    with open(best_dir / "config.json") as f:
        cfg = json.load(f)
    with open(best_dir / "postprocess.json") as f:
        pp = json.load(f)

    print("=" * 60)
    print("Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("\nPost-process params:")
    for k, v in pp.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    win = int(args.win or cfg.get("win", 600))
    use_time_features = bool(cfg.get("use_time_features", False))
    use_derivatives = bool(cfg.get("use_derivative_features", True))
    feature_names = cfg["feature_names"]
    shared_base = cfg.get("shared_base_features", ["Motion_Ang_Vector", "OIS_norm"])

    temperature = float(pp.get("temperature", 1.0))
    event_mix = float(pp.get("event_mix", 0.6))
    start_thr = float(pp.get("start_threshold", 0.55))
    stop_thr = float(pp.get("stop_threshold", 0.35))
    gap_s = float(pp.get("gap_seconds", 5.0))
    min_ep_s = float(pp.get("min_episode_seconds", 30.0))
    step_s = float(args.expected_step_seconds)

    gap_steps = max(0, int(round(gap_s / step_s)))
    min_steps = max(1, int(round(min_ep_s / step_s)))

    print(f"Window={win}  stride={args.stride}  temperature={temperature:.4f}")
    print(f"event_mix={event_mix:.3f}  start_thr={start_thr:.3f}  stop_thr={stop_thr:.3f}")
    print(f"gap_steps={gap_steps}  min_steps={min_steps}")
    print()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    clip_start = pd.Timestamp(args.start) if args.start else None
    clip_end = pd.Timestamp(args.end) if args.end else None

    print(f"\nLoading parquet: {args.parquet}")
    df_raw = read_parquet(Path(args.parquet), start_ts=clip_start, end_ts=clip_end)
    print(f"  Rows after load+clip: {len(df_raw):,}  range: {df_raw.index.min()} → {df_raw.index.max()}")

    if len(df_raw) == 0:
        raise RuntimeError("No data remaining after date clipping.")
    if len(df_raw) < win:
        raise RuntimeError(
            f"Only {len(df_raw)} rows after clipping, but window size is {win}. "
            "Need at least as many rows as the window."
        )

    print("\nBuilding features…")
    df_raw = ensure_finite_df(df_raw)
    df_raw = ensure_motion_vector(df_raw)
    df_raw = ensure_ois_norm(df_raw)
    feat = make_feature_frame(df_raw, shared_base, use_time_features, use_derivatives)

    if list(feat.columns) != feature_names:
        print("  WARNING: feature column mismatch!")
        print(f"    Expected : {feature_names}")
        print(f"    Got      : {list(feat.columns)}")
        new_feat = pd.DataFrame(0.0, index=feat.index, columns=feature_names, dtype="float32")
        for c in feature_names:
            if c in feat.columns:
                new_feat[c] = feat[c].values
        feat = new_feat
        print("  Padded missing features with zeros and reordered.")

    print(f"  Features: {list(feat.columns)}")
    print(f"  Shape   : {feat.shape}")

    print("\nLoading scaler…")
    scaler = load(best_dir / "scaler.pkl")
    X = scaler.transform(feat.to_numpy(np.float32)).astype("float32")
    X = ensure_finite_np(X, "scaled features")
    idx = feat.index

    segs = contiguous_segments(idx, step=step_s)
    starts = starts_from_segs(segs, win, stride=args.stride)
    print(f"\nContiguous segments : {len(segs)}")
    print(f"Inference windows   : {len(starts):,}")

    if len(starts) == 0:
        raise RuntimeError(
            "No inference windows created. The clipped data may be shorter than the window size "
            f"({win} steps) or the time series has gaps larger than expected_step_seconds={step_s}s."
        )

    print("\nLoading model…")
    model = DualBranchEventTCN(
        feature_names=feature_names,
        d_model=int(cfg.get("d_model", 128)),
        stem_dim=int(cfg.get("stem_dim", 48)),
        blocks=int(cfg.get("blocks", 8)),
        kernel_size=int(cfg.get("kernel_size", 7)),
        dropout=float(cfg.get("dropout", 0.12)),
    ).to(device)

    state = torch.load(best_dir / "model.pt", map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    print("\nRunning inference…")
    pred_df = run_inference(
        model=model,
        X=X,
        idx=idx,
        starts=starts,
        win=win,
        batch_size=args.batch_size,
        device=device,
        event_mix=event_mix,
        temperature=temperature,
        start_thr=start_thr,
        stop_thr=stop_thr,
        gap_steps=gap_steps,
        min_steps=min_steps,
        num_workers=args.num_workers,
    )

    rem_count = int(pred_df["pred_label"].sum())
    total_count = len(pred_df)
    print("\nPrediction summary:")
    print(f"  Total timesteps   : {total_count:,}")
    print(f"  Predicted REM (1) : {rem_count:,}  ({100 * rem_count / max(1, total_count):.1f}%)")
    print(f"  p(REM) mean       : {pred_df['pred_prob_rem'].mean():.4f}")
    print(f"  Timestamp range   : {pred_df['timestamp'].min()} → {pred_df['timestamp'].max()}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)

    print(f"\n✔ Saved predictions to: {out_path}")
    print(f"  Columns: {list(pred_df.columns)}")
    print("\nDone.")


if __name__ == "__main__":
    main()