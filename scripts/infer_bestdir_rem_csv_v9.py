#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dateutil import parser as dtparser
from joblib import load as joblib_load

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# =========================================================
# Helpers
# =========================================================
def clean_cols(cols):
    return [re.sub(r"[^\w]+", "_", str(c)).strip("_") for c in cols]


def banner(msg: str):
    bar = "=" * len(msg)
    print(f"\n{bar}\n{msg}\n{bar}\n")


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def ensure_finite_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)
    if df.isna().any().any():
        df = df.ffill().bfill()
        num = df.select_dtypes(include=[np.number, "bool"]).columns
        df[num] = df[num].fillna(0.0)
    return df


def ensure_finite_np(x: np.ndarray, name: str = "") -> np.ndarray:
    if not np.isfinite(x).all():
        bad = (~np.isfinite(x)).sum()
        print(f"WARN {name}: fixing {bad} non-finite values")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def read_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(str(path))

    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["t", "time", "timestamp", "datetime", "date", "index", "Timestamp", "t_idx", "__index_level_0__"]:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col).sort_index()
                    break
                except Exception:
                    pass
        else:
            try:
                idx = pd.to_datetime(df.index)
                df.index = idx
            except Exception as e:
                raise RuntimeError(f"Need DatetimeIndex or datetime column in {path}") from e

    df = df.copy()
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        try:
            df.index = df.index.tz_convert(None)
        except Exception:
            df.index = pd.to_datetime(df.index).tz_localize(None)

    df = df.sort_index()
    df.columns = clean_cols(df.columns)
    return df


def slice_df(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        df = df.loc[pd.Timestamp(start):]
    if end:
        df = df.loc[:pd.Timestamp(end)]
    return df


# =========================================================
# Time / manual-score parsing
# =========================================================
def _strip(x):
    return str(x).strip().strip('"').strip("'").replace("\ufeff", "")


def _norm_sep(s):
    return s.replace("\ufeff", "").replace("→", "->").replace("—", "->").replace("–", "->")


def _parse_hms(token, base_date=None):
    token = _strip(token)
    m = re.fullmatch(r"(\d+):(\d{1,2})(?::(\d{1,2}))?", token)
    if not m or base_date is None:
        return None
    hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if mm >= 60 or ss >= 60:
        return None
    return pd.Timestamp(base_date) + pd.to_timedelta(hh * 3600 + mm * 60 + ss, "s")


def parse_time_token(token, base_date=None):
    token = _strip(token)
    if not token:
        return None

    try:
        val = float(token)
        return pd.Timestamp(base_date) + pd.to_timedelta(val, "s") if base_date else None
    except Exception:
        pass

    ts = _parse_hms(token, base_date)
    if ts is not None:
        return ts

    try:
        default = pd.Timestamp(base_date).to_pydatetime() if base_date else None
        ts = dtparser.parse(token, fuzzy=True, default=default)
        ts = pd.Timestamp(ts)
        if ts.tzinfo:
            ts = ts.tz_localize(None)
    except Exception:
        return None

    if base_date and not re.search(r"\b\d{4}\b", token):
        bd = pd.Timestamp(base_date)
        mnames = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12
        }
        m2 = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", token.lower())
        if m2:
            pm = mnames[m2.group(1)]
            yr = bd.year
            if pm < bd.month - 2:
                yr += 1
            elif pm > bd.month + 9:
                yr -= 1
            try:
                ts = ts.replace(year=yr)
            except ValueError:
                pass
    return ts


REM_RE = re.compile(r"(?i)^(rem|r|2)$")
NONREM_RE = re.compile(r"(?i)^(nrem|n|wake|awake|w|0|1|3)$")


def norm_tok(t):
    return re.sub(r"\s+", "", _strip(t))


def tok2bin(t, rr=REM_RE, nr=NONREM_RE):
    t = norm_tok(t)
    if rr.match(t):
        return 1
    if nr.match(t):
        return 0
    return None


def _looks_header(s):
    s0 = _strip(s)
    return not s0 or s0.startswith(("#", "//"))


def parse_score_lines(path, base_date=None, epoch_s=30, rr=REM_RE, nr=NONREM_RE):
    rows = []
    with open(str(path), "r", encoding="utf-8", errors="ignore") as f:
        raw = [s.rstrip("\n") for s in f]
    lines = [s.strip() for s in raw if s.strip() and not _looks_header(s.strip())]

    for _ in range(2):
        if len(lines) >= 2:
            first = _strip(lines[0])
            if (
                re.fullmatch(r"[A-Za-z0-9_.\- ]+", first)
                and tok2bin(first, rr, nr) is None
                and parse_time_token(first, base_date) is None
            ):
                lines = lines[1:]

    if not lines:
        raise RuntimeError(f"Empty/unparseable: {path}")

    if all(tok2bin(l, rr, nr) is not None for l in lines):
        if not base_date:
            raise RuntimeError(f"Dense labels need base_date: {path}")
        t0 = pd.Timestamp(base_date)
        for i, l in enumerate(lines):
            s = t0 + pd.to_timedelta(i * epoch_s, "s")
            rows.append((s, s + pd.to_timedelta(epoch_s, "s"), int(tok2bin(l, rr, nr))))
        return rows

    for ln in lines:
        lnn = _norm_sep(ln)

        if "->" in lnn:
            ap = [p.strip() for p in lnn.split("->") if p.strip()]
            if len(ap) >= 3:
                s = parse_time_token(ap[0], base_date)
                e = parse_time_token(ap[1], base_date)
                y = tok2bin(ap[2], rr, nr)
                if s is not None and e is not None and y is not None:
                    rows.append((s, e, int(y)))
                    continue

            if len(ap) == 2:
                s = parse_time_token(ap[0], base_date)
                e = parse_time_token(ap[1], base_date)
                if s is not None and e is not None:
                    rows.append((s, e, 1))
                    continue

            left, right = lnn.split("->", 1)
            rp = [p.strip() for p in re.split(r"[,\t|;]", right) if p.strip()]
            s = parse_time_token(left.strip(), base_date)
            e = parse_time_token(rp[0], base_date) if rp else None
            y = tok2bin(rp[1], rr, nr) if len(rp) >= 2 else None
            if s is not None and e is not None:
                rows.append((s, e, int(y) if y is not None else 1))
                continue

        parts = [p.strip() for p in re.split(r"[,\t|;]", lnn) if p.strip()]
        if len(parts) >= 3:
            s = parse_time_token(parts[0], base_date)
            e = parse_time_token(parts[1], base_date)
            y = tok2bin(parts[2], rr, nr)
            if s is not None and e is not None and y is not None:
                rows.append((s, e, int(y)))
                continue

        if len(parts) == 2:
            s = parse_time_token(parts[0], base_date)
            e = parse_time_token(parts[1], base_date)
            if s is not None and e is not None:
                rows.append((s, e, 1))
                continue

            y = tok2bin(parts[1], rr, nr)
            if s is not None and y is not None:
                rows.append((s, s + pd.to_timedelta(epoch_s, "s"), int(y)))
                continue

        ws = lnn.split()
        if len(ws) >= 2:
            y = tok2bin(ws[-1], rr, nr)
            if y is not None:
                s = parse_time_token(" ".join(ws[:-1]), base_date)
                if s is not None:
                    rows.append((s, s + pd.to_timedelta(epoch_s, "s"), int(y)))
                    continue

        lnc = _strip(ln)
        if re.fullmatch(r"[A-Za-z0-9_.\- ]+", lnc):
            continue

        raise RuntimeError(f"Could not parse line in {path}: {ln}")

    if not rows:
        raise RuntimeError(f"No rows parsed: {path}")

    rows.sort(key=lambda x: x[0])
    return rows


def build_label_series(index, score_rows):
    y = pd.Series(0.0, index=index, dtype="float32")
    for s, e, label in score_rows:
        mask = (index >= pd.Timestamp(s)) & (index < pd.Timestamp(e))
        if mask.any():
            y.loc[mask] = float(label)
    return y.astype("uint8")


# =========================================================
# Feature engineering
# =========================================================
def ensure_motion_vector(df: pd.DataFrame) -> pd.DataFrame:
    if "Motion_Ang_Vector" in df.columns:
        return df

    lmap = {c.lower(): c for c in df.columns}
    for trip in [
        ("motion_ang_x", "motion_ang_y", "motion_ang_z"),
        ("motion_lin_x", "motion_lin_y", "motion_lin_z"),
    ]:
        cols = [lmap[k] for k in trip if k in lmap]
        if len(cols) == 3:
            out = df.copy()
            out["Motion_Ang_Vector"] = np.sqrt((out[cols].astype(float) ** 2).sum(axis=1))
            return out

    raise RuntimeError("Cannot construct Motion_Ang_Vector from parquet columns.")


def ensure_ois_norm(df: pd.DataFrame) -> pd.DataFrame:
    if "OIS_norm" in df.columns:
        return df

    src = next((c for c in ("IOS", "OIS") if c in df.columns), None)
    if not src:
        raise RuntimeError("Need OIS_norm, IOS, or OIS in parquet.")

    s = pd.to_numeric(df[src], errors="coerce").astype("float32").replace([np.inf, -np.inf], np.nan)
    if s.notna().any():
        lo = np.nanpercentile(s.dropna(), 0.5)
        hi = np.nanpercentile(s.dropna(), 99.5)
    else:
        lo, hi = 0.0, 1.0

    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        lo = float(np.nanmin(s)) if s.notna().any() else 0.0
        hi = float(np.nanmax(s)) if s.notna().any() else 1.0

    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        lo, hi = 0.0, 1.0

    out = df.copy()
    out["OIS_norm"] = ((s - lo) / max(1e-9, hi - lo)).clip(0, 1)
    return out


def prep_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_finite_df(df)
    df = ensure_motion_vector(df)
    df = ensure_ois_norm(df)
    return df


def make_feature_frame(df: pd.DataFrame, shared_base, use_time_features=False, use_derivatives=True):
    feat = df[list(shared_base)].astype("float32").copy()

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

    feat = ensure_finite_df(feat.ffill().bfill().fillna(0.0))
    return feat


# =========================================================
# Sequence helpers
# =========================================================
def contiguous_segments(index, step=1.0):
    if len(index) == 0:
        return []
    diffs = index.to_series().diff().dt.total_seconds().fillna(step).to_numpy()
    breaks = np.where(np.abs(diffs - step) > 1e-6)[0]
    ss, es = np.r_[0, breaks], np.r_[breaks, len(index)]
    return [(int(s), int(e)) for s, e in zip(ss, es) if e > s]


def starts_from_segs(segs, L, stride):
    out = []
    for s, e in segs:
        if e - s >= L:
            out.extend(range(s, e - L + 1, stride))
    return np.asarray(out, dtype=np.int64)


# =========================================================
# Model
# =========================================================
def GroupNorm1d(ch, groups=8):
    groups = min(groups, ch)
    while ch % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, ch)


class MultiKernelStem(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(5, 15, 31)):
        super().__init__()
        if in_ch <= 0:
            raise ValueError("MultiKernelStem requires in_ch > 0")
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

        h = self.ln2(h).transpose(1, 2)
        h = self.conv2(h)
        h = F.silu(self.gn2(h))
        h = self.drop(h)
        h = h.transpose(1, 2)
        return x + h


class EventSupportHead(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2,
                      groups=max(1, d_model // 8), bias=False),
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
    def __init__(self, feature_names, d_model=128, stem_dim=48, blocks=8, kernel_size=7, dropout=0.12):
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
            raise RuntimeError(f"Could not partition features: {self.feature_names}")

        self.register_buffer("motion_idx", torch.tensor(motion_idx, dtype=torch.long), persistent=False)
        self.register_buffer("ois_idx", torch.tensor(ois_idx, dtype=torch.long), persistent=False)
        self.register_buffer("aux_idx", torch.tensor(aux_idx, dtype=torch.long), persistent=False)

        self.motion_stem = MultiKernelStem(len(motion_idx), stem_dim)
        self.ois_stem = MultiKernelStem(len(ois_idx), stem_dim)
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
            xa = x.index_select(dim=2, index=self.aux_idx)
            parts.append(xa)

        h = torch.cat(parts, dim=-1)
        h = self.in_proj(h)
        for blk in self.blocks:
            h = blk(h)

        zf = self.frame_head(h).squeeze(-1)
        ze = self.event_head(h)
        return {"frame": zf, "event": ze}


# =========================================================
# Postprocessing
# =========================================================
def extract_runs(arr):
    arr = np.asarray(arr).astype(np.uint8)
    runs = []
    i = 0
    n = len(arr)
    while i < n:
        if arr[i]:
            j = i + 1
            while j < n and arr[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def merge_runs(runs, gap):
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


def filter_runs_min_len(runs, min_len):
    min_len = int(min_len)
    return [(s, e) for s, e in runs if (e - s) >= min_len]


def runs_to_mask(n, runs):
    m = np.zeros(int(n), dtype=np.uint8)
    for s, e in runs:
        m[int(s):int(e)] = 1
    return m


def hysteresis_decode_runs(probs, start_thr=0.55, stop_thr=0.35):
    probs = np.asarray(probs, dtype=np.float32)
    runs = []
    on = False
    s = None
    for i, p in enumerate(probs):
        if not on:
            if p >= start_thr:
                on = True
                s = i
        else:
            if p < stop_thr:
                runs.append((s, i))
                on = False
                s = None
    if on and s is not None:
        runs.append((s, len(probs)))
    return runs


def decode_postprocessed_runs(probs, start_thr, stop_thr, gap_steps, min_steps):
    pred_runs = hysteresis_decode_runs(probs, start_thr, stop_thr)
    pred_runs = merge_runs(pred_runs, gap_steps)
    pred_runs = filter_runs_min_len(pred_runs, min_steps)
    return pred_runs


# =========================================================
# Inference dataset
# =========================================================
class InferenceDS(Dataset):
    def __init__(self, X: np.ndarray, starts: np.ndarray, L: int):
        self.X = X
        self.starts = np.asarray(starts, dtype=np.int64)
        self.L = int(L)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        xw = self.X[s:s + self.L].copy()
        return xw, s


# =========================================================
# Main inference
# =========================================================
@torch.no_grad()
def run_inference_csv(
    model,
    X: np.ndarray,
    idx: pd.DatetimeIndex,
    starts: np.ndarray,
    win: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
    event_mix: float,
    temperature: float,
    start_thr: float,
    stop_thr: float,
    gap_steps: int,
    min_steps: int,
    y_true: Optional[np.ndarray] = None,
):
    model.eval()

    ds = InferenceDS(X, starts, win)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    rows = []
    combined_probs = []
    frame_probs = []
    event_probs = []
    ts_last = []
    true_last = []

    for xb, s_batch in loader:
        xb = xb.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp and device.type == "cuda")):
            out = model(xb)
            zf = torch.nan_to_num(out["frame"], nan=0.0, posinf=0.0, neginf=0.0)
            ze = torch.nan_to_num(out["event"], nan=0.0, posinf=0.0, neginf=0.0)
            z = zf + float(event_mix) * ze

        pf = torch.sigmoid(zf[:, -1] / temperature).float().cpu().numpy()
        pe = torch.sigmoid(ze[:, -1] / temperature).float().cpu().numpy()
        pc = torch.sigmoid(z[:, -1] / temperature).float().cpu().numpy()

        s_np = s_batch.cpu().numpy().astype(np.int64)
        for s, pcf, pff, pef in zip(s_np, pc, pf, pe):
            last_ix = int(s) + win - 1
            ts_last.append(idx[last_ix])
            combined_probs.append(float(pcf))
            frame_probs.append(float(pff))
            event_probs.append(float(pef))
            if y_true is None:
                true_last.append(-1)
            else:
                true_last.append(int(y_true[last_ix]))

    pred_runs = decode_postprocessed_runs(
        np.asarray(combined_probs, dtype=np.float32),
        start_thr=float(start_thr),
        stop_thr=float(stop_thr),
        gap_steps=int(gap_steps),
        min_steps=int(min_steps),
    )
    pred_mask = runs_to_mask(len(combined_probs), pred_runs)

    for ts, yt, pc, pf, pe, pm in zip(ts_last, true_last, combined_probs, frame_probs, event_probs, pred_mask):
        rows.append({
            "timestamp": ts,
            "true_label": int(yt),
            "pred_prob_rem": float(pc),
            "pred_prob_frame_head": float(pf),
            "pred_prob_event_head": float(pe),
            "pred_label": int(pm),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Inference for best dual TCN REM model on a new parquet window.")
    ap.add_argument("--best_dir", required=True, help="Folder containing model.pt, scaler.pkl, config.json, postprocess.json")
    ap.add_argument("--parquet", required=True, help="Parquet file to run inference on")
    ap.add_argument("--start", required=True, help="Start datetime, e.g. '2023-06-05 00:00:00'")
    ap.add_argument("--end", required=True, help="End datetime, e.g. '2023-06-08 00:00:00'")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--dataset_name", default="TEST", help="Only used for logging")
    ap.add_argument("--manual_scores", default=None, help="Optional manualscore.txt for true_label column")
    ap.add_argument("--base_date", default=None, help="Base datetime for manual_scores")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--expected_step_seconds", type=float, default=1.0)
    ap.add_argument("--score_epoch_seconds", type=int, default=30)
    ap.add_argument("--amp", action="store_true", help="Use CUDA BF16 autocast")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    args = ap.parse_args()

    best_dir = Path(args.best_dir)
    model_path = best_dir / "model.pt"
    scaler_path = best_dir / "scaler.pkl"
    config_path = best_dir / "config.json"
    post_path = best_dir / "postprocess.json"

    for p in [model_path, scaler_path, config_path, post_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    banner(f"Loading artifacts from {best_dir}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(post_path, "r", encoding="utf-8") as f:
        post = json.load(f)

    feature_names = list(config["feature_names"])
    shared_base = list(config.get("shared_base_features", ["Motion_Ang_Vector", "OIS_norm"]))
    win = int(config["win"])
    use_time_features = bool(config.get("use_time_features", False))
    use_derivative_features = bool(config.get("use_derivative_features", True))

    d_model = int(config["d_model"])
    stem_dim = int(config["stem_dim"])
    blocks = int(config["blocks"])
    kernel_size = int(config["kernel_size"])
    dropout = float(config["dropout"])

    temperature = float(post["temperature"])
    event_mix = float(post["event_mix"])
    start_thr = float(post["start_threshold"])
    stop_thr = float(post["stop_threshold"])
    gap_steps = max(0, int(round(float(post["gap_seconds"]) / args.expected_step_seconds)))
    min_steps = max(1, int(round(float(post["min_episode_seconds"]) / args.expected_step_seconds)))

    device = get_device(force_cpu=args.cpu)
    print(f"Device: {device}")

    scaler = joblib_load(scaler_path)

    model = DualBranchEventTCN(
        feature_names=feature_names,
        d_model=d_model,
        stem_dim=stem_dim,
        blocks=blocks,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    banner(f"Reading parquet: {args.parquet}")
    df_raw = read_parquet(args.parquet)
    df_raw = slice_df(df_raw, args.start, args.end)
    if df_raw.empty:
        raise RuntimeError("No parquet rows remain after slicing.")

    df_raw = prep_raw_df(df_raw)
    feat = make_feature_frame(
        df_raw,
        shared_base=shared_base,
        use_time_features=use_time_features,
        use_derivatives=use_derivative_features,
    )

    missing = [c for c in feature_names if c not in feat.columns]
    if missing:
        raise RuntimeError(f"Missing required model features: {missing}")

    feat = feat[feature_names].astype("float32")
    feat = feat.loc[~feat.index.duplicated(keep="first")].sort_index()

    if len(feat) < win:
        raise RuntimeError(f"Not enough rows after slicing for win={win}. Rows available: {len(feat)}")

    y_true = None
    if args.manual_scores:
        if not args.base_date:
            raise RuntimeError("--base_date is required when --manual_scores is supplied")
        banner(f"Parsing manual scores: {args.manual_scores}")
        score_rows = parse_score_lines(
            args.manual_scores,
            base_date=args.base_date,
            epoch_s=args.score_epoch_seconds,
        )
        y_ser = build_label_series(feat.index, score_rows)
        y_true = y_ser.to_numpy(np.uint8)
    else:
        print("No manual_scores supplied; true_label will be -1 in the CSV.")

    X = scaler.transform(feat).astype("float32")
    X = ensure_finite_np(X, "INFER_X")

    segs = contiguous_segments(feat.index, step=args.expected_step_seconds)
    starts = starts_from_segs(segs, win, stride=1)
    if len(starts) == 0:
        raise RuntimeError("No valid contiguous windows found for inference.")

    banner(f"Running inference on {args.dataset_name}")
    pred_df = run_inference_csv(
        model=model,
        X=X,
        idx=feat.index,
        starts=starts,
        win=win,
        batch_size=args.batch_size,
        device=device,
        amp=bool(args.amp),
        event_mix=event_mix,
        temperature=temperature,
        start_thr=start_thr,
        stop_thr=stop_thr,
        gap_steps=gap_steps,
        min_steps=min_steps,
        y_true=y_true,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_csv, index=False)

    banner("Done")
    print(f"Saved: {out_csv}")
    print(f"Rows:  {len(pred_df):,}")
    print(f"First: {pred_df['timestamp'].iloc[0]}")
    print(f"Last:  {pred_df['timestamp'].iloc[-1]}")
    print(
        "Columns:",
        list(pred_df.columns),
    )


if __name__ == "__main__":
    main()