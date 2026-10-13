
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_rem_tcn_event_v9.py

Mission-aligned REM episode detector for mice:
- Input features are built from Motion_Ang_Vector + OIS_norm (optionally derivatives/time)
- Dual-branch multi-scale TCN
- Two heads: frame head + event-support head
- Hysteresis decoding
- Merge predicted gaps within 5 s (configurable)
- Drop predicted runs shorter than 30 s (configurable)
- One-to-one event matching
- Optuna objective = 0.8 * event-Fbeta + 0.2 * GT coverage
- Saves every completed trial + test CSV + best folder
- CLI kept intentionally close to the prior v8 script
"""

import os, gc, re, json, math, time, shutil, warnings, argparse
from pathlib import Path
from contextlib import contextmanager

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dateutil import parser as dtparser

import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, NopPruner

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from joblib import dump

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

@contextmanager
def section(name):
    tqdm.write(f"\n▶ {name} [{_now()}]")
    try:
        yield
        tqdm.write(f"✔ {name}")
    except Exception as e:
        tqdm.write(f"✖ {name} FAILED: {e}")
        raise

def banner(msg):
    bar = "═" * len(msg)
    tqdm.write(f"\n{bar}\n{msg}\n{bar}\n")


# ═══════════════════════════════════════════════════════════════════════════
# CUDA diagnostic
# ═══════════════════════════════════════════════════════════════════════════

def check_cuda():
    banner("CUDA / GPU Diagnostic")
    if not torch.cuda.is_available():
        tqdm.write("WARNING: torch.cuda.is_available() = FALSE")
        tqdm.write("Training will run on CPU — expect it to be very slow")
        tqdm.write("Fix: restart container with --gpus all")
        return torch.device("cpu")
    dev = torch.device("cuda")
    tqdm.write(f"✔ GPU    : {torch.cuda.get_device_name(0)}")
    tqdm.write(f"✔ VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    tqdm.write(f"✔ CUDA   : {torch.version.cuda}")
    tqdm.write(f"✔ BF16   : {torch.cuda.is_bf16_supported()}")
    try:
        x = torch.randn(4, 64, device=dev, dtype=torch.bfloat16)
        y = x @ x.T
        tqdm.write(f"✔ BF16 matmul OK (norm={y.norm().item():.3f})")
    except Exception as e:
        tqdm.write(f"✖ BF16 matmul FAILED: {e}")
    return dev


# ═══════════════════════════════════════════════════════════════════════════
# System helpers
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def clean_cols(cols):
    return [re.sub(r"[^\w]+", "_", str(c)).strip("_") for c in cols]

def ensure_finite_df(df, name=""):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_categorical_dtype(df[c]):
            try:
                df[c] = df[c].astype("float32")
            except Exception:
                df[c] = df[c].astype(str)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)
    if df.isna().any().any():
        df = df.ffill().bfill()
        num = df.select_dtypes(include=[np.number, "bool"]).columns
        df[num] = df[num].fillna(0.0)
        for c in [c for c in df.columns if c not in num]:
            df[c] = df[c].fillna("")
    return df

def ensure_finite_np(x, name=""):
    if not np.isfinite(x).all():
        tqdm.write(f"WARN {name}: fixing {(~np.isfinite(x)).sum()} non-finite values")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


# ═══════════════════════════════════════════════════════════════════════════
# Time / label parsing
# ═══════════════════════════════════════════════════════════════════════════

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
    return y

def parse_ignore_ranges(path, base_date=None):
    ranges = []
    with open(str(path), "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith(("#", "//")):
                continue
            s = _norm_sep(s)
            if "->" in s:
                a, b = s.split("->", 1)
                t1 = parse_time_token(a.strip(), base_date)
                t2 = parse_time_token(b.strip(), base_date)
                if t1 is not None and t2 is not None:
                    ranges.append((t1, t2))
                    continue
            parts = [p.strip() for p in re.split(r"[,\t|;]", s) if p.strip()]
            if len(parts) >= 2:
                t1 = parse_time_token(parts[0], base_date)
                t2 = parse_time_token(parts[1], base_date)
                if t1 is not None and t2 is not None:
                    ranges.append((t1, t2))
                    continue
            raise RuntimeError(f"Could not parse ignore line: {ln.rstrip()}")
    if not ranges:
        raise RuntimeError(f"No ranges parsed: {path}")
    return ranges

def apply_ignore(df, ranges):
    idx = df.index
    keep = np.ones(len(idx), dtype=bool)
    for s, e in ranges:
        keep &= ~((idx >= pd.Timestamp(s)) & (idx < pd.Timestamp(e)))
    out = df.loc[keep]
    tqdm.write(f"  ignore: kept {len(out):,}/{len(df):,}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Feature engineering
# ═══════════════════════════════════════════════════════════════════════════

BASE_FEATURE_CANDIDATES = ["Motion_Ang_Vector", "OIS_norm"]

def ensure_motion_vector(df):
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
    raise RuntimeError("Cannot construct Motion_Ang_Vector")

def ensure_ois_norm(df):
    if "OIS_norm" in df.columns:
        return df
    src = next((c for c in ("IOS", "OIS") if c in df.columns), None)
    if not src:
        raise RuntimeError("Need OIS_norm/IOS/OIS")
    s = pd.to_numeric(df[src], errors="coerce").astype("float32").replace([np.inf, -np.inf], np.nan)
    lo = np.nanpercentile(s.dropna(), 0.5) if s.notna().any() else 0.0
    hi = np.nanpercentile(s.dropna(), 99.5) if s.notna().any() else 1.0
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        lo = float(np.nanmin(s)) if s.notna().any() else 0.0
        hi = float(np.nanmax(s)) if s.notna().any() else 1.0
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        lo, hi = 0.0, 1.0
    out = df.copy()
    out["OIS_norm"] = ((s - lo) / max(1e-9, hi - lo)).clip(0, 1)
    return out

def read_parquet(path):
    df = pd.read_parquet(str(path))
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["t", "time", "timestamp", "datetime", "date", "index", "Timestamp"]:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col).sort_index()
                    break
                except Exception:
                    pass
        else:
            raise RuntimeError(f"Need DatetimeIndex: {path}")
    df = df.copy()
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    df.columns = clean_cols(df.columns)
    return df

def slice_df(df, start=None, end=None):
    if start:
        df = df.loc[pd.Timestamp(start):]
    if end:
        df = df.loc[:pd.Timestamp(end)]
    return df

def prep_raw_df(df, name):
    df = ensure_finite_df(df, name)
    df = ensure_motion_vector(df)
    df = ensure_ois_norm(df)
    return df

def find_shared_base_features(raw_dfs):
    shared = [f for f in BASE_FEATURE_CANDIDATES if all(f in df.columns for df in raw_dfs)]
    if "Motion_Ang_Vector" not in shared or "OIS_norm" not in shared:
        raise RuntimeError(f"Insufficient shared features: {shared}")
    return shared

def make_feature_frame(df, shared_base, use_time_features=False, use_derivatives=True):
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

def load_animal_unscaled(name, parquet_path, score_path, windows, shared_base,
                         base_date=None, epoch_s=30, ignore_path=None,
                         rem_re=None, nonrem_re=None,
                         use_time_features=False, use_derivatives=True):
    rr = re.compile(rem_re) if rem_re else REM_RE
    nr = re.compile(nonrem_re) if nonrem_re else NONREM_RE

    with section(f"Load animal: {name}"):
        df_raw = read_parquet(parquet_path)
        tqdm.write(f"  raw rows = {len(df_raw):,}")
        pieces = []
        for ws, we in windows:
            p = slice_df(df_raw, ws, we)
            tqdm.write(f"  window {ws} -> {we}: {len(p):,} rows")
            pieces.append(p)
        df = pd.concat(pieces).sort_index()
        tqdm.write(f"  combined: {len(df):,} rows")
        if ignore_path:
            df = apply_ignore(df, parse_ignore_ranges(ignore_path, base_date))
        df = prep_raw_df(df, name)
        feat = make_feature_frame(df, shared_base, use_time_features, use_derivatives)
        score_rows = parse_score_lines(score_path, base_date, epoch_s, rr, nr)
        y = build_label_series(feat.index, score_rows)
        if feat.index.has_duplicates:
            keep = ~feat.index.duplicated(keep="first")
            feat, y = feat.loc[keep], y.loc[keep]
        tqdm.write(f"  rows={len(feat):,}  REM={int((y==1).sum()):,}  ({100*float((y==1).mean()):.1f}%)")
        if len(feat) == 0:
            raise RuntimeError(f"No rows: {name}")
    return feat, y.astype("uint8"), feat.index


# ═══════════════════════════════════════════════════════════════════════════
# Sequence construction
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Augmentation
# ═══════════════════════════════════════════════════════════════════════════

class TemporalAug:
    def __init__(self, time_mask=0.08, amp_jitter=0.10, p=0.5):
        self.time_mask = time_mask
        self.amp_jitter = amp_jitter
        self.p = p

    def __call__(self, x):
        if np.random.rand() > self.p:
            return x
        x = x.copy()
        T, C = x.shape
        ml = int(T * self.time_mask * np.random.rand())
        if ml > 0:
            st = np.random.randint(0, max(1, T - ml))
            x[st:st + ml] = x[max(0, st - 1):max(0, st - 1) + 1]
        x *= 1.0 + self.amp_jitter * (np.random.rand(C) * 2 - 1)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class Seq2SeqDS(Dataset):
    def __init__(self, X, y, starts, L, aug=None):
        self.X = X
        self.y = y.astype(np.int8)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.L = int(L)
        self.aug = aug

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        xw = self.X[s:s + self.L].copy()
        yw = self.y[s:s + self.L].copy()
        if self.aug:
            xw = self.aug(xw)
        return xw, yw.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

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
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=max(1, d_model // 8), bias=False),
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
        self.event_head = EventSupportHead(d_model, kernel_size=max(15, kernel_size * 4 + 3), dropout=dropout)

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


# ═══════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.35, beta=0.65, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        t = target.float()
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1.0 - score

class DualHeadEventLoss(nn.Module):
    def __init__(self, pos_weight, bce_w=0.55, tv_w=0.45, event_head_w=0.55, consistency_w=0.05):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.tv = TverskyLoss(alpha=0.35, beta=0.65)
        self.bce_w = bce_w
        self.tv_w = tv_w
        self.event_head_w = event_head_w
        self.consistency_w = consistency_w

    def _one(self, logits, target):
        return self.bce_w * self.bce(logits, target) + self.tv_w * self.tv(logits, target)

    def forward(self, out, target):
        lf = self._one(out["frame"], target)
        le = self._one(out["event"], target)
        cons = F.l1_loss(torch.sigmoid(out["frame"]), torch.sigmoid(out["event"]))
        return lf + self.event_head_w * le + self.consistency_w * cons


# ═══════════════════════════════════════════════════════════════════════════
# Event decoding / metrics
# ═══════════════════════════════════════════════════════════════════════════

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

def event_match_metrics(gt_runs, pred_runs, min_overlap_steps=30, beta=2.0):
    edges = []
    for gi, (gs, ge) in enumerate(gt_runs):
        for pi, (ps, pe) in enumerate(pred_runs):
            ov = max(0, min(ge, pe) - max(gs, ps))
            if ov >= min_overlap_steps:
                edges.append((ov, gi, pi))

    edges.sort(reverse=True, key=lambda x: (x[0], -x[1], -x[2]))
    used_g, used_p = set(), set()
    matches = []
    for ov, gi, pi in edges:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, ov))

    tp = len(matches)
    fn = len(gt_runs) - tp
    fp = len(pred_runs) - tp

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    beta2 = beta * beta
    fbeta = (1 + beta2) * prec * rec / max(1e-9, beta2 * prec + rec)

    coverages = []
    purities = []
    for gi, pi, ov in matches:
        gs, ge = gt_runs[gi]
        ps, pe = pred_runs[pi]
        coverages.append(ov / max(1, ge - gs))
        purities.append(ov / max(1, pe - ps))

    out = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(prec),
        "recall": float(rec),
        "fbeta": float(fbeta),
        "coverage": float(np.mean(coverages) if coverages else 0.0),
        "purity": float(np.mean(purities) if purities else 0.0),
        "matches": matches,
    }
    return out

def score_event_predictions(y_true, probs, start_thr, stop_thr, gap_steps, min_steps,
                            min_overlap_steps, beta=2.0):
    y_true = np.asarray(y_true).astype(np.uint8)
    probs = np.asarray(probs).astype(np.float32)

    gt_runs = merge_runs(extract_runs(y_true), gap_steps)
    pred_runs = decode_postprocessed_runs(probs, start_thr, stop_thr, gap_steps, min_steps)

    met = event_match_metrics(gt_runs, pred_runs, min_overlap_steps=min_overlap_steps, beta=beta)
    pred_mask = runs_to_mask(len(y_true), pred_runs)
    frame_f1 = f1_score(y_true, pred_mask, zero_division=0) if len(np.unique(y_true)) > 1 else float((pred_mask == y_true).mean())
    score = 0.8 * met["fbeta"] + 0.2 * met["coverage"]

    met.update({
        "frame_f1": float(frame_f1),
        "score": float(score),
        "gt_runs": gt_runs,
        "pred_runs": pred_runs,
        "pred_mask": pred_mask,
    })
    return met

def search_best_hysteresis(y_true, probs, gap_steps, min_steps, min_overlap_steps, beta=2.0,
                           start_grid=None, stop_grid=None):
    if start_grid is None:
        start_grid = np.linspace(0.45, 0.75, 7)
    if stop_grid is None:
        stop_grid = np.linspace(0.15, 0.55, 9)

    best = None
    best_key = None
    for st in start_grid:
        for sp in stop_grid:
            if sp >= st - 1e-9:
                continue
            met = score_event_predictions(
                y_true, probs, float(st), float(sp),
                gap_steps=gap_steps, min_steps=min_steps,
                min_overlap_steps=min_overlap_steps, beta=beta
            )
            key = (met["score"], met["fbeta"], met["coverage"], met["purity"], met["frame_f1"])
            if best is None or key > best_key:
                best = met
                best["start_thr"] = float(st)
                best["stop_thr"] = float(sp)
                best_key = key
    return best

def learn_temp(logits, labels, steps=200, lr=0.05):
    class T(nn.Module):
        def __init__(self):
            super().__init__()
            self.lt = nn.Parameter(torch.zeros(()))

        def forward(self, z):
            return z / torch.exp(self.lt)

        @property
        def T(self):
            return float(torch.exp(self.lt).detach())

    m = T()
    opt = torch.optim.LBFGS(m.parameters(), lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(
            torch.nan_to_num(m(logits), nan=0.0, posinf=0.0, neginf=0.0),
            labels.float()
        )
        loss.backward()
        return loss

    opt.step(cl)
    return m.T

@torch.no_grad()
def eval_seq2seq(model, loader, device, amp=True, event_mix=0.6):
    model.eval()
    lz, lzf, lze, ly = [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp and device.type == "cuda")):
            out = model(xb)
            zf = torch.nan_to_num(out["frame"], nan=0.0, posinf=0.0, neginf=0.0)
            ze = torch.nan_to_num(out["event"], nan=0.0, posinf=0.0, neginf=0.0)
            z = zf + float(event_mix) * ze
        lz.append(z[:, -1].detach().cpu())
        lzf.append(zf[:, -1].detach().cpu())
        lze.append(ze[:, -1].detach().cpu())
        ly.append(yb[:, -1].detach().cpu())
    return torch.cat(lz), torch.cat(lzf), torch.cat(lze), torch.cat(ly)


# ═══════════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════════

def build_all_matrices(args):
    banner("Loading 3 training animals + eval/test animals")

    og_raw = prep_raw_df(
        slice_df(read_parquet(args.og_parquet), args.og_window1_start, args.og_window3_end),
        "OG_PRE"
    )
    ca_raw = prep_raw_df(
        slice_df(read_parquet(args.caff_a_parquet), args.caff_a_start, args.caff_a_end),
        "CA_PRE"
    )
    cb_raw = prep_raw_df(
        slice_df(read_parquet(args.caff_b_parquet), args.caff_b_start, args.caff_b_end),
        "CB_PRE"
    )
    ev_raw = prep_raw_df(
        slice_df(read_parquet(args.eval_parquet), args.eval_start, args.eval_end),
        "EV_PRE"
    )
    te_raw = prep_raw_df(
        slice_df(read_parquet(args.test_parquet), args.test_start, args.test_end),
        "TEST_PRE"
    )

    shared = find_shared_base_features([og_raw, ca_raw, cb_raw, ev_raw, te_raw])
    tqdm.write(f"Shared base features: {shared}")

    feat_og, y_og, idx_og = load_animal_unscaled(
        "OG", args.og_parquet, args.og_scores,
        [
            (args.og_window1_start, args.og_window1_end),
            (args.og_window2_start, args.og_window2_end),
            (args.og_window3_start, args.og_window3_end),
        ],
        shared, args.og_base_date, args.score_epoch_seconds, args.og_ignore_ranges,
        args.rem_label_regex, args.nonrem_label_regex,
        args.use_time_features, not args.disable_derivative_features
    )
    feat_ca, y_ca, idx_ca = load_animal_unscaled(
        "CAFF_A", args.caff_a_parquet, args.caff_a_scores,
        [(args.caff_a_start, args.caff_a_end)],
        shared, args.caff_a_base_date, args.score_epoch_seconds, args.caff_a_ignore_ranges,
        args.rem_label_regex, args.nonrem_label_regex,
        args.use_time_features, not args.disable_derivative_features
    )
    feat_cb, y_cb, idx_cb = load_animal_unscaled(
        "CAFF_B", args.caff_b_parquet, args.caff_b_scores,
        [(args.caff_b_start, args.caff_b_end)],
        shared, args.caff_b_base_date, args.score_epoch_seconds, args.caff_b_ignore_ranges,
        args.rem_label_regex, args.nonrem_label_regex,
        args.use_time_features, not args.disable_derivative_features
    )
    feat_ev, y_ev, idx_ev = load_animal_unscaled(
        "EVAL", args.eval_parquet, args.eval_scores,
        [(args.eval_start, args.eval_end)],
        shared, args.eval_base_date, args.score_epoch_seconds, args.eval_ignore_ranges,
        args.rem_label_regex, args.nonrem_label_regex,
        args.use_time_features, not args.disable_derivative_features
    )
    feat_te, y_te, idx_te = load_animal_unscaled(
        "TEST", args.test_parquet, args.test_scores,
        [(args.test_start, args.test_end)],
        shared, args.test_base_date, args.score_epoch_seconds, args.test_ignore_ranges,
        args.rem_label_regex, args.nonrem_label_regex,
        args.use_time_features, not args.disable_derivative_features
    )

    feature_names = list(feat_og.columns)
    for nm, feat in [("CAFF_A", feat_ca), ("CAFF_B", feat_cb), ("EVAL", feat_ev), ("TEST", feat_te)]:
        if list(feat.columns) != feature_names:
            raise RuntimeError(f"Feature mismatch in {nm}: {list(feat.columns)} vs {feature_names}")

    train_fit = pd.concat([feat_og, feat_ca, feat_cb], axis=0)
    scaler = StandardScaler()
    scaler.fit(train_fit.to_numpy(np.float32))

    X_og = ensure_finite_np(scaler.transform(feat_og.to_numpy(np.float32)).astype("float32"), "OG")
    X_ca = ensure_finite_np(scaler.transform(feat_ca.to_numpy(np.float32)).astype("float32"), "CAFF_A")
    X_cb = ensure_finite_np(scaler.transform(feat_cb.to_numpy(np.float32)).astype("float32"), "CAFF_B")
    X_ev = ensure_finite_np(scaler.transform(feat_ev.to_numpy(np.float32)).astype("float32"), "EVAL")
    X_te = ensure_finite_np(scaler.transform(feat_te.to_numpy(np.float32)).astype("float32"), "TEST")

    n_og = len(X_og)
    n_ca = len(X_ca)
    n_cb = len(X_cb)

    X_train = np.concatenate([X_og, X_ca, X_cb], axis=0)
    y_train = np.concatenate([y_og.to_numpy(np.uint8), y_ca.to_numpy(np.uint8), y_cb.to_numpy(np.uint8)], axis=0)

    tqdm.write(
        f"Training: OG={n_og:,} CAFF_A={n_ca:,} CAFF_B={n_cb:,} "
        f"Total={len(X_train):,} REM={int((y_train==1).sum()):,} ({100*float((y_train==1).mean()):.1f}%)"
    )

    step = args.expected_step_seconds
    off_ca = n_og
    off_cb = n_og + n_ca

    og_segs = contiguous_segments(idx_og, step)
    ca_segs = contiguous_segments(idx_ca, step)
    cb_segs = contiguous_segments(idx_cb, step)
    ev_segs = contiguous_segments(idx_ev, step)
    te_segs = contiguous_segments(idx_te, step)

    og_dense = starts_from_segs(og_segs, args.win, 1)
    ca_dense = starts_from_segs(ca_segs, args.win, 1) + off_ca
    cb_dense = starts_from_segs(cb_segs, args.win, 1) + off_cb
    ev_st = starts_from_segs(ev_segs, args.win, 1)
    te_st = starts_from_segs(te_segs, args.win, 1)

    og_train = starts_from_segs(og_segs, args.win, args.stride)
    ca_train = starts_from_segs(ca_segs, args.win, args.stride) + off_ca
    cb_train = starts_from_segs(cb_segs, args.win, args.stride) + off_cb

    def split_train_cal(train_starts, dense_starts):
        if len(dense_starts) == 0:
            raise RuntimeError("No dense starts available for calibration split.")
        n_cal = max(1, int(len(dense_starts) * args.cal_fraction))
        cal_starts = dense_starts[-n_cal:]
        cal_min = int(cal_starts[0])
        train_keep = train_starts[train_starts + args.win - 1 < cal_min]
        if len(train_keep) == 0:
            raise RuntimeError("Calibration split consumed all training starts; reduce cal_fraction or window size.")
        return train_keep, cal_starts

    tr_og_t, ca_og = split_train_cal(og_train, og_dense)
    tr_ca_t, ca_ca = split_train_cal(ca_train, ca_dense)
    tr_cb_t, ca_cb = split_train_cal(cb_train, cb_dense)

    tr_starts = np.concatenate([tr_og_t, tr_ca_t, tr_cb_t])
    ca_starts = np.concatenate([ca_og, ca_ca, ca_cb])

    tqdm.write(
        f"Seqs: tr_og={len(tr_og_t):,} tr_ca={len(tr_ca_t):,} tr_cb={len(tr_cb_t):,} "
        f"cal={len(ca_starts):,} eval={len(ev_st):,} test={len(te_st):,}"
    )

    n1, n2, n3 = len(tr_og_t), len(tr_ca_t), len(tr_cb_t)
    total = n1 + n2 + n3
    w1 = total / (3 * max(1, n1))
    w2 = total / (3 * max(1, n2))
    w3 = total / (3 * max(1, n3))
    sw = np.concatenate([np.full(n1, w1), np.full(n2, w2), np.full(n3, w3)])
    tqdm.write(f"Sampler weights: OG={w1:.3f} CAFF_A={w2:.3f} CAFF_B={w3:.3f}")

    rem_frac = float((y_train[tr_starts + args.win - 1] == 1).mean())
    base_pw = min(8.0, (1 - rem_frac) / max(1e-6, rem_frac))
    tqdm.write(f"REM frac={rem_frac:.3f} base_pos_weight={base_pw:.2f}")

    return {
        "n_features": X_train.shape[1],
        "feature_names": feature_names,
        "shared_base": shared,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_eval": X_ev,
        "y_eval": y_ev.to_numpy(np.uint8),
        "idx_eval": idx_ev,
        "X_test": X_te,
        "y_test": y_te.to_numpy(np.uint8),
        "idx_test": idx_te,
        "tr_starts": tr_starts,
        "ca_starts": ca_starts,
        "ev_starts": ev_st,
        "te_starts": te_st,
        "sample_weights": sw,
        "base_pw": base_pw,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DataLoaders
# ═══════════════════════════════════════════════════════════════════════════

def make_loaders(prep, args, batch, aug=None):
    nw = int(args.num_workers)
    kw = dict(num_workers=nw, pin_memory=False)
    if nw > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = args.prefetch_factor

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(prep["sample_weights"].astype(np.float32)),
        num_samples=len(prep["tr_starts"]),
        replacement=True
    )

    dl_tr = DataLoader(
        Seq2SeqDS(prep["X_train"], prep["y_train"], prep["tr_starts"], args.win, aug),
        batch_size=batch, sampler=sampler, drop_last=True, **kw
    )
    dl_ca = DataLoader(
        Seq2SeqDS(prep["X_train"], prep["y_train"], prep["ca_starts"], args.win),
        batch_size=batch, shuffle=False, drop_last=False, **kw
    )
    dl_ev = DataLoader(
        Seq2SeqDS(prep["X_eval"], prep["y_eval"], prep["ev_starts"], args.win),
        batch_size=batch, shuffle=False, drop_last=False, **kw
    )
    dl_te = DataLoader(
        Seq2SeqDS(prep["X_test"], prep["y_test"], prep["te_starts"], args.win),
        batch_size=batch, shuffle=False, drop_last=False, **kw
    )
    return dl_tr, dl_ca, dl_ev, dl_te


# ═══════════════════════════════════════════════════════════════════════════
# Test CSV inference
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_seq2seq_inference_csv(model, X, y, idx, starts, win, batch_size, device, amp,
                              event_mix, temperature, start_thr, stop_thr, gap_steps, min_steps):
    model.eval()
    rows = []

    ds = Seq2SeqDS(X, y, starts, win, aug=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    combined_probs = []
    frame_probs = []
    event_probs = []
    true_last = []
    ts_last = []

    cursor = 0
    for xb, yb in loader:
        xb = xb.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(amp and device.type == "cuda")):
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

        for s, pcf, pff, pef, yt in zip(batch_starts, pc, pf, pe, yb[:, -1].cpu().numpy().astype(np.uint8)):
            ts_last.append(idx[int(s) + win - 1])
            combined_probs.append(float(pcf))
            frame_probs.append(float(pff))
            event_probs.append(float(pef))
            true_last.append(int(yt))

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


# ═══════════════════════════════════════════════════════════════════════════
# Objective
# ═══════════════════════════════════════════════════════════════════════════

def make_objective(args, prep, outdir, device):
    use_amp = bool(args.amp) and device.type == "cuda"
    in_ch = prep["n_features"]
    min_steps = max(1, int(round(args.episode_min_seconds / args.expected_step_seconds)))
    gap_steps = max(0, int(round(args.episode_merge_gap / args.expected_step_seconds)))
    min_overlap_steps = max(1, int(round(args.event_match_min_overlap_seconds / args.expected_step_seconds)))
    aug = TemporalAug(args.aug_time_mask, args.aug_amplitude_jitter, args.aug_p) if args.augment else None

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [96, 128, 160])
        stem_dim = trial.suggest_categorical("stem_dim", [32, 48, 64])
        blocks = trial.suggest_int("blocks", 5, 9)
        kernel_size = trial.suggest_categorical("kernel_size", [5, 7, 9])
        dropout = trial.suggest_float("dropout", 0.08, 0.25)
        lr = trial.suggest_float("lr", 5e-5, 2e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 2e-2, log=True)
        bs = trial.suggest_categorical("batch", [64, 96, 128, 192, 256])
        pw_s = trial.suggest_float("pos_weight_scale", 0.6, 1.4)
        event_mix = trial.suggest_float("event_mix", 0.25, 1.0)
        event_head_w = trial.suggest_float("event_head_w", 0.35, 0.80)

        banner(
            f"TRIAL {trial.number} | device={device} d={d_model} stem={stem_dim} "
            f"blocks={blocks} k={kernel_size} dr={dropout:.2f} lr={lr:.1e} "
            f"bs={bs} pw*{pw_s:.2f} mix={event_mix:.2f}"
        )

        dl_tr, dl_ca, dl_ev, dl_te = make_loaders(prep, args, bs, aug)
        model = DualBranchEventTCN(
            prep["feature_names"],
            d_model=d_model,
            stem_dim=stem_dim,
            blocks=blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        ).to(device)

        if args.compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                tqdm.write("  torch.compile(reduce-overhead) active")
            except Exception as e:
                tqdm.write(f"  WARN compile skipped: {e}")

        pw = torch.tensor([prep["base_pw"] * pw_s], device=device, dtype=torch.float32)
        crit = DualHeadEventLoss(pw, event_head_w=event_head_w)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=lr * 0.05)

        best_score, best_state, no_imp = -1.0, None, 0

        try:
            for ep in range(1, args.epochs + 1):
                model.train()
                run = 0.0
                pb = tqdm(dl_tr, desc=f"T{trial.number} E{ep}/{args.epochs}", leave=False)
                for xb, yb in pb:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        out = model(xb)
                        loss = crit(out, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
                    run += float(loss.detach())
                    pb.set_postfix_str(f"loss={run / max(1, pb.n):.4f}")
                sch.step()

                zc, _, _, yc = eval_seq2seq(model, dl_ca, device, use_amp, event_mix=event_mix)
                pc = torch.sigmoid(zc).float().numpy()
                yc_np = yc.numpy().astype(np.uint8)

                cal_best = search_best_hysteresis(
                    yc_np, pc, gap_steps=gap_steps, min_steps=min_steps,
                    min_overlap_steps=min_overlap_steps, beta=args.event_beta
                )

                tqdm.write(
                    f"[T{trial.number}] ep{ep}: cal score={cal_best['score']:.4f} "
                    f"F{args.event_beta:.1f}={cal_best['fbeta']:.4f} cov={cal_best['coverage']:.4f} "
                    f"frameF1={cal_best['frame_f1']:.4f} start={cal_best['start_thr']:.3f} stop={cal_best['stop_thr']:.3f}"
                )

                trial.report(cal_best["score"], ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if cal_best["score"] > best_score + args.min_delta:
                    best_score = cal_best["score"]
                    no_imp = 0
                    _m = getattr(model, "_orig_mod", model)
                    best_state = {k: v.detach().cpu() for k, v in _m.state_dict().items()}
                else:
                    no_imp += 1
                    if no_imp >= args.patience:
                        tqdm.write(f"[T{trial.number}] early stop ep{ep}")
                        break

            if best_state is None:
                _m = getattr(model, "_orig_mod", model)
                best_state = {k: v.detach().cpu() for k, v in _m.state_dict().items()}

            _m = getattr(model, "_orig_mod", model)
            _m.load_state_dict(best_state)

            zc, _, _, yc = eval_seq2seq(model, dl_ca, device, use_amp, event_mix=event_mix)
            yc_np = yc.numpy().astype(np.uint8)
            T_cal = learn_temp(zc.cpu(), yc.cpu())
            pc_t = torch.sigmoid(zc / T_cal).float().numpy()

            cal_final = search_best_hysteresis(
                yc_np, pc_t, gap_steps=gap_steps, min_steps=min_steps,
                min_overlap_steps=min_overlap_steps, beta=args.event_beta
            )

            zev, _, _, yev = eval_seq2seq(model, dl_ev, device, use_amp, event_mix=event_mix)
            pev = torch.sigmoid(zev / T_cal).float().numpy()
            yev_np = yev.numpy().astype(np.uint8)
            ev_met = score_event_predictions(
                yev_np, pev, cal_final["start_thr"], cal_final["stop_thr"],
                gap_steps=gap_steps, min_steps=min_steps,
                min_overlap_steps=min_overlap_steps, beta=args.event_beta
            )

            zte, _, _, yte = eval_seq2seq(model, dl_te, device, use_amp, event_mix=event_mix)
            pte = torch.sigmoid(zte / T_cal).float().numpy()
            yte_np = yte.numpy().astype(np.uint8)
            te_met = score_event_predictions(
                yte_np, pte, cal_final["start_thr"], cal_final["stop_thr"],
                gap_steps=gap_steps, min_steps=min_steps,
                min_overlap_steps=min_overlap_steps, beta=args.event_beta
            )

            tqdm.write(
                f"[T{trial.number}] EVAL score={ev_met['score']:.4f} F{args.event_beta:.1f}={ev_met['fbeta']:.4f} "
                f"P={ev_met['precision']:.4f} R={ev_met['recall']:.4f} "
                f"cov={ev_met['coverage']:.4f} purity={ev_met['purity']:.4f} "
                f"TP={ev_met['tp']} FP={ev_met['fp']} FN={ev_met['fn']} | frame-F1={ev_met['frame_f1']:.4f}"
            )
            tqdm.write(
                f"[T{trial.number}] TEST score={te_met['score']:.4f} F{args.event_beta:.1f}={te_met['fbeta']:.4f} "
                f"P={te_met['precision']:.4f} R={te_met['recall']:.4f} "
                f"cov={te_met['coverage']:.4f} purity={te_met['purity']:.4f} "
                f"TP={te_met['tp']} FP={te_met['fp']} FN={te_met['fn']} | frame-F1={te_met['frame_f1']:.4f} "
                f"T={T_cal:.4f} start={cal_final['start_thr']:.4f} stop={cal_final['stop_thr']:.4f}"
            )

            td = outdir / f"trial_{trial.number:04d}"
            td.mkdir(parents=True, exist_ok=True)

            pred_df_test = run_seq2seq_inference_csv(
                model=getattr(model, "_orig_mod", model),
                X=prep["X_test"],
                y=prep["y_test"],
                idx=prep["idx_test"],
                starts=prep["te_starts"],
                win=args.win,
                batch_size=bs,
                device=device,
                amp=use_amp,
                event_mix=event_mix,
                temperature=T_cal,
                start_thr=cal_final["start_thr"],
                stop_thr=cal_final["stop_thr"],
                gap_steps=gap_steps,
                min_steps=min_steps,
            )
            pred_df_test.to_csv(td / "test_predictions_COMBINED_CAFFEINE.csv", index=False)

            metrics_payload = {
                "objective_score": float(ev_met["score"]),
                f"eval_event_f{args.event_beta:.1f}": float(ev_met["fbeta"]),
                "eval_precision": float(ev_met["precision"]),
                "eval_recall": float(ev_met["recall"]),
                "eval_coverage": float(ev_met["coverage"]),
                "eval_purity": float(ev_met["purity"]),
                "eval_tp": int(ev_met["tp"]),
                "eval_fp": int(ev_met["fp"]),
                "eval_fn": int(ev_met["fn"]),
                "eval_frame_f1": float(ev_met["frame_f1"]),

                "test_name": "COMBINED_CAFFEINE",
                f"test_event_f{args.event_beta:.1f}": float(te_met["fbeta"]),
                "test_precision": float(te_met["precision"]),
                "test_recall": float(te_met["recall"]),
                "test_coverage": float(te_met["coverage"]),
                "test_purity": float(te_met["purity"]),
                "test_tp": int(te_met["tp"]),
                "test_fp": int(te_met["fp"]),
                "test_fn": int(te_met["fn"]),
                "test_frame_f1": float(te_met["frame_f1"]),

                "temperature": float(T_cal),
                "start_threshold": float(cal_final["start_thr"]),
                "stop_threshold": float(cal_final["stop_thr"]),
                "event_mix": float(event_mix),
                "shared_base_features": prep["shared_base"],
                "feature_names": prep["feature_names"],
            }

            with open(td / "metrics.json", "w") as f:
                json.dump(metrics_payload, f, indent=2)

            _ms = getattr(model, "_orig_mod", model)
            torch.save({k: v.cpu() for k, v in _ms.state_dict().items()}, td / "model.pt")
            dump(prep["scaler"], td / "scaler.pkl", compress=3)

            with open(td / "postprocess.json", "w") as f:
                json.dump(
                    {
                        "temperature": float(T_cal),
                        "event_mix": float(event_mix),
                        "start_threshold": float(cal_final["start_thr"]),
                        "stop_threshold": float(cal_final["stop_thr"]),
                        "gap_seconds": float(args.episode_merge_gap),
                        "min_episode_seconds": float(args.episode_min_seconds),
                        "event_match_min_overlap_seconds": float(args.event_match_min_overlap_seconds),
                    },
                    f, indent=2
                )
            with open(td / "config.json", "w") as f:
                json.dump(
                    {
                        "arch": "dual_tcn_event_v9",
                        "d_model": d_model,
                        "stem_dim": stem_dim,
                        "blocks": blocks,
                        "kernel_size": kernel_size,
                        "dropout": dropout,
                        "win": args.win,
                        "stride": args.stride,
                        "use_time_features": bool(args.use_time_features),
                        "use_derivative_features": bool(not args.disable_derivative_features),
                        "test_dataset_name": "COMBINED_CAFFEINE",
                        "shared_base_features": prep["shared_base"],
                        "feature_names": prep["feature_names"],
                    },
                    f, indent=2
                )

            return float(ev_met["score"])

        finally:
            del model, dl_tr, dl_ca, dl_ev, dl_te
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return objective


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def add_ds(ap, prefix, required=True):
    ap.add_argument(f"--{prefix}_parquet", required=required)
    ap.add_argument(f"--{prefix}_scores", required=required)
    ap.add_argument(f"--{prefix}_start", default=None)
    ap.add_argument(f"--{prefix}_end", default=None)
    ap.add_argument(f"--{prefix}_base_date", default=None)
    ap.add_argument(f"--{prefix}_ignore_ranges", default=None)

def main():
    ap = argparse.ArgumentParser(description="3-animal mice REM TCN v9 — event-aware, duration-aware")

    ap.add_argument("--og_parquet", required=True)
    ap.add_argument("--og_scores", required=True)
    ap.add_argument("--og_base_date", default=None)
    ap.add_argument("--og_ignore_ranges", default=None)
    ap.add_argument("--og_window1_start", required=True)
    ap.add_argument("--og_window1_end", required=True)
    ap.add_argument("--og_window2_start", required=True)
    ap.add_argument("--og_window2_end", required=True)
    ap.add_argument("--og_window3_start", required=True)
    ap.add_argument("--og_window3_end", required=True)

    add_ds(ap, "caff_a", required=True)
    add_ds(ap, "caff_b", required=True)
    add_ds(ap, "eval", required=True)
    add_ds(ap, "test", required=True)

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arch", default="dual_tcn")
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--min_delta", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--win", type=int, default=600)
    ap.add_argument("--stride", type=int, default=120)
    ap.add_argument("--cal_fraction", type=float, default=0.15)
    ap.add_argument("--episode_min_seconds", type=float, default=30.0)
    ap.add_argument("--episode_merge_gap", type=float, default=5.0)
    ap.add_argument("--event_match_min_overlap_seconds", type=float, default=30.0)
    ap.add_argument("--event_beta", type=float, default=2.0)
    ap.add_argument("--score_epoch_seconds", type=int, default=30)
    ap.add_argument("--expected_step_seconds", type=float, default=1.0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--aug_time_mask", type=float, default=0.08)
    ap.add_argument("--aug_amplitude_jitter", type=float, default=0.10)
    ap.add_argument("--aug_p", type=float, default=0.5)
    ap.add_argument("--use_time_features", action="store_true")
    ap.add_argument("--disable_derivative_features", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--pruner", choices=["median", "asha", "none"], default="asha")
    ap.add_argument("--rem_label_regex", default=None)
    ap.add_argument("--nonrem_label_regex", default=None)

    args = ap.parse_args()
    set_seed(args.seed)
    device = check_cuda()

    if args.arch != "dual_tcn":
        tqdm.write(f"WARN: --arch {args.arch} supplied; using dual_tcn event model instead")

    banner("3-Animal REM TCN v9 — event-aware episode detector")
    tqdm.write(f"Device       : {device}")
    tqdm.write(f"AMP(BF16)    : {args.amp and device.type=='cuda'}")
    tqdm.write(f"Compile      : {args.compile}")
    tqdm.write(f"Win/Stride   : {args.win}/{args.stride}")
    tqdm.write(f"Trials/Epochs: {args.trials}/{args.epochs}")
    tqdm.write(f"Episode rules: merge_gap={args.episode_merge_gap}s min_len={args.episode_min_seconds}s "
               f"min_overlap={args.event_match_min_overlap_seconds}s Fbeta={args.event_beta}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prep = build_all_matrices(args)

    pruner = {
        "median": MedianPruner(n_startup_trials=6, n_warmup_steps=3),
        "asha": SuccessiveHalvingPruner(min_resource=3, reduction_factor=3),
        "none": NopPruner(),
    }[args.pruner]

    study = optuna.create_study(
        direction="maximize",
        storage=f"sqlite:///{(outdir / 'study.db').resolve()}",
        load_if_exists=True,
        study_name="rem_dual_tcn_event_v9",
        pruner=pruner
    )

    study.optimize(
        make_objective(args, prep, outdir, device),
        n_trials=args.trials,
        show_progress_bar=True
    )

    try:
        study.trials_dataframe(attrs=("number", "value", "state", "params")).to_csv(outdir / "trials.csv", index=False)
    except Exception as e:
        tqdm.write(f"WARN trials.csv: {e}")

    try:
        best = study.best_trial.number
        dst = outdir / "best"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

        for f in [
            "model.pt",
            "scaler.pkl",
            "postprocess.json",
            "config.json",
            "metrics.json",
            "test_predictions_COMBINED_CAFFEINE.csv",
        ]:
            p = outdir / f"trial_{best:04d}" / f
            if p.exists():
                shutil.copy2(p, dst / f)

        banner("Tuning complete")
        print("Best objective score:", study.best_value)
        print("Best params:", study.best_params)
        print(f"Artifacts: {outdir}/best/")
    except Exception as e:
        tqdm.write(f"ERROR copying best: {e}")

if __name__ == "__main__":
    main()
