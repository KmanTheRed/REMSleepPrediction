#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tune_precision_ios_f1_restricted_ext_eval_fixed.py

Robust hyperparameter tuner (Optuna) for REM detection with:
- OG restricted to GOOD_TRAIN_RANGES
- NEW restricted strictly before first caffeine > 0.0
- Train objective: BCE + SoftF1, early stop on Cal F1 (beta=1)
- Select best by F1 on external eval parquet
- Defensive NaN/Inf handling throughout to avoid sklearn crashes
"""

import os, gc, re, json, argparse, warnings, shutil, time
warnings.filterwarnings("ignore")

from pathlib import Path
from contextlib import contextmanager
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, NopPruner

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------- Pretty logging ----------
def _now(): return time.strftime("%Y-%m-%d %H:%M:%S")

@contextmanager
def section(name: str):
    tqdm.write(f"\n▶ {name} [{_now()}]")
    try:
        yield
        tqdm.write(f"✔ {name}")
    except Exception as e:
        tqdm.write(f"✖ {name} FAILED: {e}")
        raise

def banner(msg: str):
    bar = "═" * len(msg)
    tqdm.write(f"\n{bar}\n{msg}\n{bar}\n")


def set_seed(seed=42):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def clean_cols(cols):
    return [re.sub(r"[^\w]+", "_", c).strip("_") for c in cols]


# ---------- NaN/Inf defenses ----------
def ensure_finite_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Replace ±Inf->NaN, then ffill/bfill, then 0. Logs counts."""
    bad = (~np.isfinite(df.select_dtypes(include=[np.number]))).sum().sum()
    if bad:
        tqdm.write(f"WARN: {name}: found {int(bad)} non-finite values; cleaning.")
    df = df.replace([np.inf, -np.inf], np.nan)
    if df.isna().any().any():
        df = df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return df

def ensure_finite_np(x: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(x).all():
        numbad = (~np.isfinite(x)).sum()
        tqdm.write(f"WARN: {name}: {int(numbad)} non-finite entries; replacing.")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

def sanitize_probs(p: np.ndarray, label: str) -> np.ndarray:
    if not np.isfinite(p).all():
        numbad = (~np.isfinite(p)).sum()
        tqdm.write(f"WARN: {label}: {int(numbad)} non-finite probs; replacing to 0.5.")
        p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    return p


# ---------- Data restrictions ----------
GOOD_TRAIN_RANGES = [
    ("2022-10-04", "2022-11-11"),
    ("2022-11-19", "2022-12-14"),
    ("2022-12-22", "2023-01-04"),
    ("2023-01-08", "2023-01-24"),
]

def apply_good_ranges(df: pd.DataFrame, ranges) -> pd.DataFrame:
    idx = df.index
    mask = np.zeros(len(idx), dtype=bool)
    for start_str, end_str in ranges:
        s = pd.to_datetime(start_str)
        e = pd.to_datetime(end_str) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        mask |= (idx >= s) & (idx <= e)
    kept = df.loc[mask]
    tqdm.write(f"apply_good_ranges: kept {len(kept):,} / {len(df):,} rows across {len(ranges)} ranges.")
    if kept.empty:
        raise RuntimeError("apply_good_ranges produced empty dataframe. Check date bounds.")
    return kept

def find_caffeine_column(df: pd.DataFrame):
    cands = [c for c in df.columns if "caff" in c.lower() or c.lower()=="caffeine" or "coffee" in c.lower()]
    if not cands: return None
    for c in cands:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            return c
    return cands[0]

def first_caffeine_time_strict(df: pd.DataFrame):
    col = find_caffeine_column(df)
    if col is None:
        return None, None
    s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    m = s > 0.0
    if not m.any(): return col, None
    first_idx = df.index[np.argmax(m.values)]
    return col, pd.Timestamp(first_idx)

def restrict_before_first_caffeine(df: pd.DataFrame, require=True) -> pd.DataFrame:
    col, t = first_caffeine_time_strict(df)
    if col is None:
        msg = "No caffeine-like column found."
        if require: raise RuntimeError(msg)
        tqdm.write("WARN: " + msg + " Proceeding without caffeine trim.")
        return df
    if t is None:
        msg = f"Caffeine column '{col}' has no values > 0.0."
        if require: raise RuntimeError(msg)
        tqdm.write("WARN: " + msg + " Proceeding without caffeine trim.")
        return df
    trimmed = df.loc[df.index < t]
    tqdm.write(f"restrict_before_first_caffeine: using '{col}', cut at {t}. kept {len(trimmed):,} / {len(df):,} rows.")
    if trimmed.empty:
        raise RuntimeError("Caffeine trim produced empty dataframe.")
    return trimmed


# ---------- Labeling & features ----------
def label_with_sleep_detect(df: pd.DataFrame,
                            ios_col_guess=("OIS_norm","IOS","OIS"),
                            motion_vec_name="Motion_Ang_Vector",
                            awake_threshold=0.50,
                            sleep_window="2.5T") -> pd.DataFrame:
    df = df.copy()

    # motion vector
    if motion_vec_name not in df.columns:
        xyz = [c for c in df.columns if c.lower() in ("motion_ang_x","motion_ang_y","motion_ang_z")]
        if len(xyz) == 3:
            df[motion_vec_name] = np.sqrt((df[xyz]**2).sum(axis=1))
        else:
            lxyz = [c for c in df.columns if c.lower() in ("motion_lin_x","motion_lin_y","motion_lin_z")]
            if len(lxyz) == 3:
                df[motion_vec_name] = np.sqrt((df[lxyz]**2).sum(axis=1))
            else:
                raise RuntimeError("No Motion_Ang_Vector or XYZ motion columns found.")

    # IOS_norm
    ios_col = None
    for c in ios_col_guess:
        if c in df.columns:
            ios_col = c; break
    if ios_col is None:
        raise RuntimeError("No IOS/OIS column found (OIS_norm/IOS/OIS).")
    if ios_col != "OIS_norm":
        s = pd.to_numeric(df[ios_col], errors="coerce").astype("float32")
        lo, hi = np.nanpercentile(s.dropna(), [0.5, 99.5]) if s.notna().any() else (0.0, 1.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = s.min(skipna=True), s.max(skipna=True)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0
        s = (s - lo) / max(1e-9, (hi - lo))
        s = s.clip(0, 1)
        df["OIS_norm"] = s

    try:
        from sleep_detect import extract_awake_rem_nrem
    except Exception as e:
        raise RuntimeError("sleep_detect not found. Ensure it's importable.") from e

    out = extract_awake_rem_nrem(
        df, bf_par="CBV", awake_par=motion_vec_name,
        awake_threshold=awake_threshold, sleep_window=sleep_window,
        return_timedelta=True,
    ).copy()
    out["y_rem"] = (out["state"] == "REM").astype("uint8")
    out["is_sleep"] = out["state"].isin(["NREM", "REM"]).astype("uint8")
    return out


def add_temporal_context(feat_df: pd.DataFrame,
                         y_rem: pd.Series,
                         sleep_mask: pd.Series) -> pd.DataFrame:
    idx = feat_df.index
    y = y_rem.reindex(idx).fillna(0).to_numpy(dtype=np.uint8)
    sleep = sleep_mask.reindex(idx).fillna(False).to_numpy(dtype=bool)

    tslr = np.zeros_like(y, dtype=np.int32)
    run = 0
    for i in range(len(y)):
        if i > 0 and y[i-1] == 0 and y[i] == 1:
            run = 0
        else:
            run += 1
        tslr[i] = run

    tsso = np.zeros_like(y, dtype=np.int32)
    run = 0
    for i in range(len(y)):
        if i > 0 and (not sleep[i-1]) and sleep[i]:
            run = 0
        else:
            run = (run + 1) if sleep[i] else 0
        tsso[i] = run

    feat_df = feat_df.copy()
    feat_df["time_sin"] = np.sin(2*np.pi*feat_df.index.hour/24).astype("float32")
    feat_df["time_cos"] = np.cos(2*np.pi*feat_df.index.hour/24).astype("float32")
    feat_df["ts_last_rem"] = np.log1p(np.clip(tslr, 0, 90*60)).astype("float32")
    feat_df["ts_sleep_onset"] = np.log1p(np.clip(tsso, 0, 120*60)).astype("float32")
    return feat_df


def pick_feature_columns(df: pd.DataFrame):
    feats = []
    if "Motion_Ang_Vector" in df.columns:
        feats.append("Motion_Ang_Vector")
    else:
        raise RuntimeError("Motion_Ang_Vector missing after preprocessing.")
    if "HR" in df.columns:
        feats.append("HR")
    if "OIS_norm" in df.columns:
        feats.append("OIS_norm")
    elif "IOS" in df.columns:
        feats.append("IOS")
    else:
        raise RuntimeError("Expected OIS_norm or IOS in dataframe.")
    return feats


# ---------- Datasets & Model ----------
class SeqDataset(Dataset):
    def __init__(self, X, y, starts, L):
        self.X = X; self.y = y; self.starts = starts.astype(np.int64); self.L = int(L)
    def __len__(self): return len(self.starts)
    def __getitem__(self, i):
        s = self.starts[i]; e = s + self.L
        return self.X[s:e], self.y[e-1]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10_000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe)
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class TinyTransformer(nn.Module):
    def __init__(self, in_ch, d_model=96, nhead=4, ff=192, layers=3, dropout=0.1):
        super().__init__()
        self.prj = nn.Linear(in_ch, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                               batch_first=True, activation="gelu",
                                               norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.prj(x); x = self.pos(x)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        h = self.enc(torch.cat([cls, x], dim=1))
        z = self.head(self.norm(h[:,0])).squeeze(-1)
        return z

class SoftF1Loss(nn.Module):
    def __init__(self, beta: float = 1.0, eps: float = 1e-7):
        super().__init__(); self.beta2 = beta*beta; self.eps = eps
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        tp = (p*targets).sum()
        fp = (p*(1-targets)).sum()
        fn = ((1-p)*targets).sum()
        num = (1 + self.beta2) * tp
        den = (1 + self.beta2) * tp + self.beta2*fn + fp + self.eps
        return 1.0 - (num / den)


# ---------- Helpers ----------
def evaluate_logits(model, loader, device, desc="Eval"):
    model.eval()
    Lz, Ly = [], []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=desc, leave=False):
            xb = xb.to(device); yb = yb.to(device).float()
            z = model(xb)
            if not torch.isfinite(z).all():
                tqdm.write("WARN: non-finite logits encountered; replacing with zeros.")
                z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            Lz.append(z.detach().cpu()); Ly.append(yb.detach().cpu())
    return torch.cat(Lz), torch.cat(Ly)

def build_split_indices(n_total, split1, split2, L, stride):
    tr = np.arange(0, max(0, split1 - L + 1), stride, dtype=np.int64)
    ca = np.arange(split1, max(split1, split2 - L + 1), stride, dtype=np.int64)
    te = np.arange(split2, max(split2, n_total - L + 1), stride, dtype=np.int64)
    if len(tr) == 0 or len(ca) == 0 or len(te) == 0:
        raise RuntimeError(f"Empty split: tr={len(tr)}, ca={len(ca)}, te={len(te)}. Increase data or adjust --win.")
    return tr, ca, te

def choose_threshold_fbeta(y_true, p, beta=1.0):
    p = sanitize_probs(p, "p_cal")
    if (y_true.sum() == 0) or (y_true.sum() == len(y_true)):
        # degenerate case: pick 0.5 and compute F1 safely
        thr = 0.5
        yhat = (p >= thr).astype(np.uint8)
        prec = precision_score(y_true, yhat, zero_division=0)
        rec = recall_score(y_true, yhat, zero_division=0)
        f = f1_score(y_true, yhat, zero_division=0) if beta == 1.0 else (1+beta**2)*(prec*rec)/(beta**2*prec+rec+1e-9)
        return float(thr), float(prec), float(rec), float(f)
    prec, rec, thr = precision_recall_curve(y_true, p)
    thr = np.r_[0.0, thr]
    f = (1+beta**2) * (prec*rec) / (beta**2*prec + rec + 1e-9)
    i = np.nanargmax(f)
    return float(thr[i]), float(prec[i]), float(rec[i]), float(f[i])

def learn_temperature(logits, labels, max_steps=200, lr=0.05):
    class _Temp(nn.Module):
        def __init__(self): super().__init__(); self.logT = nn.Parameter(torch.zeros(()))
        def forward(self, z): return z / torch.exp(self.logT)
        @property
        def T(self): return float(torch.exp(self.logT).detach().cpu())
    m = _Temp(); opt = torch.optim.LBFGS(m.parameters(), lr=lr, max_iter=max_steps, line_search_fn="strong_wolfe")
    labels = labels.float()
    def closure():
        opt.zero_grad(set_to_none=True)
        z = m(logits)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        loss = F.binary_cross_entropy_with_logits(z, labels)
        loss.backward(); return loss
    opt.step(closure)
    return m.T


# ---------- Dataset builders ----------
def build_train_matrix(og_path, new_path, win, stride):
    with section("Load + restrict OG (GOOD_TRAIN_RANGES)"):
        df1 = pd.read_parquet(og_path)
        if not isinstance(df1.index, pd.DatetimeIndex):
            raise ValueError("OG parquet must have DatetimeIndex")
        df1 = df1.copy(); df1.index = df1.index.tz_localize(None).sort_values()
        df1.columns = clean_cols(df1.columns)
        tqdm.write(f"OG total rows: {len(df1):,}")
        df1 = apply_good_ranges(df1, GOOD_TRAIN_RANGES)
        df1 = ensure_finite_df(df1, "OG raw")
        df1 = label_with_sleep_detect(df1)

    with section("Load + trim NEW (before first caffeine>0)"):
        df2 = pd.read_parquet(new_path)
        if not isinstance(df2.index, pd.DatetimeIndex):
            raise ValueError("NEW parquet must have DatetimeIndex")
        df2 = df2.copy(); df2.index = df2.index.tz_localize(None).sort_values()
        df2.columns = clean_cols(df2.columns)
        tqdm.write(f"NEW total rows: {len(df2):,}")
        df2 = restrict_before_first_caffeine(df2, require=True)
        df2 = ensure_finite_df(df2, "NEW raw")
        df2 = label_with_sleep_detect(df2)

    with section("Merge sources"):
        common = sorted(set(df1.columns).intersection(set(df2.columns)))
        df = pd.concat([df1[common], df2[common]], axis=0).sort_index()
        tqdm.write(f"Merged rows: {len(df):,}")

    with section("Build features"):
        feat_cols = pick_feature_columns(df)
        tqdm.write(f"Base features: {feat_cols}")
        feat_df = df[feat_cols].astype("float32").ffill().bfill()
        feat_df = ensure_finite_df(feat_df, "features base")
        y_rem = df["y_rem"].astype("uint8")
        sleep_mask = df["is_sleep"].astype("bool")
        feat_df = add_temporal_context(feat_df, y_rem, sleep_mask)
        feat_df = ensure_finite_df(feat_df, "features + temporal")
        X = feat_df.to_numpy(np.float32)
        y = y_rem.reindex(feat_df.index).to_numpy(np.uint8)
        X = ensure_finite_np(X, "X_train")
        N, C = X.shape
        tqdm.write(f"Matrix: N={N:,}, C={C}")
        if N < win + 100:
            raise RuntimeError(f"Too few rows ({N}) for window {win}.")
    split1, split2 = int(0.60*N), int(0.80*N)
    tr_st, ca_st, te_st = build_split_indices(N, split1, split2, win, stride)
    tqdm.write(f"Starts -> Train={len(tr_st):,}  Cal={len(ca_st):,}  Test={len(te_st):,}")
    return X, y, feat_cols, tr_st, ca_st, te_st, split1, split2

def build_eval_matrix(eval_path, feat_names_expected):
    with section("Load EVAL parquet"):
        df = pd.read_parquet(eval_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Eval parquet must have DatetimeIndex")
        df = df.copy(); df.index = df.index.tz_localize(None).sort_values()
        df.columns = clean_cols(df.columns)
        tqdm.write(f"Eval rows: {len(df):,}")
        df = ensure_finite_df(df, "EVAL raw")
        df = label_with_sleep_detect(df)

    with section("Eval -> features"):
        base_cols = pick_feature_columns(df)
        feat_df = df[base_cols].astype("float32").ffill().bfill()
        feat_df = ensure_finite_df(feat_df, "EVAL features base")
        y_rem = df["y_rem"].astype("uint8")
        sleep_mask = df["is_sleep"].astype("bool")
        feat_df = add_temporal_context(feat_df, y_rem, sleep_mask)
        feat_df = ensure_finite_df(feat_df, "EVAL + temporal")
        missing = [c for c in feat_names_expected if c not in feat_df.columns]
        if missing:
            raise RuntimeError(f"Eval missing required feature columns: {missing}")
        feat_df = feat_df[feat_names_expected]
        X_eval = feat_df.to_numpy(np.float32)
        X_eval = ensure_finite_np(X_eval, "X_eval")
        y_eval = y_rem.reindex(feat_df.index).to_numpy(np.uint8)
        tqdm.write(f"Eval matrix: N={len(X_eval):,}, C={X_eval.shape[1]}")
        if len(X_eval) < 100:
            tqdm.write("WARN: Eval set is quite small; metrics may be noisy.")
    return X_eval, y_eval


# ---------- Optuna objective ----------
def make_objective(args, out_base):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    banner("Building TRAIN matrices")
    X, y, feat_cols, tr_st, ca_st, te_st, split1, split2 = build_train_matrix(
        args.og, args.new, args.win, args.stride
    )
    scaler = StandardScaler()
    scaler.fit(X[:split1])
    X = scaler.transform(X).astype("float32")
    X = ensure_finite_np(X, "X_train_scaled")

    feat_names_expected = list(feat_cols) + ["time_sin","time_cos","ts_last_rem","ts_sleep_onset"]
    banner("Building EVAL matrix")
    X_eval_raw, y_eval = build_eval_matrix(args.eval, feat_names_expected)

    def make_dls(batch):
        ds_tr = SeqDataset(X, y, tr_st, args.win)
        ds_ca = SeqDataset(X, y, ca_st, args.win)
        ds_te = SeqDataset(X, y, te_st, args.win)
        dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True,  num_workers=0, pin_memory=True, drop_last=True)
        dl_ca = DataLoader(ds_ca, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)
        dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)
        return dl_tr, dl_ca, dl_te

    def objective(trial: optuna.trial.Trial):
        try:
            d_model = trial.suggest_categorical("d_model", [64, 96, 128, 160])
            nhead_choices = [h for h in [2,4,8] if d_model % h == 0]
            nhead = trial.suggest_categorical("nhead", nhead_choices)
            layers = trial.suggest_int("layers", 2, 5)
            ff_mult = trial.suggest_int("ff_mult", 2, 6)
            ff_dim = ff_mult * d_model
            dropout = trial.suggest_float("dropout", 0.0, 0.3)
            lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-2, log=True)
            batch = trial.suggest_categorical("batch", [128, 192, 256])
            f1_lambda = trial.suggest_float("f1_lambda", 0.3, 0.7)
            beta = 1.0

            banner(f"TRIAL {trial.number}  d_model={d_model} nhead={nhead} layers={layers} ff={ff_dim} drop={dropout:.2f}  "
                   f"lr={lr:.2e} wd={weight_decay:.2e} batch={batch} f1_lambda={f1_lambda:.2f}")

            dl_tr, dl_ca, dl_te = make_dls(batch)

            model = TinyTransformer(in_ch=len(feat_cols)+4, d_model=d_model, nhead=nhead,
                                    ff=ff_dim, layers=layers, dropout=dropout).to(device)
            bce = nn.BCEWithLogitsLoss()
            softf1 = SoftF1Loss(beta=beta)
            def criterion(logits, targets):
                return f1_lambda * bce(logits, targets) + (1 - f1_lambda) * softf1(logits, targets)

            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

            best_cal_f, best_state = -1.0, None
            no_imp = 0

            for epoch in range(1, args.epochs+1):
                model.train(); run = 0.0
                pbar = tqdm(dl_tr, desc=f"[Trial {trial.number}] Epoch {epoch}/{args.epochs}", leave=False)
                for xb, yb in pbar:
                    xb = xb.to(device); yb = yb.to(device).float()
                    opt.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                        z = model(xb)
                        loss = criterion(z, yb)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler_amp.step(opt); scaler_amp.update()
                    run += float(loss)
                    pbar.set_postfix_str(f"loss={run/(pbar.n+1e-9):.4f}")
                sched.step()

                # Cal evaluation (pre-temperature)
                zc, yc_t = evaluate_logits(model, dl_ca, device, "Cal")
                zc = torch.nan_to_num(zc, nan=0.0, posinf=0.0, neginf=0.0)
                p_cal_raw = torch.sigmoid(zc).numpy()
                p_cal_raw = sanitize_probs(p_cal_raw, "p_cal_raw")
                y_cal = yc_t.numpy().astype(np.uint8)

                thr_ep, P_c, R_c, F_c = choose_threshold_fbeta(y_cal, p_cal_raw, beta=beta)
                tqdm.write(f"[Trial {trial.number}] Epoch {epoch}: Cal F1={F_c:.4f}  (P={P_c:.3f}, R={R_c:.3f}, thr={thr_ep:.3f})")

                trial.report(F_c, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if F_c > best_cal_f + args.min_delta:
                    best_cal_f = F_c
                    best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
                    no_imp = 0
                else:
                    no_imp += 1
                    if no_imp >= args.patience:
                        tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch}.")
                        break

            if best_state is None:
                best_state = model.state_dict()
            model.load_state_dict(best_state)

            # Temperature scaling + final Cal threshold
            zc, yc_t = evaluate_logits(model, dl_ca, device, "Cal logits")
            zc = torch.nan_to_num(zc, nan=0.0, posinf=0.0, neginf=0.0)
            T = learn_temperature(zc.cpu(), yc_t.cpu(), max_steps=200, lr=0.05)
            p_cal = 1.0/(1.0 + np.exp(-(zc.cpu().numpy()/T)))
            p_cal = sanitize_probs(p_cal, "p_cal_postT")
            y_cal = yc_t.cpu().numpy().astype(np.uint8)
            thr_final, P_thr, R_thr, F_thr = choose_threshold_fbeta(y_cal, p_cal, beta=beta)
            tqdm.write(f"[Trial {trial.number}] Post-T Cal: T={T:.3f} thr={thr_final:.3f} F1={F_thr:.4f}")

            # External EVAL metrics (selection)
            X_eval = scaler.transform(X_eval_raw).astype("float32")
            X_eval = ensure_finite_np(X_eval, "X_eval_scaled")
            starts_eval = np.arange(0, max(0, len(X_eval) - args.win + 1), 1, dtype=np.int64)
            if len(starts_eval) == 0:
                raise RuntimeError("Eval set too short for given window.")
            dl_eval = DataLoader(SeqDataset(X_eval, y_eval, starts_eval, args.win),
                                 batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)

            ze, ye_t = evaluate_logits(model, dl_eval, device, "Eval logits")
            ze = torch.nan_to_num(ze, nan=0.0, posinf=0.0, neginf=0.0)
            p_eval = 1.0/(1.0 + np.exp(-(ze.cpu().numpy()/T)))
            p_eval = sanitize_probs(p_eval, "p_eval")
            y_eval_np = ye_t.cpu().numpy().astype(np.uint8)
            y_pred = (p_eval >= thr_final).astype(np.uint8)

            acc = accuracy_score(y_eval_np, y_pred)
            prec = precision_score(y_eval_np, y_pred, zero_division=0)
            rec = recall_score(y_eval_np, y_pred, zero_division=0)
            f1 = f1_score(y_eval_np, y_pred, zero_division=0)
            tqdm.write(f"[Trial {trial.number}] EVAL: Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

            # Optional internal test metrics
            zt, yt_t = evaluate_logits(model, dl_te, device, "Test logits")
            zt = torch.nan_to_num(zt, nan=0.0, posinf=0.0, neginf=0.0)
            p_test = 1.0/(1.0 + np.exp(-(zt.cpu().numpy()/T)))
            p_test = sanitize_probs(p_test, "p_test")
            y_test = yt_t.cpu().numpy().astype(np.uint8)
            y_test_hat = (p_test >= thr_final).astype(np.uint8)
            test_auc = roc_auc_score(y_test, p_test)
            test_pr = average_precision_score(y_test, p_test)
            test_f1 = f1_score(y_test, y_test_hat, zero_division=0)

            # Persist trial artifacts
            trial_dir = Path(out_base, f"trial_{trial.number:04d}")
            trial_dir.mkdir(parents=True, exist_ok=True)
            meta = dict(
                d_model=d_model, nhead=nhead, layers=layers, ff_mult=ff_mult, ff_dim=ff_dim, dropout=dropout,
                lr=lr, weight_decay=weight_decay, batch=batch, f1_lambda=f1_lambda,
                win=args.win, stride=args.stride, epochs=args.epochs, patience=args.patience,
                cal_best_F1=float(best_cal_f), cal_T=float(T), cal_thr=float(thr_final),
                eval_accuracy=float(acc), eval_precision=float(prec), eval_recall=float(rec), eval_f1=float(f1),
                test_auc=float(test_auc), test_pr_auc=float(test_pr), test_f1=float(test_f1),
            )
            json.dump(meta, open(trial_dir/"metrics.json","w"), indent=2)

            torch.save({k: v.cpu() for k,v in model.state_dict().items()}, trial_dir/"best_tx_stage.pt")
            from joblib import dump
            dump(scaler, trial_dir/"tx_stage_scaler.pkl", compress=3)
            json.dump({"temperature": float(T)}, open(trial_dir/"tx_stage_temperature.json","w"))
            cfg = dict(
                og=str(Path(args.og).resolve()),
                new=str(Path(args.new).resolve()),
                eval=str(Path(args.eval).resolve()),
                outdir=str(trial_dir.resolve()),
                win=args.win, stride=args.stride, epochs=args.epochs, batch=batch, seed=args.seed,
                d_model=d_model, heads=nhead, layers=layers, ff_dim=ff_dim, dropout=dropout,
                lr=lr, weight_decay=weight_decay, patience=args.patience, grad_clip=args.grad_clip,
                loss="bce+softf1", f1_lambda=f1_lambda, f_beta=1.0,
                temperature=float(T), threshold=float(thr_final),
                features=feat_cols + ["time_sin","time_cos","ts_last_rem","ts_sleep_onset"],
                objective="Select by F1 on external eval dataset",
                restrictions=dict(
                    og_good_train_ranges=GOOD_TRAIN_RANGES,
                    new_trim="before first caffeine > 0.0"
                )
            )
            json.dump(cfg, open(trial_dir/"tx_stage_config.json","w"), indent=2)

            # cleanup
            del model, dl_tr, dl_ca, dl_te, dl_eval
            torch.cuda.empty_cache(); gc.collect()

            return float(f1)

        except optuna.TrialPruned:
            torch.cuda.empty_cache(); gc.collect()
            raise
        except Exception as e:
            torch.cuda.empty_cache(); gc.collect()
            raise e

    return objective


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--og", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--win", type=int, default=120)
    ap.add_argument("--stride", type=int, default=1)

    ap.add_argument("--pruner", choices=["median","asha","none"], default="median")
    args = ap.parse_args()

    banner("F1 tuner starting")
    tqdm.write(f"OG      : {args.og}")
    tqdm.write(f"NEW     : {args.new}  (trim before first caffeine>0)")
    tqdm.write(f"EVAL    : {args.eval}  (selection by F1)")
    tqdm.write(f"OUTDIR  : {args.outdir}")
    tqdm.write(f"GPU     : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    tqdm.write(f"Win/Str : {args.win}/{args.stride}  Epochs: {args.epochs}  Trials: {args.trials}")
    tqdm.write(f"EarlyStop: patience={args.patience}, min_delta={args.min_delta}")

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    study_db = f"sqlite:///{(out/'study.db').resolve()}"

    if args.pruner == "median":
        pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=3)
    elif args.pruner == "asha":
        pruner = SuccessiveHalvingPruner(min_resource=3, reduction_factor=3)
    else:
        pruner = NopPruner()

    study = optuna.create_study(
        direction="maximize", storage=study_db, load_if_exists=True,
        study_name="precision_ios_f1_ext_eval", pruner=pruner
    )
    objective = make_objective(args, out)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # dump results
    try:
        df = study.trials_dataframe(attrs=("number","value","state","params","user_attrs"))
        df.to_csv(out/"trials.csv", index=False)
    except Exception as e:
        tqdm.write(f"WARN: Could not write trials.csv ({e})")

    # materialize best/
    try:
        best_num = study.best_trial.number
        src = Path(out, f"trial_{best_num:04d}")
        dst = Path(out, "best")
        if dst.exists(): shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        for fname in ["best_tx_stage.pt","tx_stage_scaler.pkl","tx_stage_temperature.json","tx_stage_config.json","metrics.json"]:
            p = src / fname
            if p.exists(): shutil.copy2(p, dst / fname)
        banner("Tuning finished")
        print("Best Eval F1:", study.best_value)
        print("Best params :", study.best_params)
        print(f"Artifacts  :\n - {out/'study.db'}\n - {out/'trials.csv'}\n - {out/'best'}")
    except Exception as e:
        tqdm.write(f"ERROR while copying best artifacts: {e}")
        banner("Tuning finished (no best/ copy created)")

if __name__ == "__main__":
    main()
 