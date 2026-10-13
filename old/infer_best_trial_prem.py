#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
infer_best_trial_prem.py

- Loads the BEST Optuna trial from a tuning outdir (using best/ if present,
  otherwise reading study.db).
- Rebuilds the TinyTransformer with those hyperparameters.
- Runs REM probability inference on a given parquet file.
- Saves a CSV with columns: time, p_rem   (1 Hz, aligned to last step of each window).
- Prints the recommended Panel command to visualize the first 7 days using
  your existing view_rem_panel_caffeine2.py.
"""

import argparse, json, os, gc, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import load as joblib_load
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ---------- small helpers ----------

def set_seed(seed=42):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def clean_cols(cols):
    return [re.sub(r"[^\w]+", "_", c).strip("_") for c in cols]


def ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "index" in df.columns:
            df = df.set_index("index")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataframe is not indexed by DatetimeIndex.")
    df = df.copy()
    df.index = df.index.tz_localize(None)
    df = df.sort_index()
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="last")]
    return df


def ensure_finite_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        bad = (~np.isfinite(num)).sum().sum()
        if bad:
            print(f"[WARN] {name}: {int(bad)} non-finite entries; cleaning.")
        df = df.replace([np.inf, -np.inf], np.nan)
        if df.isna().any().any():
            df = df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return df


def ensure_finite_np(x: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(x).all():
        bad = (~np.isfinite(x)).sum()
        print(f"[WARN] {name}: {int(bad)} non-finite entries; nan_to_num.")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


# ---------- label + features (same logic as training script) ----------

def label_with_sleep_detect(df: pd.DataFrame,
                            ios_col_guess=("OIS_norm", "IOS", "OIS"),
                            motion_vec_name="Motion_Ang_Vector",
                            awake_threshold=0.50,
                            sleep_window="2.5T") -> pd.DataFrame:
    df = df.copy()

    # motion vector
    if motion_vec_name not in df.columns:
        xyz = [c for c in df.columns if c.lower() in ("motion_ang_x", "motion_ang_y", "motion_ang_z")]
        if len(xyz) == 3:
            df[motion_vec_name] = np.sqrt((df[xyz] ** 2).sum(axis=1))
        else:
            lxyz = [c for c in df.columns if c.lower() in ("motion_lin_x", "motion_lin_y", "motion_lin_z")]
            if len(lxyz) == 3:
                df[motion_vec_name] = np.sqrt((df[lxyz] ** 2).sum(axis=1))
            else:
                raise RuntimeError("No Motion_Ang_Vector or XYZ motion columns found.")

    # OIS_norm
    ios_col = None
    for c in ios_col_guess:
        if c in df.columns:
            ios_col = c
            break
    if ios_col is None:
        raise RuntimeError("No IOS/OIS column found (OIS_norm/IOS/OIS).")

    if ios_col != "OIS_norm":
        s = pd.to_numeric(df[ios_col], errors="coerce").astype("float32")
        if s.notna().any():
            lo, hi = np.nanpercentile(s.dropna(), [0.5, 99.5])
        else:
            lo, hi = 0.0, 1.0
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
        raise RuntimeError("sleep_detect not found; make sure it is importable.") from e

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
        if i > 0 and y[i - 1] == 0 and y[i] == 1:
            run = 0
        else:
            run += 1
        tslr[i] = run

    tsso = np.zeros_like(y, dtype=np.int32)
    run = 0
    for i in range(len(y)):
        if i > 0 and (not sleep[i - 1]) and sleep[i]:
            run = 0
        else:
            run = (run + 1) if sleep[i] else 0
        tsso[i] = run

    feat_df = feat_df.copy()
    feat_df["time_sin"] = np.sin(2 * np.pi * feat_df.index.hour / 24).astype("float32")
    feat_df["time_cos"] = np.cos(2 * np.pi * feat_df.index.hour / 24).astype("float32")
    feat_df["ts_last_rem"] = np.log1p(np.clip(tslr, 0, 90 * 60)).astype("float32")
    feat_df["ts_sleep_onset"] = np.log1p(np.clip(tsso, 0, 120 * 60)).astype("float32")
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


# ---------- Model ----------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class TinyTransformer(nn.Module):
    def __init__(self, in_ch, d_model=96, nhead=4, ff=192, layers=3, dropout=0.1):
        super().__init__()
        self.prj = nn.Linear(in_ch, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.prj(x)
        x = self.pos(x)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        h = self.enc(torch.cat([cls, x], dim=1))
        z = self.head(self.norm(h[:, 0])).squeeze(-1)
        return z


# ---------- main ----------

def load_best_dir(study_root: Path, study_name: str) -> Path:
    # Prefer "best" if present
    best = study_root / "best"
    if best.exists() and (best / "tx_stage_config.json").exists():
        print(f"[INFO] Using best/ directory: {best}")
        return best

    # Otherwise read Optuna study to find best trial
    import optuna
    db_url = f"sqlite:///{(study_root / 'study.db').resolve()}"
    print(f"[INFO] best/ not found — loading Optuna study '{study_name}' from {db_url}")
    study = optuna.load_study(study_name=study_name, storage=db_url)
    num = study.best_trial.number
    trial_dir = study_root / f"trial_{num:04d}"
    if not trial_dir.exists():
        raise RuntimeError(f"Best trial dir not found: {trial_dir}")
    print(f"[INFO] Best trial is {num}, dir={trial_dir}")
    return trial_dir


def run_inference(study_root: Path, study_name: str, data_path: Path, out_csv: Path):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    best_dir = load_best_dir(study_root, study_name)

    cfg_path = best_dir / "tx_stage_config.json"
    temp_path = best_dir / "tx_stage_temperature.json"
    scaler_path = best_dir / "tx_stage_scaler.pkl"
    ckpt_path = best_dir / "best_tx_stage.pt"

    cfg = json.load(open(cfg_path, "r"))
    win = int(cfg["win"])
    d_model = int(cfg["d_model"])
    heads = int(cfg["heads"])
    layers = int(cfg["layers"])
    ff_dim = int(cfg["ff_dim"])
    dropout = float(cfg["dropout"])
    thr = float(cfg.get("threshold", 0.5))
    feat_names = cfg["features"]

    T = float(json.load(open(temp_path, "r"))["temperature"])

    print(f"[INFO] Window={win}, d_model={d_model}, heads={heads}, layers={layers}, ff_dim={ff_dim}, dropout={dropout}")
    print(f"[INFO] Temperature={T:.3f}, threshold={thr:.3f}")
    print(f"[INFO] Features: {feat_names}")

    scaler = joblib_load(scaler_path)

    model = TinyTransformer(
        in_ch=len(feat_names),
        d_model=d_model,
        nhead=heads,
        ff=ff_dim,
        layers=layers,
        dropout=dropout,
    ).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    print(f"[INFO] Loading parquet: {data_path}")
    df = pd.read_parquet(data_path)
    df.index.name = "time"
    df.columns = clean_cols(df.columns)
    df = ensure_dtindex(df)
    df = ensure_finite_df(df, "raw")

    print("[INFO] Running sleep_detect to construct OIS_norm / Motion_Ang_Vector / sleep state...")
    df = label_with_sleep_detect(df)
    df = ensure_finite_df(df, "after sleep_detect")

    feat_base = pick_feature_columns(df)
    feat_df = df[feat_base].astype("float32").ffill().bfill()
    feat_df = ensure_finite_df(feat_df, "features base")
    feat_df = add_temporal_context(
        feat_df,
        df["y_rem"].astype("uint8"),
        df["is_sleep"].astype("bool"),
    )
    feat_df = ensure_finite_df(feat_df, "features + temporal")

    # Reorder to exactly match training features
    missing = [c for c in feat_names if c not in feat_df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns in test data: {missing}")
    feat_df = feat_df[feat_names]

    X = feat_df.to_numpy(np.float32)
    X = ensure_finite_np(X, "X_raw")
    X = scaler.transform(X).astype("float32")
    X = ensure_finite_np(X, "X_scaled")

    N, C = X.shape
    if N < win:
        raise RuntimeError(f"Too few rows ({N}) for window {win}.")
    print(f"[INFO] Test matrix: N={N:,}, C={C}")

    # Sliding window inference with stride=1
    starts = np.arange(0, N - win + 1, 1, dtype=np.int64)
    times = feat_df.index[win - 1:]  # aligns each window to its last second

    batch_size = 512
    logits_list = []
    print("[INFO] Running inference over full file...")
    for i in tqdm(range(0, len(starts), batch_size), desc="Inference", unit="batch"):
        s_batch = starts[i:i + batch_size]
        xb = np.stack([X[s:s + win] for s in s_batch], axis=0)
        xb_t = torch.from_numpy(xb).to(device)
        with torch.no_grad():
            z = model(xb_t)
            if not torch.isfinite(z).all():
                print("[WARN] non-finite logits; nan_to_num.")
                z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        logits_list.append(z.cpu().numpy())

    logits = np.concatenate(logits_list, axis=0)
    p = 1.0 / (1.0 + np.exp(-(logits / T)))
    p = np.clip(p, 0.0, 1.0)

    print(f"[INFO] Saving predictions to {out_csv}")
    out_df = pd.DataFrame({"time": times, "p_rem": p.astype("float32")})
    out_df.to_csv(out_csv, index=False)

    print("\n[INFO] Done.")
    print("\n[INFO] To visualize FIRST 7 DAYS (after caffeine>=0.3, in your updated viewer), run:\n")
    print(
        "panel serve view_rem_panel_caffeine2.py "
        "--port 5025 --session-token-expiration 86400 "
        "--allow-websocket-origin=localhost:5025 "
        "--allow-websocket-origin=127.0.0.1:5025 "
        "--autoreload --show --args "
        f'--data \"{data_path}\" --pred_csv \"{out_csv}\" --threshold {thr:.3f}'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study_root", required=True,
                    help="Root tuning outdir (e.g., merged_precision_modelNEW_tune_f1_ext_eval)")
    ap.add_argument("--study_name", default="precision_ios_f1_restricted_ext_eval",
                    help="Optuna study_name used in tuning script")
    ap.add_argument("--data", required=True,
                    help="Parquet to run inference on (e.g., test night)")
    ap.add_argument("--out_csv", required=True,
                    help="Where to save CSV with time,p_rem")
    args = ap.parse_args()

    run_inference(Path(args.study_root), args.study_name, Path(args.data), Path(args.out_csv))
    gc.collect()


if __name__ == "__main__":
    main()
