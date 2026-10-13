import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =========================================================
# PATHS
# =========================================================
BASE = Path(r"D:\downloads\Kethan\Kethan")
RUN_DIR = BASE / "runs3" / "rem_mice_manual_v4"
OUT_DIR = BASE / "runs3" / "top5_reconstructed_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRIALS_CSV = RUN_DIR / "trials.csv"

OG_PARQUET = BASE / "data" / "pandas_1.0S.pq"
OG_SCORES  = BASE / "data" / "ManualScores" / "pandas_1.0S.txt"
OG_IGNORE  = BASE / "data" / "ManualScores" / "ignore_ranges.txt"

CAFF_A_PARQUET = BASE / "newData" / "Caffeine_20211021A_optimized_1.0.pq"
CAFF_A_SCORES  = BASE / "newData" / "ManualScores" / "Caffeine_20211021A" / "manualscore.txt"

EVAL_PARQUET = BASE / "newData" / "Caffeine_20211229B_optimized_1.0.pq"
EVAL_SCORES  = BASE / "newData" / "ManualScores" / "Caffeine_20211229B" / "manualscores.txt"

# ---- new: combined caffeine ----
COMB_CAFF_PARQUET = BASE / "newData" / "COMBINED_CAFFEINE.pq"
COMB_CAFF_SCORES  = BASE / "newData" / "ManualScores" / "COMBINED_CAFFEINE" / "manualscore.txt"

# =========================================================
# GLOBALS
# =========================================================
WIN     = 120
STRIDE  = 3
EPOCHS  = 30
PATIENCE = 8
SEED    = 42
TOP_K   = 5

OG_BASE_DATE = datetime.fromisoformat("2022-10-04")
OG_START     = datetime.fromisoformat("2022-10-04")
OG_END       = datetime.fromisoformat("2023-01-23")

CAFF_A_BASE_DATE = datetime.fromisoformat("2021-10-25")
CAFF_A_START     = datetime.fromisoformat("2021-10-25")
CAFF_A_END       = datetime.fromisoformat("2021-11-04")

EVAL_BASE_DATE = datetime.fromisoformat("2022-01-12")
EVAL_START     = datetime.fromisoformat("2022-01-12")
EVAL_END       = datetime.fromisoformat("2022-01-19")

COMB_CAFF_BASE_DATE = datetime.fromisoformat("2022-04-08")
COMB_CAFF_START     = datetime.fromisoformat("2022-04-08")
COMB_CAFF_END       = datetime.fromisoformat("2022-04-15")


# =========================================================
# HELPERS
# =========================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_TIME_RANGE_COMMA_PAT = re.compile(
    r"^\s*([0-9]{1,3}:[0-9]{2}:[0-9]{2})\s*,\s*([0-9]{1,3}:[0-9]{2}:[0-9]{2})\s*$"
)

_IGNORE_RANGE_PAT = re.compile(
    r"^\s*([A-Za-z]{3})\s+([0-9]{1,2})\s+([0-9]{2}:[0-9]{2}:[0-9]{2})\s*(?:→|->)\s*"
    r"([A-Za-z]{3})\s+([0-9]{1,2})\s+([0-9]{2}:[0-9]{2}:[0-9]{2})\s*$"
)


def parse_hhmmss_to_seconds(hhmmss: str) -> int:
    h, m, s = map(int, hhmmss.split(":"))
    return h * 3600 + m * 60 + s


def infer_year(base_date: datetime, month_num: int) -> int:
    if base_date.month >= 10 and month_num <= 3:
        return base_date.year + 1
    return base_date.year


def parse_manual_score_ranges(scores_path: Path, base_date: datetime):
    ranges = []
    with open(scores_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if "training_range" in low or low in ("pandas_1.0s",):
                continue
            m = _TIME_RANGE_COMMA_PAT.match(line)
            if not m:
                continue
            t0 = parse_hhmmss_to_seconds(m.group(1))
            t1 = parse_hhmmss_to_seconds(m.group(2))
            if t1 <= t0:
                continue
            ranges.append((
                base_date + timedelta(seconds=t0),
                base_date + timedelta(seconds=t1),
            ))
    return sorted(ranges, key=lambda x: x[0])


def parse_ignore_ranges(ignore_path: Path, base_date: datetime):
    ranges = []
    with open(ignore_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = _IGNORE_RANGE_PAT.match(line)
            if not m:
                continue

            smon  = _MONTHS[m.group(1).lower()]
            sday  = int(m.group(2))
            stime = m.group(3)
            emon  = _MONTHS[m.group(4).lower()]
            eday  = int(m.group(5))
            etime = m.group(6)

            sy = infer_year(base_date, smon)
            ey = infer_year(base_date, emon)

            sh, smin, ss = map(int, stime.split(":"))
            eh, emin, es = map(int, etime.split(":"))

            start = datetime(sy, smon, sday, sh, smin, ss)
            end   = datetime(ey, emon, eday, eh, emin, es)

            if end <= start:
                end = datetime(ey + 1, emon, eday, eh, emin, es)

            ranges.append((start, end))

    return sorted(ranges, key=lambda x: x[0])


def build_label_mask(index: pd.DatetimeIndex, pos_ranges):
    y = np.zeros(len(index), dtype=np.int64)
    for a, b in pos_ranges:
        y[(index >= a) & (index < b)] = 1
    return y


def apply_ignore_ranges(df: pd.DataFrame, ignore_ranges):
    mask = np.ones(len(df), dtype=bool)
    idx  = df.index
    for a, b in ignore_ranges:
        mask &= ~((idx >= a) & (idx < b))
    return df.loc[mask]


def ensure_tz_naive(df: pd.DataFrame):
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)
    return df


def load_parquet_with_datetime_index(path: Path):
    """
    Handles both single-file parquets and directory-of-parts parquets
    (e.g. COMBINED_CAFFEINE.pq which is stored as a folder of part*.parquet files).
    Tries multiple common datetime column names and always prints the real index range.
    """
    df = pd.read_parquet(path)  # pandas handles directories automatically

    # Try to set the datetime index from a recognised column name
    DT_CANDIDATES = ["t", "time", "timestamp", "datetime", "date", "index", "Timestamp"]
    set_from_col = False
    for col in DT_CANDIDATES:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col).sort_index()
                set_from_col = True
                tqdm.write(f"    -> datetime index set from column '{col}'")
                break
            except Exception:
                continue

    if not set_from_col:
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise RuntimeError(
                    f"Could not parse a DatetimeIndex from {path}.\n"
                    f"Columns available: {list(df.columns)}\n"
                    f"Index dtype: {df.index.dtype}\n"
                    f"Original error: {e}"
                )
        df = df.sort_index()

    df = ensure_tz_naive(df)

    tqdm.write(
        f"    -> raw index range: {df.index.min()}  ->  {df.index.max()}"
        f"  ({len(df):,} rows)"
    )
    return df


# =========================================================
# FEATURE REMAPPING
# =========================================================
CANONICAL_FEATURES = [
    "OIS Background",
    "OIS Signal",
    "OIS Stimulation Current",
    "Motion Ang. X",
    "Motion Ang. Y",
    "Motion Ang. Z",
    "Motion Lin. X",
    "Motion Lin. Y",
    "Motion Lin. Z",
    "OIS",
    "Motion Ang. Vector",
    "Motion Lin. Vector",
    "HR",
]

SYNONYMS = {
    "OIS Background":           ["OIS Background",           "IOS Background"],
    "OIS Signal":               ["OIS Signal",               "IOS Signal"],
    "OIS Stimulation Current":  ["OIS Stimulation Current",  "IOS Stimulation Current"],
    "OIS":                      ["OIS",                      "IOS"],
    "Motion Ang. X":            ["Motion Ang. X"],
    "Motion Ang. Y":            ["Motion Ang. Y"],
    "Motion Ang. Z":            ["Motion Ang. Z"],
    "Motion Lin. X":            ["Motion Lin. X"],
    "Motion Lin. Y":            ["Motion Lin. Y"],
    "Motion Lin. Z":            ["Motion Lin. Z"],
    "Motion Ang. Vector":       ["Motion Ang. Vector"],
    "Motion Lin. Vector":       ["Motion Lin. Vector"],
    "HR":                       ["HR"],
}


def pick_columns_for_dataset(df: pd.DataFrame, dataset_name: str):
    cols    = list(df.columns)
    picked  = []
    missing = []
    for feat in CANONICAL_FEATURES:
        found = None
        for alt in SYNONYMS[feat]:
            if alt in cols:
                found = alt
                break
        if found is None:
            missing.append(feat)
        else:
            picked.append(found)

    if missing:
        raise RuntimeError(f"{dataset_name} missing required features: {missing}")
    return picked


# =========================================================
# DATASET
# =========================================================
@dataclass
class SourceConfig:
    name:               str
    parquet_path:       Path
    base_date:          datetime
    start:              datetime
    end:                datetime
    manual_scores_path: Path
    ignore_ranges_path: Path | None = None


def build_source_matrices(cfg: SourceConfig):
    tqdm.write(f"  Loading parquet : {cfg.parquet_path}")
    df = load_parquet_with_datetime_index(cfg.parquet_path)

    # Print actual range BEFORE filtering so date mismatches are obvious
    tqdm.write(
        f"    -> requested filter: {cfg.start}  ->  {cfg.end}"
    )

    df_filtered = df.loc[(df.index >= cfg.start) & (df.index < cfg.end)]

    if df_filtered.empty:
        raise RuntimeError(
            f"[{cfg.name}] 0 rows after date filtering!\n"
            f"  Parquet range : {df.index.min()}  ->  {df.index.max()}\n"
            f"  Requested     : {cfg.start}  ->  {cfg.end}\n"
            f"  Hint: update COMB_CAFF_START / COMB_CAFF_END in the script "
            f"to match the parquet's actual range above."
        )
    df = df_filtered

    if cfg.ignore_ranges_path is not None:
        df = apply_ignore_ranges(
            df, parse_ignore_ranges(cfg.ignore_ranges_path, cfg.base_date)
        )

    feature_cols = pick_columns_for_dataset(df, cfg.name)
    feat_df = df[feature_cols].copy()

    # ---- diagnose & impute NaNs so sparse datasets don't drop to 0 rows ----
    nan_counts = feat_df.isna().sum()
    bad_cols   = nan_counts[nan_counts > 0]
    if not bad_cols.empty:
        tqdm.write(f"    [{cfg.name}] NaN counts per column (will be imputed):")
        for col, cnt in bad_cols.items():
            tqdm.write(f"      {col}: {cnt:,} NaNs ({100*cnt/len(feat_df):.1f}%)")
        # forward-fill then backward-fill short gaps
        feat_df = feat_df.ffill().bfill()
        # column mean for partially-missing columns
        col_means = feat_df.mean()
        feat_df = feat_df.fillna(col_means)
        # zero sentinel for columns that are ENTIRELY NaN (no mean available)
        still_all_nan = feat_df.columns[feat_df.isna().all()]
        if len(still_all_nan):
            tqdm.write(
                f"    [{cfg.name}] Columns entirely NaN — filling with 0 as sentinel: "
                + ", ".join(still_all_nan)
            )
        feat_df = feat_df.fillna(0.0)

    X  = feat_df.to_numpy()
    y  = build_label_mask(
        df.index,
        parse_manual_score_ranges(cfg.manual_scores_path, cfg.base_date),
    )
    ok = np.isfinite(X).all(axis=1)

    if ok.sum() == 0:
        # Still 0 after imputation — show per-column inf/nan summary
        inf_counts = (~np.isfinite(X)).sum(axis=0)
        tqdm.write(f"    [{cfg.name}] WARNING: 0 finite rows remain after imputation!")
        for i, col in enumerate(feature_cols):
            if inf_counts[i] > 0:
                tqdm.write(f"      {col}: {inf_counts[i]:,} non-finite values")
        raise RuntimeError(
            f"[{cfg.name}] 0 rows remain after NaN imputation and finite filter.\n"
            f"Check the columns listed above — they may be entirely NaN/inf in this date window."
        )

    pct = 100 * y[ok].mean() if ok.sum() > 0 else 0.0
    tqdm.write(
        f"  [{cfg.name}]  rows={ok.sum():,}  pos={y[ok].sum():,}"
        f"  ({pct:.1f}% REM)"
    )
    return X[ok], y[ok], feature_cols, df.index[ok]


class WindowDataset(Dataset):
    def __init__(self, X, y, win, stride):
        self.X      = X.astype(np.float32, copy=False)
        self.y      = y.astype(np.int64,   copy=False)
        self.win    = int(win)
        self.stride = int(stride)
        n           = len(X)
        self.starts = np.arange(0, max(0, n - win + 1), stride, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        x = self.X[s : s + self.win]
        y = int(self.y[s : s + self.win].max())
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# =========================================================
# MODEL
# =========================================================
class TinyTransformer(nn.Module):
    def __init__(self, n_feat, d_model, nhead, nlayers, dropout):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc  = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.enc(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.head(x)


# =========================================================
# EVAL + TRAIN
# =========================================================
@torch.no_grad()
def eval_model(model, loader, device, amp, desc="Val"):
    model.eval()
    ys, ps, probs = [], [], []

    bar = tqdm(loader, desc=f"  {desc}", leave=False, unit="batch",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for xb, yb in bar:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(xb)
        p    = torch.softmax(logits.float(), dim=1)[:, 1]
        pred = logits.argmax(dim=1)

        ys.extend(yb.numpy().tolist())
        ps.extend(pred.cpu().numpy().tolist())
        probs.extend(p.cpu().numpy().tolist())

    if len(ys) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "ys": ys, "ps": ps, "probs": probs}

    return {
        "f1":        float(f1_score(ys, ps, zero_division=0)),
        "precision": float(precision_score(ys, ps, zero_division=0)),
        "recall":    float(recall_score(ys, ps, zero_division=0)),
        "ys":        ys,
        "ps":        ps,
        "probs":     probs,
    }


def train_model(model, train_loader, val_loader_a, val_loader_b,
                device, amp, lr, weight_decay,
                rank: int, trial_number: int):
    """
    Trains with early stopping on the *average* F1 across both eval sets.
    Returns the model loaded with the best checkpoint and the best metrics dict.
    """
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_avg_f1  = -1.0
    best_state   = None
    patience_left = PATIENCE
    best_metrics  = None

    epoch_bar = tqdm(
        range(1, EPOCHS + 1),
        desc=f"  Rank {rank} / Trial {trial_number} — Epochs",
        unit="epoch",
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    for epoch in epoch_bar:
        # ---- training pass ----
        model.train()
        running_loss = 0.0
        n_batches    = 0

        train_bar = tqdm(
            train_loader,
            desc=f"    Train ep{epoch:02d}",
            leave=False,
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        for xb, yb in train_bar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(xb)
                loss   = F.cross_entropy(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()
            n_batches    += 1
            train_bar.set_postfix(loss=f"{running_loss / n_batches:.4f}")

        # ---- eval on both sets ----
        metrics_a = eval_model(model, val_loader_a, device, amp, desc="Eval-A (20211229B)")
        metrics_b = eval_model(model, val_loader_b, device, amp, desc="Eval-B (COMB_CAFF)")

        avg_f1 = (metrics_a["f1"] + metrics_b["f1"]) / 2.0

        avg_prec = (metrics_a["precision"] + metrics_b["precision"]) / 2.0
        avg_rec  = (metrics_a["recall"]    + metrics_b["recall"])    / 2.0

        epoch_bar.set_postfix(
            loss      = f"{running_loss / max(1, n_batches):.4f}",
            avg_f1    = f"{avg_f1:.4f}",
            avg_P     = f"{avg_prec:.4f}",
            avg_R     = f"{avg_rec:.4f}",
            f1_A      = f"{metrics_a['f1']:.4f}",
            P_A       = f"{metrics_a['precision']:.4f}",
            R_A       = f"{metrics_a['recall']:.4f}",
            f1_B      = f"{metrics_b['f1']:.4f}",
            P_B       = f"{metrics_b['precision']:.4f}",
            R_B       = f"{metrics_b['recall']:.4f}",
            patience  = patience_left,
        )

        tqdm.write(
            f"    Epoch {epoch:02d}  "
            f"loss={running_loss / max(1, n_batches):.4f}  "
            f"│  Eval-A (20211229B):  "
            f"F1={metrics_a['f1']:.4f}  "
            f"P={metrics_a['precision']:.4f}  "
            f"R={metrics_a['recall']:.4f}  "
            f"│  Eval-B (COMB_CAFF):  "
            f"F1={metrics_b['f1']:.4f}  "
            f"P={metrics_b['precision']:.4f}  "
            f"R={metrics_b['recall']:.4f}  "
            f"│  avg F1={avg_f1:.4f}  "
            f"avg P={avg_prec:.4f}  "
            f"avg R={avg_rec:.4f}  "
            f"│  patience={patience_left}"
        )

        if avg_f1 > best_avg_f1:
            best_avg_f1  = avg_f1
            best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"eval_a": metrics_a, "eval_b": metrics_b, "avg_f1": avg_f1}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                tqdm.write("    ⏹  Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training never produced a valid checkpoint.")

    model.load_state_dict(best_state)
    return model, best_metrics


# =========================================================
# INFERENCE PASS (returns a DataFrame)
# =========================================================
@torch.no_grad()
def run_inference(model, X_scaled, y_true, eval_index, batch_size, device, amp, label: str):
    """
    Runs inference over a scaled feature matrix and returns a predictions DataFrame.
    """
    infer_ds     = WindowDataset(X_scaled, np.zeros(len(X_scaled), dtype=np.int64), WIN, STRIDE)
    infer_loader = DataLoader(infer_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    starts       = infer_ds.starts

    model.eval()
    rows = []

    infer_bar = tqdm(
        infer_loader,
        desc=f"  Inference [{label}]",
        leave=False,
        unit="batch",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for batch_idx, (xb, _) in enumerate(infer_bar):
        xb    = xb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(xb)
        probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        base = batch_idx * batch_size
        for j in range(len(probs)):
            s  = int(starts[base + j])
            ts = eval_index[s + WIN - 1]
            rows.append({
                "timestamp":     ts,
                "pred_label":    int(preds[j]),
                "pred_prob_rem": float(probs[j]),
            })

    return pd.DataFrame(rows)


# =========================================================
# TOP TRIAL EXTRACTION
# =========================================================
def load_top_trials(trials_csv: Path, top_k: int = 5):
    df = pd.read_csv(trials_csv)

    if "state" in df.columns:
        df = df[df["state"].astype(str).str.upper().isin(["COMPLETE", "1", "FINISHED"])].copy()

    if "value" not in df.columns:
        raise RuntimeError("trials.csv is missing 'value' column.")

    df = df[np.isfinite(df["value"])].copy()
    df = df.sort_values("value", ascending=False).reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No completed trials found in trials.csv.")

    return df.head(top_k).copy()


def extract_trial_params(row: pd.Series):
    def first_existing(*names, default=None):
        for n in names:
            if n in row.index and pd.notna(row[n]):
                return row[n]
        return default

    params = {
        "batch_size":   int(first_existing("params_batch_size",   "batch_size")),
        "lr":         float(first_existing("params_lr",           "lr")),
        "weight_decay": float(first_existing("params_weight_decay","weight_decay")),
        "d_model":      int(first_existing("params_d_model",      "d_model")),
        "n_heads":      int(first_existing("params_n_heads", "params_nhead", "n_heads", "nhead")),
        "n_layers":     int(first_existing("params_n_layers","params_nlayers","n_layers","nlayers")),
        "dropout":    float(first_existing("params_dropout",      "dropout")),
    }

    if params["d_model"] % params["n_heads"] != 0:
        raise RuntimeError(
            f"Invalid trial params: d_model {params['d_model']} "
            f"not divisible by n_heads {params['n_heads']}"
        )

    return params


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp    = torch.cuda.is_available()

    tqdm.write(f"Using device : {device}")
    tqdm.write(f"AMP enabled  : {amp}")
    tqdm.write("=" * 80)

    # ------------------------------------------------------------------
    # Load all source datasets
    # ------------------------------------------------------------------
    print("\n[1/4] Loading datasets …\n")

    sources_bar = tqdm(
        ["OG", "CAFF_A", "EVAL (20211229B)", "COMB_CAFF"],
        desc="Datasets",
        unit="ds",
    )

    og = SourceConfig(
        "OG", OG_PARQUET, OG_BASE_DATE, OG_START, OG_END, OG_SCORES, OG_IGNORE,
    )
    caff_a = SourceConfig(
        "CAFF_A", CAFF_A_PARQUET, CAFF_A_BASE_DATE, CAFF_A_START, CAFF_A_END, CAFF_A_SCORES,
    )
    ev = SourceConfig(
        "EVAL", EVAL_PARQUET, EVAL_BASE_DATE, EVAL_START, EVAL_END, EVAL_SCORES,
    )
    comb_caff = SourceConfig(
        "COMB_CAFF", COMB_CAFF_PARQUET, COMB_CAFF_BASE_DATE,
        COMB_CAFF_START, COMB_CAFF_END, COMB_CAFF_SCORES,
    )

    sources_bar.update(0)
    X1, y1, _, _           = build_source_matrices(og);      sources_bar.update(1)
    X2, y2, _, _           = build_source_matrices(caff_a);  sources_bar.update(1)
    Xe,  ye,  _, eval_idx  = build_source_matrices(ev);      sources_bar.update(1)
    Xcc, ycc, _, cc_idx    = build_source_matrices(comb_caff); sources_bar.update(1)
    sources_bar.close()

    X_train = np.concatenate([X1, X2], axis=0)
    y_train = np.concatenate([y1, y2], axis=0)

    tqdm.write(f"\nTrain rows       : {len(X_train):,}")
    tqdm.write(f"Eval-A rows      : {len(Xe):,}  (Caffeine_20211229B)")
    tqdm.write(f"Eval-B rows      : {len(Xcc):,}  (COMBINED_CAFFEINE)")

    # ------------------------------------------------------------------
    # Load top trials
    # ------------------------------------------------------------------
    print("\n[2/4] Loading top trials …\n")
    top_trials = load_top_trials(TRIALS_CSV, TOP_K)
    tqdm.write(f"Top-{TOP_K} trials loaded from {TRIALS_CSV}")
    tqdm.write(top_trials[["number", "value"]].to_string(index=False))

    # ------------------------------------------------------------------
    # Reconstruct + retrain each trial
    # ------------------------------------------------------------------
    print(f"\n[3/4] Reconstructing top-{TOP_K} models …\n")

    summary_rows = []

    outer_bar = tqdm(
        top_trials.iterrows(),
        total=len(top_trials),
        desc="Top-K reconstruction",
        unit="model",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        position=0,
    )

    for rank, (_, row) in enumerate(outer_bar, start=1):
        trial_number   = int(row["number"]) if "number" in row.index and pd.notna(row["number"]) else rank - 1
        original_value = float(row["value"])
        params         = extract_trial_params(row)

        trial_dir = OUT_DIR / f"rank_{rank:02d}_trial_{trial_number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        outer_bar.set_description(
            f"Rank {rank}/{TOP_K}  trial={trial_number}  orig_F1={original_value:.4f}"
        )
        tqdm.write("\n" + "=" * 80)
        tqdm.write(
            f"  Rank {rank}/{TOP_K}  |  trial={trial_number}  |  "
            f"original_F1={original_value:.4f}"
        )
        tqdm.write(f"  Params: {params}")

        # ---- scale ----
        scaler           = StandardScaler()
        X_train_scaled   = scaler.fit_transform(X_train.astype(np.float32))
        X_eval_scaled    = scaler.transform(Xe.astype(np.float32))
        X_cc_scaled      = scaler.transform(Xcc.astype(np.float32))

        # ---- dataloaders ----
        train_ds  = WindowDataset(X_train_scaled, y_train, WIN, STRIDE)
        eval_ds_a = WindowDataset(X_eval_scaled,  ye,      WIN, STRIDE)
        eval_ds_b = WindowDataset(X_cc_scaled,    ycc,     WIN, STRIDE)

        bs = params["batch_size"]
        train_loader  = DataLoader(train_ds,  batch_size=bs, shuffle=True,  num_workers=0)
        eval_loader_a = DataLoader(eval_ds_a, batch_size=bs, shuffle=False, num_workers=0)
        eval_loader_b = DataLoader(eval_ds_b, batch_size=bs, shuffle=False, num_workers=0)

        tqdm.write(
            f"  Windows — train={len(train_ds):,}  "
            f"eval_A={len(eval_ds_a):,}  eval_B={len(eval_ds_b):,}"
        )

        # ---- build model ----
        model = TinyTransformer(
            n_feat  = len(CANONICAL_FEATURES),
            d_model = params["d_model"],
            nhead   = params["n_heads"],
            nlayers = params["n_layers"],
            dropout = params["dropout"],
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tqdm.write(f"  Model params    : {n_params:,}")

        # ---- train ----
        model, best_metrics = train_model(
            model         = model,
            train_loader  = train_loader,
            val_loader_a  = eval_loader_a,
            val_loader_b  = eval_loader_b,
            device        = device,
            amp           = amp,
            lr            = params["lr"],
            weight_decay  = params["weight_decay"],
            rank          = rank,
            trial_number  = trial_number,
        )

        ma = best_metrics["eval_a"]
        mb = best_metrics["eval_b"]
        avg_f1 = best_metrics["avg_f1"]

        tqdm.write(
            f"\n  ✓  Best checkpoint:"
            f"\n     Eval-A (20211229B)  F1={ma['f1']:.4f}  P={ma['precision']:.4f}  R={ma['recall']:.4f}"
            f"\n     Eval-B (COMB_CAFF) F1={mb['f1']:.4f}  P={mb['precision']:.4f}  R={mb['recall']:.4f}"
            f"\n     Average F1         = {avg_f1:.4f}"
        )

        # ---- save model artefacts ----
        torch.save(model.state_dict(), trial_dir / "best_model.pt")
        joblib.dump(scaler, trial_dir / "scaler.joblib")

        with open(trial_dir / "features.json", "w", encoding="utf-8") as fp:
            json.dump({"features": CANONICAL_FEATURES}, fp, indent=2)

        with open(trial_dir / "config.json", "w", encoding="utf-8") as fp:
            json.dump({
                "rank":                    rank,
                "original_trial_number":   trial_number,
                "original_trial_value":    original_value,
                # ---- Eval-A (Caffeine_20211229B) ----
                "eval_a_name":             "Caffeine_20211229B",
                "eval_a_f1":               ma["f1"],
                "eval_a_precision":        ma["precision"],
                "eval_a_recall":           ma["recall"],
                # ---- Eval-B (COMBINED_CAFFEINE) ----
                "eval_b_name":             "COMBINED_CAFFEINE",
                "eval_b_f1":               mb["f1"],
                "eval_b_precision":        mb["precision"],
                "eval_b_recall":           mb["recall"],
                # ---- composite metric used for early-stopping + ranking ----
                "avg_f1_both_evals":       avg_f1,
                # ---- hyperparams ----
                "params":                  params,
                "win":                     WIN,
                "stride":                  STRIDE,
                "epochs":                  EPOCHS,
                "patience":                PATIENCE,
            }, fp, indent=2)

        # ---- inference CSVs for both eval sets ----
        tqdm.write("  Running inference on both eval sets …")

        pred_df_a = run_inference(
            model, X_eval_scaled, ye, eval_idx, bs, device, amp,
            label="Caffeine_20211229B",
        )
        pred_df_a.to_csv(trial_dir / "pred_Caffeine_20211229B.csv", index=False)

        pred_df_b = run_inference(
            model, X_cc_scaled, ycc, cc_idx, bs, device, amp,
            label="COMBINED_CAFFEINE",
        )
        pred_df_b.to_csv(trial_dir / "pred_COMBINED_CAFFEINE.csv", index=False)

        tqdm.write(f"  ✓  Saved to: {trial_dir}")

        summary_rows.append({
            "rank":                  rank,
            "original_trial_number": trial_number,
            "original_trial_value":  original_value,
            "avg_f1_both_evals":     avg_f1,
            "eval_a_f1":             ma["f1"],
            "eval_a_precision":      ma["precision"],
            "eval_a_recall":         ma["recall"],
            "eval_b_f1":             mb["f1"],
            "eval_b_precision":      mb["precision"],
            "eval_b_recall":         mb["recall"],
            **params,
        })

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    print("\n[4/4] Saving summary …\n")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "top5_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    tqdm.write("=" * 80)
    tqdm.write("Done.")
    tqdm.write(f"Summary  → {summary_csv}")
    tqdm.write(f"Models   → {OUT_DIR}")
    tqdm.write("")
    tqdm.write(
        summary_df[["rank", "original_trial_number", "avg_f1_both_evals",
                    "eval_a_f1", "eval_b_f1"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()