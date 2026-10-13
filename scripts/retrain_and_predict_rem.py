import json
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
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# =========================
# PATHS
# =========================
BASE = Path(r"D:\downloads\Kethan\Kethan")
RUN_DIR = BASE / "runs3" / "rem_mice_manual_v4"
OUT_DIR = BASE / "runs3" / "rem_mice_manual_v4_local_infer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OG_PARQUET = BASE / "data" / "pandas_1.0S.pq"
OG_SCORES = BASE / "data" / "ManualScores" / "pandas_1.0S.txt"
OG_IGNORE = BASE / "data" / "ManualScores" / "ignore_ranges.txt"

CAFF_A_PARQUET = BASE / "newData" / "Caffeine_20211021A_optimized_1.0.pq"
CAFF_A_SCORES = BASE / "newData" / "ManualScores" / "Caffeine_20211021A" / "manualscore.txt"

EVAL_PARQUET = BASE / "newData" / "Caffeine_20211229B_optimized_1.0.pq"
EVAL_SCORES = BASE / "newData" / "ManualScores" / "Caffeine_20211229B" / "manualscores.txt"

# =========================
# FIXED BEST PARAMS
# =========================
BEST = {
    "batch_size": 512,
    "lr": 0.0003257350089455583,
    "weight_decay": 1.32839750761014e-05,
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 4,
    "dropout": 4.192227547521535e-05,
}

WIN = 120
STRIDE = 3
EPOCHS = 30
PATIENCE = 8
SEED = 42

OG_BASE_DATE = datetime.fromisoformat("2022-10-04")
OG_START = datetime.fromisoformat("2022-10-04")
OG_END = datetime.fromisoformat("2023-01-23")

CAFF_A_BASE_DATE = datetime.fromisoformat("2021-10-25")
CAFF_A_START = datetime.fromisoformat("2021-10-25")
CAFF_A_END = datetime.fromisoformat("2021-11-04")

EVAL_BASE_DATE = datetime.fromisoformat("2022-01-12")
EVAL_START = datetime.fromisoformat("2022-01-12")
EVAL_END = datetime.fromisoformat("2022-01-19")


# =========================
# HELPERS
# =========================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_TIME_RANGE_COMMA_PAT = re.compile(r"^\s*([0-9]{1,3}:[0-9]{2}:[0-9]{2})\s*,\s*([0-9]{1,3}:[0-9]{2}:[0-9]{2})\s*$")
_IGNORE_RANGE_PAT = re.compile(
    r"^\s*([A-Za-z]{3})\s+([0-9]{1,2})\s+([0-9]{2}:[0-9]{2}:[0-9]{2})\s*(?:→|->)\s*([A-Za-z]{3})\s+([0-9]{1,2})\s+([0-9]{2}:[0-9]{2}:[0-9]{2})\s*$"
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
            if "training_range" in line.lower() or line.lower() in ("pandas_1.0s",):
                continue
            m = _TIME_RANGE_COMMA_PAT.match(line)
            if not m:
                continue
            t0 = parse_hhmmss_to_seconds(m.group(1))
            t1 = parse_hhmmss_to_seconds(m.group(2))
            if t1 <= t0:
                continue
            ranges.append((base_date + timedelta(seconds=t0), base_date + timedelta(seconds=t1)))
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
            smon = _MONTHS[m.group(1).lower()]
            sday = int(m.group(2))
            stime = m.group(3)
            emon = _MONTHS[m.group(4).lower()]
            eday = int(m.group(5))
            etime = m.group(6)

            sy = infer_year(base_date, smon)
            ey = infer_year(base_date, emon)

            sh, smin, ss = map(int, stime.split(":"))
            eh, emin, es = map(int, etime.split(":"))

            start = datetime(sy, smon, sday, sh, smin, ss)
            end = datetime(ey, emon, eday, eh, emin, es)
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
    idx = df.index
    for a, b in ignore_ranges:
        mask &= ~((idx >= a) & (idx < b))
    return df.loc[mask]

def ensure_tz_naive(df: pd.DataFrame):
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)
    return df

def load_parquet_with_datetime_index(path: Path):
    df = pd.read_parquet(path)
    if "t" in df.columns:
        df["t"] = pd.to_datetime(df["t"])
        df = df.set_index("t").sort_index()
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    return ensure_tz_naive(df)


# =========================
# FEATURE REMAPPING
# =========================
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
    "OIS Background": ["OIS Background", "IOS Background"],
    "OIS Signal": ["OIS Signal", "IOS Signal"],
    "OIS Stimulation Current": ["OIS Stimulation Current", "IOS Stimulation Current"],
    "OIS": ["OIS", "IOS"],
    "Motion Ang. X": ["Motion Ang. X"],
    "Motion Ang. Y": ["Motion Ang. Y"],
    "Motion Ang. Z": ["Motion Ang. Z"],
    "Motion Lin. X": ["Motion Lin. X"],
    "Motion Lin. Y": ["Motion Lin. Y"],
    "Motion Lin. Z": ["Motion Lin. Z"],
    "Motion Ang. Vector": ["Motion Ang. Vector"],
    "Motion Lin. Vector": ["Motion Lin. Vector"],
    "HR": ["HR"],
}

def pick_columns_for_dataset(df: pd.DataFrame, dataset_name: str):
    cols = list(df.columns)
    picked = []
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


# =========================
# DATASET
# =========================
@dataclass
class SourceConfig:
    name: str
    parquet_path: Path
    base_date: datetime
    start: datetime
    end: datetime
    manual_scores_path: Path
    ignore_ranges_path: Path | None = None


def build_source_matrices(cfg: SourceConfig):
    df = load_parquet_with_datetime_index(cfg.parquet_path)
    df = df.loc[(df.index >= cfg.start) & (df.index < cfg.end)]
    if cfg.ignore_ranges_path is not None:
        df = apply_ignore_ranges(df, parse_ignore_ranges(cfg.ignore_ranges_path, cfg.base_date))

    feature_cols = pick_columns_for_dataset(df, cfg.name)
    X = df[feature_cols].to_numpy()
    y = build_label_mask(df.index, parse_manual_score_ranges(cfg.manual_scores_path, cfg.base_date))
    ok = np.isfinite(X).all(axis=1)
    return X[ok], y[ok], feature_cols, df.index[ok]


class WindowDataset(Dataset):
    def __init__(self, X, y, win, stride):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.win = int(win)
        self.stride = int(stride)
        n = len(X)
        self.starts = np.arange(0, max(0, n - win + 1), stride, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        x = self.X[s:s+self.win]
        y = int(self.y[s:s+self.win].max())
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# =========================
# MODEL
# =========================
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
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
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


@torch.no_grad()
def eval_model(model, loader, device, amp):
    model.eval()
    ys, ps, probs = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(xb)
        p = torch.softmax(logits.float(), dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        ys.extend(yb.numpy().tolist())
        ps.extend(pred.cpu().numpy().tolist())
        probs.extend(p.cpu().numpy().tolist())
    return ys, ps, probs


def train_model(model, train_loader, val_loader, device, amp):
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=BEST["lr"],
        weight_decay=BEST["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_f1 = -1.0
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        ys, ps, _ = eval_model(model, val_loader, device, amp)
        f1 = 0.0 if len(ys) == 0 else f1_score(ys, ps, zero_division=0)
        print(f"Epoch {epoch:02d}  val_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("Training never produced a valid checkpoint.")
    model.load_state_dict(best_state)
    return model, best_f1


# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available()

    print("Using device:", device)

    og = SourceConfig(
        "OG",
        OG_PARQUET,
        OG_BASE_DATE,
        OG_START,
        OG_END,
        OG_SCORES,
        OG_IGNORE,
    )
    caff_a = SourceConfig(
        "CAFF_A",
        CAFF_A_PARQUET,
        CAFF_A_BASE_DATE,
        CAFF_A_START,
        CAFF_A_END,
        CAFF_A_SCORES,
        None,
    )
    ev = SourceConfig(
        "EVAL",
        EVAL_PARQUET,
        EVAL_BASE_DATE,
        EVAL_START,
        EVAL_END,
        EVAL_SCORES,
        None,
    )

    X1, y1, f1, _ = build_source_matrices(og)
    X2, y2, f2, _ = build_source_matrices(caff_a)
    Xe, ye, fe, eval_index = build_source_matrices(ev)

    # By construction the selected feature order is canonical and consistent
    X_train = np.concatenate([X1, X2], axis=0)
    y_train = np.concatenate([y1, y2], axis=0)
    X_eval = Xe
    y_eval = ye

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.astype(np.float32))
    X_eval = scaler.transform(X_eval.astype(np.float32))

    train_ds = WindowDataset(X_train, y_train, WIN, STRIDE)
    eval_ds = WindowDataset(X_eval, y_eval, WIN, STRIDE)

    train_loader = DataLoader(train_ds, batch_size=BEST["batch_size"], shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=BEST["batch_size"], shuffle=False, num_workers=0)

    model = TinyTransformer(
        n_feat=len(CANONICAL_FEATURES),
        d_model=BEST["d_model"],
        nhead=BEST["n_heads"],
        nlayers=BEST["n_layers"],
        dropout=BEST["dropout"],
    ).to(device)

    model, best_f1 = train_model(model, train_loader, eval_loader, device, amp)
    print("Best eval F1 after retrain:", best_f1)

    torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    with open(OUT_DIR / "features.json", "w", encoding="utf-8") as f:
        json.dump({"features": CANONICAL_FEATURES}, f, indent=2)

    # Inference on eval parquet at window level
    infer_ds = WindowDataset(X_eval, np.zeros_like(y_eval), WIN, STRIDE)
    infer_loader = DataLoader(infer_ds, batch_size=BEST["batch_size"], shuffle=False, num_workers=0)

    model.eval()
    rows = []
    starts = infer_ds.starts
    with torch.no_grad():
        for batch_idx, (xb, _) in enumerate(infer_loader):
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(xb)
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            base = batch_idx * BEST["batch_size"]
            for j in range(len(probs)):
                s = int(starts[base + j])
                ts = eval_index[s + WIN - 1]
                rows.append({
                    "timestamp": ts,
                    "pred_label": int(preds[j]),
                    "pred_prob_rem": float(probs[j]),
                })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(OUT_DIR / "pred_Caffeine_20211229B.csv", index=False)

    print("Saved:")
    print(OUT_DIR / "best_model.pt")
    print(OUT_DIR / "scaler.joblib")
    print(OUT_DIR / "features.json")
    print(OUT_DIR / "pred_Caffeine_20211229B.csv")


if __name__ == "__main__":
    main()