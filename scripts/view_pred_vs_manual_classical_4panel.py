#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import time
import warnings
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable, Optional, List

import numpy as np
import pandas as pd

import panel as pn
import holoviews as hv
import hvplot.pandas  # noqa: F401
from bokeh.models import DatetimeTickFormatter

from sleep_detect_2 import extract_awake_rem_nrem

warnings.filterwarnings("ignore")
hv.extension("bokeh")
pn.extension("tabulator", notifications=True)
pn.config.theme = "dark"

# =========================================================
# Shading colour palette
# =========================================================
SHADE_ALL_3 = {"color": "#18C5C5", "alpha": 0.42}
SHADE_ML_MAN = {"color": "#32CD7A", "alpha": 0.34}
SHADE_ML_CLASSICAL = {"color": "#8E6CFF", "alpha": 0.34}
SHADE_MAN_CLASSICAL = {"color": "#FF9F43", "alpha": 0.32}
SHADE_ML_ONLY = {"color": "#4C78FF", "alpha": 0.28}
SHADE_MAN_ONLY = {"color": "#E8C547", "alpha": 0.26}
SHADE_CLASSICAL_ONLY = {"color": "#FF5A5F", "alpha": 0.26}

# =========================================================
# Candidate columns
# =========================================================
TIME_CANDIDATES = [
    "t_idx",
    "timestamp", "time_stamp", "index", "t",
    "__index_level_0__", "datetime", "date_time", "time"
]

IOS_CANDIDATES = [
    "IOS", "OIS", "IOS Signal", "OIS Signal",
    "IOS Background", "OIS Background",
]

MOTION_CANDIDATES = [
    "Motion Ang. Vector", "Motion Ang Vector", "Motion_Ang_Vector",
    "Motion Lin. Vector", "Motion Lin Vector", "Motion_Lin_Vector",
]

CAFFEINE_CANDIDATES = ["caffeine", "Caffeine"]

REM_BOOL_CANDIDATES = [
    "rem", "REM", "is_rem", "pred_rem", "rem_mask", "rem_bool", "rem_label",
]

REM_SCORE_CANDIDATES = [
    "pred_prob_rem", "rem_prob", "prob_rem", "p_rem", "rem_score", "rem_probability",
]

STAGE_CANDIDATES = [
    "stage", "state", "sleep_stage", "label", "pred_label", "stage_label",
]


# =========================================================
# Helpers
# =========================================================
def _drop_dup_index(obj, keep: str = "last"):
    if isinstance(obj, (pd.Series, pd.DataFrame)) and not obj.index.is_unique:
        obj = obj[~obj.index.duplicated(keep=keep)]
    return obj


def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        chosen = None
        for c in TIME_CANDIDATES:
            if c in df.columns:
                chosen = c
                break

        if chosen is not None:
            idx = pd.to_datetime(df[chosen], errors="coerce")
            df = df.drop(columns=[chosen])
        else:
            if isinstance(df.index, pd.RangeIndex):
                raise ValueError(
                    f"Could not find a real datetime column. Existing index is RangeIndex, "
                    f"not datetime. Available columns: {list(df.columns)[:50]}"
                )

            idx_dtype = getattr(df.index, "dtype", None)
            if idx_dtype is not None and str(idx_dtype).startswith(("int", "uint", "float")):
                raise ValueError(
                    f"Could not find a real datetime column. Existing index is numeric, "
                    f"not datetime. Available columns: {list(df.columns)[:50]}"
                )

            idx_try = pd.to_datetime(df.index, errors="coerce")
            if np.any(~pd.isna(idx_try)):
                idx = idx_try
            else:
                raise ValueError(
                    f"Could not find a datetime index/column. "
                    f"Available columns: {list(df.columns)[:50]}"
                )

    with suppress(Exception):
        idx = idx.tz_localize(None)
    with suppress(Exception):
        idx = idx.tz_convert(None)

    good = ~pd.isna(idx)
    if not np.all(good):
        df = df.iloc[np.flatnonzero(good)]

    df.index = pd.DatetimeIndex(idx[good])

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    return _drop_dup_index(df, keep="last")


def _spans_from_mask(idx: pd.DatetimeIndex, mask: np.ndarray):
    spans = []
    n = len(idx)
    if n == 0:
        return spans

    m = mask.astype(np.int8)
    diff = m[1:] - m[:-1]
    on = np.flatnonzero(diff == 1)
    off = np.flatnonzero(diff == -1)

    if mask[0]:
        on = np.r_[[-1], on]
    if mask[-1]:
        off = np.r_[off, [n - 1]]

    for s_i, e_i in zip(on + 1, off + 1):
        a = idx[int(s_i)]
        b = idx[int(max(0, e_i - 1))]
        spans.append((a, b))
    return spans


def _vspan_overlay(spans, color: str, alpha: float) -> hv.Overlay:
    elems = [hv.VSpan(a, b).opts(color=color, alpha=alpha) for a, b in spans]
    return hv.Overlay(elems) if elems else hv.Overlay([])


def _series_to_bool_mask(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.fillna(False).to_numpy(dtype=bool)

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).to_numpy() > 0

    txt = s.astype(str).str.strip().str.lower()
    true_vals = {"1", "true", "t", "yes", "y", "rem", "r"}
    return txt.isin(true_vals).to_numpy(dtype=bool)


def _coerce_stage_to_rem_mask(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").fillna(-999)
        return vals.isin([1, 2, 3]).to_numpy(dtype=bool)

    txt = s.astype(str).str.strip().str.lower()
    rem_words = {"rem", "r", "stage_rem", "sleep_rem"}
    return txt.isin(rem_words).to_numpy(dtype=bool)


def _extract_mask_from_obj(obj: Any) -> Optional[np.ndarray]:
    if obj is None:
        return None

    if isinstance(obj, pd.Series):
        return _series_to_bool_mask(obj)

    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if arr.ndim != 1:
            return None
        if arr.dtype == bool:
            return arr.astype(bool)
        if np.issubdtype(arr.dtype, np.number):
            return arr > 0
        arr_s = pd.Series(arr)
        return _series_to_bool_mask(arr_s)

    if isinstance(obj, (list, tuple)):
        arr = np.asarray(obj)
        if arr.ndim == 1 and len(arr) > 0:
            if arr.dtype == bool:
                return arr.astype(bool)
            if np.issubdtype(arr.dtype, np.number):
                return arr > 0
        return None

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        with suppress(Exception):
            df = _ensure_dtindex(df)

        for c in REM_BOOL_CANDIDATES:
            if c in df.columns:
                return _series_to_bool_mask(df[c])

        for c in REM_SCORE_CANDIDATES:
            if c in df.columns:
                return (pd.to_numeric(df[c], errors="coerce").fillna(0) > 0.5).to_numpy(dtype=bool)

        for c in STAGE_CANDIDATES:
            if c in df.columns:
                return _coerce_stage_to_rem_mask(df[c])

        if df.shape[1] == 1:
            return _extract_mask_from_obj(df.iloc[:, 0])

        return None

    if isinstance(obj, dict):
        for key in ["rem", "REM", "rem_mask", "pred_rem", "rem_bool"]:
            if key in obj:
                return _extract_mask_from_obj(obj[key])

        for key in ["stage", "state", "sleep_stage", "label"]:
            if key in obj:
                return _extract_mask_from_obj(obj[key])

        return None

    return None


def _mask_from_ranges(index: pd.DatetimeIndex, ranges: Iterable[Any]) -> np.ndarray:
    mask = np.zeros(len(index), dtype=bool)
    for item in ranges:
        if item is None:
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            a, b = item[0], item[1]
            try:
                a = pd.to_datetime(a)
                b = pd.to_datetime(b)
            except Exception:
                continue
            mask |= ((index >= a) & (index < b))
    return mask


def _align_mask_to_index(mask_like: Any, source_index: pd.DatetimeIndex, target_index: pd.DatetimeIndex) -> np.ndarray:
    mask_arr = np.asarray(mask_like).astype(bool)
    n = min(len(mask_arr), len(source_index))
    if n == 0 or len(target_index) == 0:
        return np.zeros(len(target_index), dtype=bool)

    s = pd.Series(mask_arr[:n], index=source_index[:n]).sort_index()
    s = _drop_dup_index(s, keep="last")

    exact = s.reindex(target_index)
    if exact.notna().all():
        return exact.fillna(False).to_numpy(dtype=bool)

    aligned = s.reindex(target_index, method="nearest", tolerance=pd.Timedelta(seconds=2))
    return aligned.fillna(False).to_numpy(dtype=bool)


# =========================================================
# Multi-model overlay
# =========================================================
def _seven_state_overlay(
    pred_win: pd.DataFrame,
    show_all3: bool,
    show_ml_man: bool,
    show_ml_cla: bool,
    show_man_cla: bool,
    show_ml_only: bool,
    show_man_only: bool,
    show_cla_only: bool,
) -> hv.Overlay:
    idx = pred_win.index
    ml = pred_win["pred_thr"].astype(bool).to_numpy()
    man = pred_win["manual_rem"].astype(bool).to_numpy()
    cla = pred_win["classical_rem"].astype(bool).to_numpy()

    masks = {
        "all3": ml & man & cla,
        "ml_man": ml & man & ~cla,
        "ml_cla": ml & ~man & cla,
        "man_cla": ~ml & man & cla,
        "ml_only": ml & ~man & ~cla,
        "man_only": ~ml & man & ~cla,
        "cla_only": ~ml & ~man & cla,
    }

    all_spans = hv.Overlay([])

    if show_cla_only:
        all_spans *= _vspan_overlay(_spans_from_mask(idx, masks["cla_only"]), **SHADE_CLASSICAL_ONLY)
    if show_man_only:
        all_spans *= _vspan_overlay(_spans_from_mask(idx, masks["man_only"]), **SHADE_MAN_ONLY)
    if show_ml_only:
        all_spans *= _vspan_overlay(_spans_from_mask(idx, masks["ml_only"]), **SHADE_ML_ONLY)
    if show_man_cla:
        all_spans *= _vspan_overlay(_spans_from_mask(idx, masks["man_cla"]), **SHADE_MAN_CLASSICAL)
    if show_ml_cla:
        all_spans *= _vspan_overlay(_spans_from_mask(idx, masks["ml_cla"]), **SHADE_ML_CLASSICAL)
    if show_ml_man:
        all_spans *= _vspan_overlay(_spans_from_mask(idx, masks["ml_man"]), **SHADE_ML_MAN)
    if show_all3:
        all_spans *= _vspan_overlay(_spans_from_mask(idx, masks["all3"]), **SHADE_ALL_3)

    return all_spans


# =========================================================
# Manual score parsing
# =========================================================
_TIME_RANGE_COMMA_PAT = re.compile(
    r"^\s*([0-9]{1,3}:[0-9]{2}:[0-9]{2})\s*,\s*([0-9]{1,3}:[0-9]{2}:[0-9]{2})\s*$"
)


def _parse_hhmmss_to_seconds(hhmmss: str) -> int:
    h, m, s = map(int, hhmmss.split(":"))
    return h * 3600 + m * 60 + s


def _parse_manual_score_ranges(scores_path: Path, base_date: pd.Timestamp):
    ranges = []
    with open(scores_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            low = line.lower()
            if "training_range" in low or "manual_scores" in low or low in {
                "pandas_1.0s",
                "caffeine_20211229b",
            }:
                continue

            m = _TIME_RANGE_COMMA_PAT.match(line)
            if not m:
                continue

            t0 = _parse_hhmmss_to_seconds(m.group(1))
            t1 = _parse_hhmmss_to_seconds(m.group(2))
            if t1 <= t0:
                continue

            ranges.append(
                (
                    base_date + pd.Timedelta(seconds=t0),
                    base_date + pd.Timedelta(seconds=t1),
                )
            )
    return ranges


def _build_manual_mask(index: pd.DatetimeIndex, manual_ranges):
    mask = np.zeros(len(index), dtype=bool)
    if len(index) == 0:
        return mask
    for a, b in manual_ranges:
        mask |= ((index >= a) & (index < b))
    return mask


# =========================================================
# Prediction loading
# =========================================================
def _load_pred_csv(pred_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)

    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' column not found in {pred_csv}")

    if "pred_prob_rem" not in df.columns:
        raise ValueError(f"'pred_prob_rem' column not found in {pred_csv}")

    keep_cols = ["timestamp", "pred_prob_rem"]
    if "pred_label" in df.columns:
        keep_cols.append("pred_label")

    df = df[keep_cols]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp")
    df = _ensure_dtindex(df)

    df["pred_prob_rem"] = pd.to_numeric(df["pred_prob_rem"], errors="coerce")
    df = df.dropna(subset=["pred_prob_rem"])

    if "pred_label" not in df.columns:
        df["pred_label"] = (df["pred_prob_rem"] >= 0.5).astype(np.int8)
    else:
        df["pred_label"] = pd.to_numeric(df["pred_label"], errors="coerce").fillna(0).astype(np.int8)

    return df


# =========================================================
# Parquet schema helpers
# =========================================================
def _get_parquet_columns(parquet_path: Path) -> List[str]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    return list(pf.schema.names)


def _first_present(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _read_parquet_columns(parquet_path: Path, columns: List[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(parquet_path, columns=columns)
    except Exception:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path, columns=columns)
        return table.to_pandas()


def _load_sensor_parquet(
    parquet_path: Path,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    cols = _get_parquet_columns(parquet_path)

    ios_col = _first_present(cols, IOS_CANDIDATES)
    motion_col = _first_present(cols, MOTION_CANDIDATES)
    caffeine_col = _first_present(cols, CAFFEINE_CANDIDATES)
    time_col = _first_present(cols, TIME_CANDIDATES)

    if ios_col is None:
        raise RuntimeError(
            f"Could not find IOS/OIS signal column in parquet. Available columns include: {cols[:50]}"
        )
    if motion_col is None:
        raise RuntimeError(
            f"Could not find Motion Vector column in parquet. Available columns include: {cols[:50]}"
        )

    wanted = [ios_col, motion_col]
    if caffeine_col is not None and caffeine_col not in wanted:
        wanted.append(caffeine_col)
    if time_col is not None and time_col not in wanted:
        wanted.append(time_col)

    df = _read_parquet_columns(parquet_path, wanted)
    df = _ensure_dtindex(df)

    if start_ts is not None:
        df = df.loc[df.index >= pd.to_datetime(start_ts)]
    if end_ts is not None:
        df = df.loc[df.index <= pd.to_datetime(end_ts)]

    return df


def _pick_ios_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for c in IOS_CANDIDATES:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            return _drop_dup_index(s, keep="last").rename(c)
    return None


def _pick_motion_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for c in MOTION_CANDIDATES:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            return _drop_dup_index(s, keep="last").rename(c)
    return None


def _pick_caffeine_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for c in CAFFEINE_CANDIDATES:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            return _drop_dup_index(s, keep="last").rename(c)
    return None


def _compute_classical_rem_mask(
    sensor_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> tuple[np.ndarray, str]:
    if sensor_df.empty or len(target_index) == 0:
        return np.zeros(len(target_index), dtype=bool), "Classical REM skipped (empty sensor window)."

    result = None
    errors = []

    for candidate in [sensor_df.copy(), sensor_df]:
        try:
            result = extract_awake_rem_nrem(candidate)
            break
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")

    if result is None:
        msg = "Classical REM failed: " + " | ".join(errors[:2]) if errors else "Classical REM failed."
        return np.zeros(len(target_index), dtype=bool), msg

    mask_direct = _extract_mask_from_obj(result)
    if mask_direct is not None:
        if isinstance(result, pd.Series):
            source_index = result.index if isinstance(result.index, pd.DatetimeIndex) else sensor_df.index
            return _align_mask_to_index(mask_direct, pd.DatetimeIndex(source_index), target_index), "Classical REM enabled."
        if isinstance(result, pd.DataFrame):
            source_index = result.index if isinstance(result.index, pd.DatetimeIndex) else sensor_df.index
            return _align_mask_to_index(mask_direct, pd.DatetimeIndex(source_index), target_index), "Classical REM enabled."
        return _align_mask_to_index(mask_direct, sensor_df.index, target_index), "Classical REM enabled."

    if isinstance(result, dict):
        for key in ["rem", "REM", "rem_mask", "pred_rem"]:
            if key in result:
                submask = _extract_mask_from_obj(result[key])
                if submask is not None:
                    return _align_mask_to_index(submask, sensor_df.index, target_index), "Classical REM enabled."
        for key in ["rem_ranges", "REM_ranges", "rem_spans", "REM_spans"]:
            if key in result:
                mask = _mask_from_ranges(target_index, result[key])
                return mask.astype(bool), "Classical REM enabled."

    if isinstance(result, (list, tuple)):
        if len(result) >= 2:
            rem_obj = result[1]
            submask = _extract_mask_from_obj(rem_obj)
            if submask is not None:
                return _align_mask_to_index(submask, sensor_df.index, target_index), "Classical REM enabled."
            if isinstance(rem_obj, (list, tuple)):
                mask = _mask_from_ranges(target_index, rem_obj)
                if mask.any():
                    return mask.astype(bool), "Classical REM enabled."

    return np.zeros(len(target_index), dtype=bool), (
        "Classical REM returned an unrecognized format; overlay left off."
    )


# =========================================================
# Main app
# =========================================================
def build_app(
    pred_csv: Path,
    sensor_parquet: Path,
    manual_scores: Path,
    base_date: pd.Timestamp,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
    clip: bool = False,
):
    t0 = time.perf_counter()

    pred = _load_pred_csv(pred_csv)

    if clip and start_ts is not None:
        pred = pred.loc[pred.index >= pd.to_datetime(start_ts)]
    if clip and end_ts is not None:
        pred = pred.loc[pred.index <= pd.to_datetime(end_ts)]

    manual_ranges = _parse_manual_score_ranges(manual_scores, base_date)

    sensor = _load_sensor_parquet(
        sensor_parquet,
        start_ts=start_ts if clip else None,
        end_ts=end_ts if clip else None,
    )

    full_start = max(pred.index.min(), sensor.index.min())
    full_end = min(pred.index.max(), sensor.index.max())

    if start_ts is not None:
        full_start = max(full_start, pd.to_datetime(start_ts))
    if end_ts is not None:
        full_end = min(full_end, pd.to_datetime(end_ts))

    if full_end <= full_start:
        raise RuntimeError(
            f"Prediction CSV and sensor parquet have no overlapping time window. "
            f"Pred range: {pred.index.min()} -> {pred.index.max()} | "
            f"Sensor range: {sensor.index.min()} -> {sensor.index.max()}"
        )

    pred = pred.loc[(pred.index >= full_start) & (pred.index <= full_end)]
    sensor = sensor.loc[(sensor.index >= full_start) & (sensor.index <= full_end)]

    if pred.empty or sensor.empty:
        raise RuntimeError("No data remains after clipping.")

    ios_s = _pick_ios_series(sensor)
    motion_s = _pick_motion_series(sensor)
    caffeine_s = _pick_caffeine_series(sensor)

    if ios_s is None:
        raise RuntimeError("Could not find IOS/OIS signal column in parquet.")
    if motion_s is None:
        raise RuntimeError("Could not find Motion Vector column in parquet.")

    pred["manual_rem"] = _build_manual_mask(pred.index, manual_ranges).astype(bool)

    classical_mask, classical_status = _compute_classical_rem_mask(sensor, pred.index)
    pred["classical_rem"] = classical_mask.astype(bool)

    load_secs = time.perf_counter() - t0
    rng_full = (full_start, full_end)

    s_init = rng_full[0]
    e_init = min(rng_full[0] + pd.Timedelta(minutes=30), rng_full[1])

    daterange = pn.widgets.DatetimeRangeSlider(
        name="Time window",
        start=rng_full[0],
        end=rng_full[1],
        value=(s_init, e_init),
        step=1000,
        sizing_mode="stretch_width",
    )

    step_minutes = pn.widgets.IntInput(name="Step minutes", value=30, step=30, start=1, width=120)
    back_btn = pn.widgets.Button(name="- Step", button_type="warning", width=90)
    forward_btn = pn.widgets.Button(name="+ Step", button_type="warning", width=90)
    shrink_btn = pn.widgets.Button(name="- Window", button_type="primary", width=100)
    expand_btn = pn.widgets.Button(name="+ Window", button_type="primary", width=100)

    threshold = pn.widgets.FloatSlider(name="Threshold", start=0.0, end=1.0, step=0.01, value=0.50, width=180)
    show_prob_points = pn.widgets.Checkbox(name="Show p(REM) points", value=False)

    show_all3 = pn.widgets.Checkbox(name="■ All 3 agree", value=True)
    show_ml_man = pn.widgets.Checkbox(name="■ ML + Manual", value=True)
    show_ml_cla = pn.widgets.Checkbox(name="■ ML + Classical", value=True)
    show_man_cla = pn.widgets.Checkbox(name="■ Manual + Classical", value=True)
    show_ml_only = pn.widgets.Checkbox(name="■ ML only", value=True)
    show_man_only = pn.widgets.Checkbox(name="■ Manual only", value=True)
    show_cla_only = pn.widgets.Checkbox(name="■ Classical only", value=True)

    TOOLS = ["xpan", "xwheel_zoom", "reset", "crosshair"]

    def _clamp_window(start_new: pd.Timestamp, end_new: pd.Timestamp):
        full_start_, full_end_ = rng_full
        start_new = max(full_start_, pd.to_datetime(start_new))
        end_new = min(full_end_, pd.to_datetime(end_new))
        if end_new <= start_new:
            return None
        return (start_new.to_pydatetime(), end_new.to_pydatetime())

    def _shift_window(delta_minutes: int):
        cur_start, cur_end = daterange.value
        cur_start = pd.to_datetime(cur_start)
        cur_end = pd.to_datetime(cur_end)
        delta = pd.Timedelta(minutes=int(delta_minutes))

        width = cur_end - cur_start
        new_start = cur_start + delta
        new_end = cur_end + delta

        full_start_, full_end_ = rng_full

        if new_start < full_start_:
            new_start = full_start_
            new_end = new_start + width
        if new_end > full_end_:
            new_end = full_end_
            new_start = new_end - width

        clamped = _clamp_window(new_start, new_end)
        if clamped is not None:
            daterange.value = clamped

    def _resize_window(delta_minutes: int):
        cur_start, cur_end = daterange.value
        cur_start = pd.to_datetime(cur_start)
        cur_end = pd.to_datetime(cur_end)

        delta = pd.Timedelta(minutes=int(delta_minutes))
        new_end = cur_end + delta

        min_width = pd.Timedelta(minutes=1)
        if (new_end - cur_start) < min_width:
            new_end = cur_start + min_width

        clamped = _clamp_window(cur_start, new_end)
        if clamped is not None:
            daterange.value = clamped

    back_btn.on_click(lambda event: _shift_window(-int(step_minutes.value)))
    forward_btn.on_click(lambda event: _shift_window(int(step_minutes.value)))
    shrink_btn.on_click(lambda event: _resize_window(-int(step_minutes.value)))
    expand_btn.on_click(lambda event: _resize_window(int(step_minutes.value)))

    @pn.depends(
        daterange,
        threshold,
        show_prob_points,
        show_all3,
        show_ml_man,
        show_ml_cla,
        show_man_cla,
        show_ml_only,
        show_man_only,
        show_cla_only,
    )
    def synced_panels(
        rng_val,
        thr,
        _show_prob_points,
        _show_all3,
        _show_ml_man,
        _show_ml_cla,
        _show_man_cla,
        _show_ml_only,
        _show_man_only,
        _show_cla_only,
    ):
        t0w, t1w = rng_val
        t0w = pd.to_datetime(t0w)
        t1w = pd.to_datetime(t1w)

        pred_v = pred.loc[(pred.index >= t0w) & (pred.index <= t1w)].copy()
        sensor_v = sensor.loc[(sensor.index >= t0w) & (sensor.index <= t1w)]

        if pred_v.empty or sensor_v.empty:
            return pn.Column(pn.pane.Markdown("### No data in selected window."))

        pred_v["pred_thr"] = (pred_v["pred_prob_rem"] >= float(thr)).astype(np.int8)

        ios_v = ios_s.loc[(ios_s.index >= t0w) & (ios_s.index <= t1w)].dropna()
        motion_v = motion_s.loc[(motion_s.index >= t0w) & (motion_s.index <= t1w)].dropna()
        caffeine_v = None
        if caffeine_s is not None:
            caffeine_v = caffeine_s.loc[(caffeine_s.index >= t0w) & (caffeine_s.index <= t1w)].dropna()

        shade_overlay = _seven_state_overlay(
            pred_v,
            _show_all3,
            _show_ml_man,
            _show_ml_cla,
            _show_man_cla,
            _show_ml_only,
            _show_man_only,
            _show_cla_only,
        )

        ios_df = pd.DataFrame({"time": ios_v.index, "ios": ios_v.to_numpy()})
        ios_curve = ios_df.hvplot.line(
            x="time",
            y="ios",
            ylabel="IOS",
            xlabel="Time",
            title=f"OIS/IOS with REM spans — {t0w:%Y-%m-%d %H:%M:%S} → {t1w:%Y-%m-%d %H:%M:%S}",
            line_width=1.3,
            tools=TOOLS,
            responsive=True,
        )
        ios_panel = (shade_overlay * ios_curve).opts(
            shared_axes=True,
            frame_height=220,
            xformatter=DatetimeTickFormatter(hours="%H:%M", days="%m/%d %H:%M"),
        )

        motion_df = pd.DataFrame({"time": motion_v.index, "motion": motion_v.to_numpy()})
        motion_curve = motion_df.hvplot.line(
            x="time",
            y="motion",
            ylabel="Motion Ang. Vector",
            xlabel="Time",
            title="Motion",
            line_width=1.2,
            tools=TOOLS,
            responsive=True,
        )
        motion_panel = (shade_overlay * motion_curve).opts(
            shared_axes=True,
            frame_height=180,
            xformatter=DatetimeTickFormatter(hours="%H:%M", days="%m/%d %H:%M"),
        )

        caffeine_panel = None
        if caffeine_v is not None and len(caffeine_v) > 0:
            caffeine_df = pd.DataFrame({"time": caffeine_v.index, "caffeine": caffeine_v.to_numpy()})
            caff_curve = caffeine_df.hvplot.line(
                x="time",
                y="caffeine",
                ylabel="Caffeine",
                xlabel="Time",
                title="Caffeine",
                line_width=1.2,
                tools=TOOLS,
                responsive=True,
            )
            caffeine_panel = (shade_overlay * caff_curve).opts(
                shared_axes=True,
                frame_height=150,
                xformatter=DatetimeTickFormatter(hours="%H:%M", days="%m/%d %H:%M"),
            )

        prem_df = pd.DataFrame({
            "time": pred_v.index,
            "p_rem": pred_v["pred_prob_rem"].to_numpy(),
        })
        prem_curve = prem_df.hvplot.line(
            x="time",
            y="p_rem",
            ylabel="p(REM)",
            xlabel="Time",
            title="p(REM) + Threshold / REM shading",
            line_width=1.6,
            tools=TOOLS,
            responsive=True,
        )
        thr_line = hv.HLine(float(thr)).opts(color="#33B5FF", line_dash="dashed", line_width=2)

        prem_panel = shade_overlay * prem_curve * thr_line
        if bool(show_prob_points.value):
            prem_panel = prem_panel * prem_df.hvplot.scatter(
                x="time", y="p_rem", size=4, alpha=0.5, responsive=True
            )
        prem_panel = prem_panel.opts(
            shared_axes=True,
            frame_height=220,
            ylim=(-0.02, 1.02),
            xformatter=DatetimeTickFormatter(hours="%H:%M", days="%m/%d %H:%M"),
        )

        ml = pred_v["pred_thr"].astype(bool)
        man = pred_v["manual_rem"].astype(bool)
        cla = pred_v["classical_rem"].astype(bool)

        n_all3 = int((ml & man & cla).sum())
        n_ml_man = int((ml & man & ~cla).sum())
        n_ml_cla = int((ml & ~man & cla).sum())
        n_man_cla = int((~ml & man & cla).sum())
        n_ml_only = int((ml & ~man & ~cla).sum())
        n_man_only = int((~ml & man & ~cla).sum())
        n_cla_only = int((~ml & ~man & cla).sum())

        info = pn.pane.Markdown(
            f"**Data (aligned, clipped):** {rng_full[0]:%Y-%m-%d %H:%M:%S} → {rng_full[1]:%Y-%m-%d %H:%M:%S} "
            f"| Loaded in **{load_secs:.1f}s** "
            f"| Window: **{t0w:%Y-%m-%d %H:%M:%S} → {t1w:%Y-%m-%d %H:%M:%S}** "
            f"| Threshold: **{thr:.2f}**  \n"
            f"**Classical status:** {classical_status}  \n"
            f"💠 **All 3:** {n_all3} &nbsp;|&nbsp; "
            f"🟩 **ML+Manual:** {n_ml_man} &nbsp;|&nbsp; "
            f"🟪 **ML+Classical:** {n_ml_cla} &nbsp;|&nbsp; "
            f"🟧 **Manual+Classical:** {n_man_cla} &nbsp;|&nbsp; "
            f"🟦 **ML only:** {n_ml_only} &nbsp;|&nbsp; "
            f"🟨 **Manual only:** {n_man_only} &nbsp;|&nbsp; "
            f"🟥 **Classical only:** {n_cla_only}",
            sizing_mode="stretch_width",
        )

        panels = [ios_panel, motion_panel]
        if caffeine_panel is not None:
            panels.append(caffeine_panel)
        panels.append(prem_panel)

        return pn.Column(info, *panels, sizing_mode="stretch_both")

    header = pn.Row(
        pn.pane.Markdown("## Prediction / Manual / Classical Viewer", sizing_mode="stretch_width")
    )

    ctrls = pn.Row(
        daterange,
        step_minutes,
        back_btn,
        forward_btn,
        shrink_btn,
        expand_btn,
        threshold,
        show_prob_points,
        sizing_mode="stretch_width",
    )

    shade_ctrls_top = pn.Row(
        pn.pane.Markdown("**REM shading:**", width=110),
        show_all3,
        show_ml_man,
        show_ml_cla,
        show_man_cla,
        sizing_mode="stretch_width",
    )

    shade_ctrls_bottom = pn.Row(
        pn.pane.Markdown("", width=110),
        show_ml_only,
        show_man_only,
        show_cla_only,
        sizing_mode="stretch_width",
    )

    sidebar = pn.Column(
        "## Files",
        pn.pane.Markdown(f"**Pred CSV**  \n`{pred_csv}`"),
        pn.pane.Markdown(f"**Sensor parquet**  \n`{sensor_parquet}`"),
        pn.pane.Markdown(f"**Manual scores**  \n`{manual_scores}`"),
        pn.pane.Markdown(f"**Base date**  \n`{base_date}`"),
        pn.pane.Markdown(
            "**Panels:**\n"
            "- IOS/OIS\n"
            "- Motion\n"
            "- Caffeine *(if present)*\n"
            "- P(REM)\n\n"
            "**REM shading:**\n"
            "- 💠 All 3 agree\n"
            "- 🟩 ML + Manual only\n"
            "- 🟪 ML + Classical only\n"
            "- 🟧 Manual + Classical only\n"
            "- 🟦 ML only\n"
            "- 🟨 Manual only\n"
            "- 🟥 Classical only\n\n"
            "**Other:**\n"
            "- Dashed line = threshold\n"
            "- Shared time axis across all panels\n",
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )

    templ = pn.template.FastListTemplate(
        title="view_pred_vs_manual_classical_4panel",
        main=[pn.Column(header, ctrls, shade_ctrls_top, shade_ctrls_bottom, synced_panels, sizing_mode="stretch_both")],
        theme="dark",
        sidebar=[sidebar],
    )
    return templ


# =========================================================
# CLI
# =========================================================
def _parse_cli_once():
    ap = argparse.ArgumentParser(description="4-panel prediction vs manual vs classical REM viewer.")
    ap.add_argument("--pred_csv", required=True, help="Path to prediction CSV")
    ap.add_argument("--sensor_parquet", required=True, help="Path to sensor parquet")
    ap.add_argument("--manual_scores", required=True, help="Path to manual scores txt")
    ap.add_argument("--base_date", required=True, help="Base datetime for manual ranges, e.g. '2022-01-12 00:00:00'")
    ap.add_argument("--start", type=str, default=None, help="Optional clip start datetime")
    ap.add_argument("--end", type=str, default=None, help="Optional clip end datetime")
    ap.add_argument("--clip", action="store_true", help="Actually clip to [start,end]")
    try:
        args, _ = ap.parse_known_args()
    except SystemExit:
        args = argparse.Namespace(
            pred_csv=None,
            sensor_parquet=None,
            manual_scores=None,
            base_date=None,
            start=None,
            end=None,
            clip=False,
        )
    return args


_ARGS = _parse_cli_once()


def _error_template(msg: str, tb: str = ""):
    return pn.template.FastListTemplate(
        title="view_pred_vs_manual_classical_4panel",
        theme="dark",
        main=[
            pn.Column(
                pn.pane.Markdown("## Viewer startup error"),
                pn.pane.Markdown(msg),
                pn.pane.Markdown("### Traceback"),
                pn.pane.Markdown(f"```text\n{tb}\n```"),
                sizing_mode="stretch_both",
            )
        ],
    )


def app():
    if not _ARGS.pred_csv or not _ARGS.sensor_parquet or not _ARGS.manual_scores or not _ARGS.base_date:
        msg = pn.pane.Markdown(
            "### Missing required args\n\n"
            "Run with:\n\n"
            "`panel serve view_pred_vs_manual_classical_4panel.py --show --args "
            "--pred_csv PATH_TO_PRED.csv "
            "--sensor_parquet PATH_TO_PARQUET.pq "
            "--manual_scores PATH_TO_MANUAL.txt "
            "--base_date '2023-06-08 00:00:00' "
            "--start '2023-06-08 00:00:00' "
            "--end '2023-06-15 00:00:00' --clip`",
            sizing_mode="stretch_both",
        )
        return pn.template.FastListTemplate(
            title="view_pred_vs_manual_classical_4panel",
            main=[msg],
            theme="dark",
        )

    try:
        start_ts = pd.to_datetime(_ARGS.start) if _ARGS.start else None
        end_ts = pd.to_datetime(_ARGS.end) if _ARGS.end else None
        base_date = pd.to_datetime(_ARGS.base_date)

        return build_app(
            Path(_ARGS.pred_csv),
            Path(_ARGS.sensor_parquet),
            Path(_ARGS.manual_scores),
            base_date=base_date,
            start_ts=start_ts,
            end_ts=end_ts,
            clip=_ARGS.clip,
        )
    except Exception as e:
        return _error_template(
            f"**Error:** `{type(e).__name__}: {e}`",
            traceback.format_exc(),
        )


APP = app()
APP.servable("view_pred_vs_manual_classical_4panel")

if __name__ == "__main__":
    APP.show()
