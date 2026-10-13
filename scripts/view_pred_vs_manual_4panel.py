#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from sleep_detect_2 import extract_awake_rem_nrem
import argparse
import re
import time
import warnings
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

import panel as pn
import holoviews as hv
import hvplot.pandas  # noqa: F401
from bokeh.models import DatetimeTickFormatter

warnings.filterwarnings("ignore")
hv.extension("bokeh")
pn.extension("tabulator", notifications=True)
pn.config.theme = "dark"

# =========================================================
# Shading colour palette
# =========================================================
SHADE_BOTH = {"color": "#2BC4C4", "alpha": 0.40}
SHADE_ML_ONLY = {"color": "#6A7FDB", "alpha": 0.32}
SHADE_MAN_ONLY = {"color": "#E8A838", "alpha": 0.28}

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


# =========================================================
# Helpers
# =========================================================
def _drop_dup_index(obj, keep: str = "last"):
    if isinstance(obj, (pd.Series, pd.DataFrame)) and not obj.index.is_unique:
        obj = obj[~obj.index.duplicated(keep=keep)]
    return obj


def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make dataframe use a DatetimeIndex, using:
      1) existing index if already datetime-like
      2) known time columns if present
      3) otherwise try parsing the existing index

    IMPORTANT:
      - prefer real time columns like t_idx over a RangeIndex / integer index
      - otherwise 0,1,2,... can be misread as nanoseconds from epoch
    """
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
            # Only fall back to parsing the existing index if it is not just numeric
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


# =========================================================
# Three-way agreement overlay
# =========================================================
def _three_way_overlay(pred_win: pd.DataFrame, show_both: bool, show_ml: bool, show_man: bool) -> hv.Overlay:
    idx = pred_win.index
    ml_mask = pred_win["pred_thr"].astype(bool).to_numpy()
    man_mask = pred_win["manual_rem"].astype(bool).to_numpy()

    both_mask = ml_mask & man_mask
    ml_only_mask = ml_mask & ~man_mask
    man_only_mask = ~ml_mask & man_mask

    all_spans = hv.Overlay([])

    if show_man:
        man_spans = _spans_from_mask(idx, man_only_mask)
        all_spans = all_spans * _vspan_overlay(man_spans, **SHADE_MAN_ONLY)

    if show_ml:
        ml_spans = _spans_from_mask(idx, ml_only_mask)
        all_spans = all_spans * _vspan_overlay(ml_spans, **SHADE_ML_ONLY)

    if show_both:
        both_spans = _spans_from_mask(idx, both_mask)
        all_spans = all_spans * _vspan_overlay(both_spans, **SHADE_BOTH)

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
    """
    Read only requested columns, robustly.
    Uses pandas first, then pyarrow if needed.
    """
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
    """
    Robust parquet loader:
    - reads only needed signal columns
    - works whether time is stored as:
        * a normal column
        * a hidden parquet index column
        * t_idx
    """
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

    shade_both = pn.widgets.Checkbox(name="■ Both (teal)", value=True)
    shade_ml_only = pn.widgets.Checkbox(name="■ ML only (blue)", value=True)
    shade_man_only = pn.widgets.Checkbox(name="■ Manual only (amber)", value=True)

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

    @pn.depends(daterange, threshold, show_prob_points, shade_both, shade_ml_only, shade_man_only)
    def synced_panels(rng_val, thr, _show_prob_points, _shade_both, _shade_ml, _shade_man):
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

        shade_overlay = _three_way_overlay(pred_v, _shade_both, _shade_ml, _shade_man)

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
            title="p(REM) + Threshold / Episode shading",
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

        n_both = int((pred_v["pred_thr"].astype(bool) & pred_v["manual_rem"].astype(bool)).sum())
        n_ml_only = int((pred_v["pred_thr"].astype(bool) & ~pred_v["manual_rem"].astype(bool)).sum())
        n_man_only = int((~pred_v["pred_thr"].astype(bool) & pred_v["manual_rem"].astype(bool)).sum())

        info = pn.pane.Markdown(
            f"**Data (aligned, clipped):** {rng_full[0]:%Y-%m-%d %H:%M:%S} → {rng_full[1]:%Y-%m-%d %H:%M:%S} "
            f"| Loaded in **{load_secs:.1f}s** "
            f"| Window: **{t0w:%Y-%m-%d %H:%M:%S} → {t1w:%Y-%m-%d %H:%M:%S}** "
            f"| Threshold: **{thr:.2f}**  \n"
            f"🟦 **Both (teal):** {n_both} windows &nbsp;|&nbsp; "
            f"🟪 **ML only (blue):** {n_ml_only} windows &nbsp;|&nbsp; "
            f"🟨 **Manual only (amber):** {n_man_only} windows",
            sizing_mode="stretch_width",
        )

        panels = [ios_panel, motion_panel]
        if caffeine_panel is not None:
            panels.append(caffeine_panel)
        panels.append(prem_panel)

        return pn.Column(info, *panels, sizing_mode="stretch_both")

    header = pn.Row(
        pn.pane.Markdown("## Prediction / Manual Viewer (4-panel)", sizing_mode="stretch_width")
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

    shade_ctrls = pn.Row(
        pn.pane.Markdown("**REM shading:**", width=110),
        shade_both,
        shade_ml_only,
        shade_man_only,
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
            "- 🟦 Teal = both agree (REM)\n"
            "- 🟪 Blue = ML only predicts REM\n"
            "- 🟨 Amber = manual only labels REM\n\n"
            "**Other:**\n"
            "- Dashed line = threshold\n"
            "- Shared time axis across all panels\n",
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )

    templ = pn.template.FastListTemplate(
        title="view_pred_vs_manual_4panel",
        main=[pn.Column(header, ctrls, shade_ctrls, synced_panels, sizing_mode="stretch_both")],
        theme="dark",
        sidebar=[sidebar],
    )
    return templ


# =========================================================
# CLI
# =========================================================
def _parse_cli_once():
    ap = argparse.ArgumentParser(description="4-panel prediction vs manual REM viewer.")
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
        title="view_pred_vs_manual_4panel",
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
            "`panel serve view_pred_vs_manual_4panel.py --show --args "
            "--pred_csv PATH_TO_PRED.csv "
            "--sensor_parquet PATH_TO_PARQUET.pq "
            "--manual_scores PATH_TO_MANUAL.txt "
            "--base_date '2022-01-12 00:00:00' "
            "--start '2022-01-12 00:00:00' "
            "--end '2022-01-19 00:00:00' --clip`",
            sizing_mode="stretch_both",
        )
        return pn.template.FastListTemplate(
            title="view_pred_vs_manual_4panel",
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
APP.servable("view_pred_vs_manual_4panel")

if __name__ == "__main__":
    APP.show()