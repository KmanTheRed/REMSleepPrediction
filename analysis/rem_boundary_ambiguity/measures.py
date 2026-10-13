"""
Per-episode measures: baselines, smoothing, and the three measure groups
from protocol.md Section 5.

  Group A (primary):    change-point interval width (onset/offset), seconds
  Group B (secondary):  noise-band crossing time (onset/offset), seconds
  Group C (explanatory): depth, descent/ascent 20-80%, min-time fraction,
                          rebound fraction, motion at offset

All functions here take already-sliced numpy arrays (a window's worth of
signal + its own time axis in seconds) -- window selection and clipping at
the trough minimum (protocol.md Section 5, change-log 2026-09-28) is done by
the caller (episodes.py), not here, so these stay independently testable.
"""
import numpy as np
import pandas as pd

from changepoint import changepoint_ci

MIN_WINDOW_MARGIN = 8  # seconds of data required on each side of a candidate tau


def moving_median(x, window):
    """Centered moving median, window in samples. Edge samples use a
    shrinking window (min_periods=1), vectorized via pandas for speed."""
    half = window // 2
    return (
        pd.Series(x)
        .rolling(window=2 * half + 1, center=True, min_periods=1)
        .median()
        .to_numpy()
    )


def baseline_stats(raw_window):
    """Median level and 1.4826*MAD-of-linearly-detrended-residuals SD, per
    protocol.md Section 4. raw_window is 1D raw (unsmoothed) signal."""
    n = len(raw_window)
    if n < 3:
        return float(np.median(raw_window)) if n else np.nan, np.nan
    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, raw_window, 1)
    resid = raw_window - np.polyval(coeffs, x)
    mad = np.median(np.abs(resid - np.median(resid)))
    sd = 1.4826 * mad
    level = float(np.median(raw_window))
    return level, float(sd)


def noise_band_onset(t, y, baseline_level, sd, lo_k=1.0, hi_k=3.0):
    """Time from leaving lo_k*SD of baseline to reaching hi_k*SD below
    baseline (departure direction = signal decreasing, i.e. OIS dipping).
    t, y are the local window's time axis (seconds, t=0 at boundary is NOT
    assumed) and smoothed signal. Returns np.nan if either crossing is not
    found in-window."""
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    dev = baseline_level - y  # positive = below baseline (a dip)
    leave_idx = np.argmax(dev > lo_k * sd) if np.any(dev > lo_k * sd) else None
    if leave_idx is None:
        return np.nan
    reach_mask = dev > hi_k * sd
    reach_mask[:leave_idx] = False
    if not np.any(reach_mask):
        return np.nan
    reach_idx = np.argmax(reach_mask)
    return float(t[reach_idx] - t[leave_idx])


def noise_band_offset(t, y, baseline_level, sd, lo_k=1.0, hi_k=3.0):
    """Time from the LAST sample at >= hi_k*SD below baseline back to the
    FIRST sample (after that) within lo_k*SD of baseline."""
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    dev = baseline_level - y
    deep_mask = dev >= hi_k * sd
    if not np.any(deep_mask):
        return np.nan
    last_deep_idx = np.where(deep_mask)[0][-1]
    near_mask = dev <= lo_k * sd
    near_mask[: last_deep_idx + 1] = False
    if not np.any(near_mask):
        return np.nan
    near_idx = np.argmax(near_mask)
    return float(t[near_idx] - t[last_deep_idx])


def crossing_time_onset(t, y, baseline_level, sd, k=3.0):
    """Absolute time (in t's own units) of the first sample departing more
    than k*SD below baseline_level. Used for 'core duration' (protocol.md
    Section 8): the 3 SD onset crossing. NaN if never reached."""
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    dev = baseline_level - y
    mask = dev > k * sd
    if not np.any(mask):
        return np.nan
    return float(t[np.argmax(mask)])


def crossing_time_offset(t, y, baseline_level, sd, k=3.0):
    """Absolute time of the LAST sample still >= k*SD below baseline_level
    (the 3 SD offset crossing for 'core duration'). NaN if never reached."""
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    dev = baseline_level - y
    mask = dev >= k * sd
    if not np.any(mask):
        return np.nan
    return float(t[np.where(mask)[0][-1]])


def morphology(t, y, pre_baseline_level, pre_sd, onset_t, offset_t):
    """Group C covariates computed over the full smoothed episode trace
    (t, y spanning [onset, offset] plus a little pad is fine -- callers pass
    what they have). depth is relative to pre-baseline."""
    if len(y) == 0:
        return dict(depth_pct=np.nan, depth_sd=np.nan, descent_20_80_s=np.nan,
                     ascent_20_80_s=np.nan, min_time_frac=np.nan, rebound_frac=np.nan)

    min_idx = int(np.argmin(y))
    min_val = y[min_idx]
    depth_abs = pre_baseline_level - min_val
    depth_pct = 100.0 * depth_abs / pre_baseline_level if pre_baseline_level else np.nan
    depth_sd = depth_abs / pre_sd if (pre_sd and np.isfinite(pre_sd) and pre_sd > 0) else np.nan

    duration = offset_t - onset_t
    min_time_frac = (t[min_idx] - onset_t) / duration if duration > 0 else np.nan

    # descent: 20%/80% of depth on the way down (before min_idx)
    def crossing_time(segment_t, segment_y, target, rising):
        dev = pre_baseline_level - segment_y  # positive = deeper
        mask = dev >= target if rising else dev <= target
        idx = np.argmax(mask) if np.any(mask) else None
        return segment_t[idx] if idx is not None else np.nan

    pre_t, pre_y = t[: min_idx + 1], y[: min_idx + 1]
    t20 = crossing_time(pre_t, pre_y, 0.2 * depth_abs, rising=True)
    t80 = crossing_time(pre_t, pre_y, 0.8 * depth_abs, rising=True)
    descent_20_80_s = (t80 - t20) if (np.isfinite(t20) and np.isfinite(t80)) else np.nan

    post_t, post_y = t[min_idx:], y[min_idx:]
    dev_post = pre_baseline_level - post_y
    mask80 = dev_post <= 0.8 * depth_abs
    mask20 = dev_post <= 0.2 * depth_abs
    ta80 = post_t[np.argmax(mask80)] if np.any(mask80) else np.nan
    ta20 = post_t[np.argmax(mask20)] if np.any(mask20) else np.nan
    ascent_20_80_s = (ta20 - ta80) if (np.isfinite(ta80) and np.isfinite(ta20)) else np.nan

    # rebound: a genuine mid-episode double-dip -- a rise from the trough
    # that is subsequently REVERSED by a redescent back into deep territory
    # (>=50% of depth), as opposed to the ordinary final ascent back to
    # baseline at the end of the episode, which is not a rebound and must
    # not count (an earlier version conflated the two: it tracked the
    # running minimum after the global trough and took the largest
    # subsequent rise, which is just "how much the signal recovers by
    # offset" and was >1 for most episodes -- fixed here by requiring the
    # rise to be followed by a redescent, not just measuring net recovery).
    rebound_frac = 0.0
    seg = y[min_idx:]
    n_seg = len(seg)
    i = 0
    while i < n_seg - 1:
        j = i
        while j + 1 < n_seg and seg[j + 1] >= seg[j]:
            j += 1
        if j == i:
            i += 1
            continue
        peak_rise = seg[j] - seg[i]
        k = j
        while k + 1 < n_seg and seg[k + 1] <= seg[k]:
            k += 1
        if k > j and depth_abs > 0:
            dev_at_redescent = pre_baseline_level - seg[k]
            if dev_at_redescent >= 0.5 * depth_abs:
                rebound_frac = max(rebound_frac, peak_rise / depth_abs)
        i = k if k > j else j + 1

    return dict(depth_pct=depth_pct, depth_sd=depth_sd, descent_20_80_s=descent_20_80_s,
                ascent_20_80_s=ascent_20_80_s, min_time_frac=min_time_frac,
                rebound_frac=rebound_frac)


def changepoint_measure(t_local, y_local, kind, block_len, n_boot, seed):
    """Group A wrapper: runs the changepoint CI on an already-clipped local
    window and returns the width plus the raw fit for diagnostics."""
    n = len(t_local)
    if n < 2 * MIN_WINDOW_MARGIN + 4:
        return dict(tau=np.nan, ci_lo=np.nan, ci_hi=np.nan, width=np.nan)
    tau_grid = np.arange(t_local.min() + MIN_WINDOW_MARGIN,
                          t_local.max() - MIN_WINDOW_MARGIN + 1)
    if len(tau_grid) < 2:
        return dict(tau=np.nan, ci_lo=np.nan, ci_hi=np.nan, width=np.nan)
    return changepoint_ci(t_local, y_local, kind, tau_grid, block_len=block_len,
                           n_boot=n_boot, seed=seed)
