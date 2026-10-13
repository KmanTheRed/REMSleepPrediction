"""
Diagnostics #2 and #3 (requested 2026-09-28, see CHANGELOG.md): for every real
kept episode, record what the primary change-point fit actually looks like --
not just the summary width, but the fitted slope, residual SD, residual
lag-1 autocorrelation, pre-baseline slope, and whether the fit or its CI
bounds sit at the edge of the search window (truncation).

Reuses the exact windowing/clipping used by run_pipeline.process_episode
(same CP_WINDOW, same clip-at-time_of_min, same smoothing) so this describes
the fit that actually produced per_episode.csv, not an approximation of it.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io_utils as io
from episodes import build_episodes
from measures import baseline_stats, moving_median, MIN_WINDOW_MARGIN
from changepoint import grid_search_tau, moving_block_bootstrap_resid
from run_pipeline import _extract_window, CP_WINDOW, PRE_BASELINE, POST_BASELINE, PRIMARY_SMOOTH

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
EDGE_TOL = 1.0  # seconds: within this of a grid bound counts as "at the bound"


def _lag1_autocorr(x):
    if len(x) < 4:
        return np.nan
    x = x - x.mean()
    num = np.sum(x[:-1] * x[1:])
    den = np.sum(x ** 2)
    return float(num / den) if den > 0 else np.nan


def _fit_with_diagnostics(t_local, y_local, kind, smoothing, seed, n_boot=200):
    """One grid_search_tau call for the point estimate (kept, so its resid is
    reused for both the AR1 diagnostic and the bootstrap -- no duplicate fit)."""
    n = len(t_local)
    if n < 2 * MIN_WINDOW_MARGIN + 4:
        return None
    tau_grid = np.arange(t_local.min() + MIN_WINDOW_MARGIN, t_local.max() - MIN_WINDOW_MARGIN + 1)
    if len(tau_grid) < 2:
        return None

    tau_hat, beta, fitted, resid, sse = grid_search_tau(t_local, y_local, kind, tau_grid)
    resid_c = resid - resid.mean()
    block_len = 3 * smoothing
    rng = np.random.default_rng(seed)
    taus_boot = np.empty(n_boot)
    for b in range(n_boot):
        boot_resid = moving_block_bootstrap_resid(resid_c, block_len, len(t_local), rng)
        tau_b, *_ = grid_search_tau(t_local, fitted + boot_resid, kind, tau_grid)
        taus_boot[b] = tau_b
    ci_lo, ci_hi = np.percentile(taus_boot, [2.5, 97.5])

    grid_lo, grid_hi = float(tau_grid.min()), float(tau_grid.max())
    return dict(
        tau=tau_hat, ci_lo=float(ci_lo), ci_hi=float(ci_hi), width=float(ci_hi - ci_lo),
        slope=beta[1], resid_sd=float(np.std(resid)), resid_ar1=_lag1_autocorr(resid),
        window_t_min=float(t_local.min()), window_t_max=float(t_local.max()),
        tau_grid_lo=grid_lo, tau_grid_hi=grid_hi,
        tau_at_lo_bound=abs(tau_hat - grid_lo) <= EDGE_TOL,
        tau_at_hi_bound=abs(tau_hat - grid_hi) <= EDGE_TOL,
        ci_lo_at_bound=abs(ci_lo - grid_lo) <= EDGE_TOL,
        ci_hi_at_bound=abs(ci_hi - grid_hi) <= EDGE_TOL,
    )


def run(animals=("A1", "A3", "A8", "A11"), smoothing=PRIMARY_SMOOTH):
    rows = []
    for animal in animals:
        rec, bouts, skipped = io.load_animal(animal)
        duration_s = rec["elapsed_sec"][-1]
        eps, _ = build_episodes(animal, bouts, rec, duration_s)
        for ep in eps:
            onset, offset = ep.onset, ep.offset
            _, pre_raw = _extract_window(rec, onset + PRE_BASELINE[0], onset + PRE_BASELINE[1])
            _, post_raw = _extract_window(rec, offset + POST_BASELINE[0], offset + POST_BASELINE[1])
            pre_slope = float(np.polyfit(np.arange(len(pre_raw)), pre_raw, 1)[0])
            post_slope = float(np.polyfit(np.arange(len(post_raw)), post_raw, 1)[0])

            t_full, y_full_raw = _extract_window(rec, onset - CP_WINDOW, offset + CP_WINDOW)
            y_full = moving_median(y_full_raw, smoothing)
            in_ep = (t_full >= onset) & (t_full <= offset)
            if not np.any(in_ep):
                continue
            time_of_min = float(t_full[in_ep][int(np.argmin(y_full[in_ep]))])

            onset_hi = min(onset + CP_WINDOW, time_of_min)
            on_mask = (t_full >= onset - CP_WINDOW) & (t_full <= onset_hi)
            t_on_rel = t_full[on_mask] - onset
            y_on = y_full[on_mask]

            offset_lo = max(offset - CP_WINDOW, time_of_min)
            off_mask = (t_full >= offset_lo) & (t_full <= offset + CP_WINDOW)
            t_off_rel = t_full[off_mask] - offset
            y_off = y_full[off_mask]

            seed = hash((animal, ep.bout_idx)) % (2 ** 31)
            fit_on = _fit_with_diagnostics(t_on_rel, y_on, "rising_after", smoothing, seed)
            fit_off = _fit_with_diagnostics(t_off_rel, y_off, "flat_after", smoothing, seed + 1)
            if fit_on is None or fit_off is None:
                continue

            row = dict(animal=animal, bout_idx=ep.bout_idx, episode_duration_s=offset - onset,
                       caffeine=ep.caffeine, pre_baseline_slope=pre_slope, post_baseline_slope=post_slope)
            row.update({f"onset_{k}": v for k, v in fit_on.items()})
            row.update({f"offset_{k}": v for k, v in fit_off.items()})
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    os.makedirs(os.path.join(OUT_DIR, "diagnostics"), exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "diagnostics", "fit_and_truncation.csv"), index=False)
    print(df.shape)

    print("\n--- Diagnostic 2: truncation fractions ---")
    for b in ["onset", "offset"]:
        for col in ["tau_at_lo_bound", "tau_at_hi_bound", "ci_lo_at_bound", "ci_hi_at_bound"]:
            frac = df[f"{b}_{col}"].mean()
            print(f"  {b}_{col}: {frac:.3f} ({int(df[f'{b}_{col}'].sum())}/{len(df)})")

    print("\n--- Diagnostic 3: fitted slope / noise / AR1 / baseline slope by duration quartile ---")
    df["duration_q"] = pd.qcut(df["episode_duration_s"], 4)
    cols = ["episode_duration_s", "onset_slope", "onset_resid_sd", "onset_resid_ar1",
            "offset_slope", "offset_resid_sd", "offset_resid_ar1",
            "pre_baseline_slope", "post_baseline_slope",
            "onset_width", "offset_width"]
    summary = df.groupby("duration_q", observed=True)[cols].median()
    pd.set_option("display.width", 200)
    print(summary.to_string())
    summary.to_csv(os.path.join(OUT_DIR, "diagnostics", "slope_noise_ar1_by_quartile.csv"))
