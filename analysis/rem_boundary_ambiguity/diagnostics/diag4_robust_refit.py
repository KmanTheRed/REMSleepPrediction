"""
Diagnostic #4 (requested 2026-09-28): refit with
  (a) a FIXED +-60s window per boundary, independent of duration/time-of-min,
  (b) a model allowing continued decline after departure (sloped floor,
      changepoint.grid_search_tau_slopedfloor: ramp at s1 for w0=15s, then
      continue at s2 -- absorbs the exact V-shape violation that motivated
      the original trough-minimum clipping, without needing to shrink the
      window for short episodes),
  (c) block bootstrap with a longer block length than the original 15s --
      diag_fit_and_truncation.py measured residual lag-1 autocorrelation of
      0.87-0.92 on real data, far higher than the 3x-smoothing=15s heuristic
      was calibrated for; block_len=45 here,
  (d) motion (Motion Ang. Vector, z-scored per-window) as an extra covariate
      at offset only, per the request ("motion as a covariate at offset").

Run on BOTH the synthetic invariance data (does the artifact survive this
refit?) and a real-data sample stratified by duration quartile (does the
real duration effect survive?).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io_utils as io
from episodes import build_episodes
from measures import moving_median
from changepoint import changepoint_ci_slopedfloor
from run_pipeline import _extract_window, PRIMARY_SMOOTH, PRE_BASELINE, POST_BASELINE

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
FIXED_WINDOW = 60.0
W0 = 15.0
BLOCK_LEN = 45  # informed by measured AR1 ~0.9, not the original 3*smoothing=15
N_BOOT = 150
N_PER_QUARTILE = 150  # real-data subsample size per duration quartile (compute budget)


def fit_fixed_window(t_full, y_full_raw, boundary_t, kind, smoothing, seed, motion_full=None):
    y_full = moving_median(y_full_raw, smoothing)
    mask = (t_full >= boundary_t - FIXED_WINDOW) & (t_full <= boundary_t + FIXED_WINDOW)
    t_rel = t_full[mask] - boundary_t
    y_local = y_full[mask]
    motion_local = None
    if motion_full is not None:
        m = motion_full[mask]
        sd = m.std()
        motion_local = (m - m.mean()) / sd if sd > 0 else np.zeros_like(m)
    if len(t_rel) < 40:
        return None
    tau_grid = np.arange(t_rel.min() + 8, t_rel.max() - 8 + 1)
    if len(tau_grid) < 2:
        return None
    return changepoint_ci_slopedfloor(t_rel, y_local, kind, tau_grid, w0=W0, motion=motion_local,
                                       block_len=BLOCK_LEN, n_boot=N_BOOT, seed=seed)


def run_on_real(animals=("A1", "A3", "A8", "A11"), n_per_quartile=N_PER_QUARTILE, seed=0):
    per_episode = pd.read_csv(os.path.join(OUT_DIR, "per_episode.csv"))
    per_episode["duration_q"] = pd.qcut(per_episode["episode_duration_s"], 4)
    rng = np.random.default_rng(seed)
    sample = (
        per_episode.groupby("duration_q", observed=True, group_keys=False)
        .apply(lambda g: g.sample(n=min(n_per_quartile, len(g)), random_state=seed))
    )
    recs = {a: io.load_animal(a)[0] for a in sample["animal"].unique()}

    rows = []
    for _, row in sample.iterrows():
        rec = recs[row["animal"]]
        onset, offset = row["onset"], row["offset"]
        t_full, y_full_raw = _extract_window(rec, onset - FIXED_WINDOW - 5, offset + FIXED_WINDOW + 5)
        motion_i0 = io.sec_to_row(rec, onset - FIXED_WINDOW - 5)
        motion_i1 = io.sec_to_row(rec, offset + FIXED_WINDOW + 5)
        motion_full = rec["motion"][motion_i0:motion_i1 + 1]
        if len(motion_full) != len(t_full):
            motion_full = None  # length mismatch guard (edge of recording)

        seed_i = hash((row["animal"], row["bout_idx"])) % (2 ** 31)
        r_on = fit_fixed_window(t_full, y_full_raw, onset, "rising_after", PRIMARY_SMOOTH, seed_i)
        r_off = fit_fixed_window(t_full, y_full_raw, offset, "flat_after", PRIMARY_SMOOTH, seed_i + 1,
                                  motion_full=motion_full)
        if r_on is None or r_off is None:
            continue
        rows.append(dict(animal=row["animal"], bout_idx=row["bout_idx"],
                          episode_duration_s=row["episode_duration_s"], caffeine=row["caffeine"],
                          onset_ambiguity_s=r_on["width"], offset_ambiguity_s=r_off["width"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_on_real()
    os.makedirs(os.path.join(OUT_DIR, "diagnostics"), exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "diagnostics", "robust_refit_real.csv"), index=False)
    print("real-data refit sample:", df.shape)

    print("\n--- Diagnostic 4 (real data): does the duration effect persist under the robust refit? ---")
    for measure in ["onset_ambiguity_s", "offset_ambiguity_s"]:
        clean = df[["episode_duration_s", measure]].replace([np.inf, -np.inf], np.nan).dropna()
        rho, p = spearmanr(clean["episode_duration_s"], clean[measure])
        print(f"  {measure:20s} rho={rho:+.3f}  p={p:.4g}  n={len(clean)}")

    df["duration_q"] = pd.qcut(df["episode_duration_s"], 4)
    print(df.groupby("duration_q", observed=True)[["onset_ambiguity_s", "offset_ambiguity_s"]]
          .median().to_string())
