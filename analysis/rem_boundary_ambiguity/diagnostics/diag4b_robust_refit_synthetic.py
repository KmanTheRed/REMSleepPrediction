"""
Applies the diag4 robust refit (fixed +-60s window, sloped-floor model,
block_len=45 bootstrap) to the SAME synthetic invariance episodes used in
diag1 -- the decisive test of whether this refit actually removes the
spurious duration correlation the original method produced under true
constant ambiguity, or whether it persists.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measures import moving_median
from run_pipeline import PRIMARY_SMOOTH
from changepoint import changepoint_ci_slopedfloor
from diag1_synthetic_invariance import simulate_flat_floor, simulate_ramp, OUT_DIR

FIXED_WINDOW = 60.0
W0 = 15.0
BLOCK_LEN = 45
N_BOOT = 100


def fit_fixed_window(t_full, y_full_raw, boundary_t, kind, smoothing, seed):
    y_full = moving_median(y_full_raw, smoothing)
    mask = (t_full >= boundary_t - FIXED_WINDOW) & (t_full <= boundary_t + FIXED_WINDOW)
    t_rel = t_full[mask] - boundary_t
    y_local = y_full[mask]
    if len(t_rel) < 40:
        return np.nan
    tau_grid = np.arange(t_rel.min() + 8, t_rel.max() - 8 + 1)
    if len(tau_grid) < 2:
        return np.nan
    result = changepoint_ci_slopedfloor(t_rel, y_local, kind, tau_grid, w0=W0,
                                         block_len=BLOCK_LEN, n_boot=N_BOOT, seed=seed)
    return result["width"]


def run(durations=range(20, 251, 15), n_reps=20, seed0=0):
    rows = []
    rng_master = np.random.default_rng(seed0)
    for shape, sim_fn in [("flat_floor", simulate_flat_floor), ("ramp", simulate_ramp)]:
        for duration in durations:
            pad = FIXED_WINDOW + 5
            for rep in range(n_reps):
                rng = np.random.default_rng(rng_master.integers(0, 2**31))
                t, y = sim_fn(duration, pad, rng=rng)
                onset_w = fit_fixed_window(t, y, 0.0, "rising_after", PRIMARY_SMOOTH,
                                            seed=rng.integers(0, 2**31))
                offset_w = fit_fixed_window(t, y, duration, "flat_after", PRIMARY_SMOOTH,
                                             seed=rng.integers(0, 2**31))
                rows.append(dict(shape=shape, duration=duration, rep=rep,
                                  onset_ambiguity_s=onset_w, offset_ambiguity_s=offset_w))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    df.to_csv(os.path.join(OUT_DIR, "diagnostics", "robust_refit_synthetic.csv"), index=False)
    print(df.shape)
    print("\n--- Diagnostic 4 (synthetic): does the artifact survive the robust refit? ---")
    for shape, grp in df.groupby("shape"):
        for measure in ["onset_ambiguity_s", "offset_ambiguity_s"]:
            clean = grp[["duration", measure]].replace([np.inf, -np.inf], np.nan).dropna()
            rho, p = spearmanr(clean["duration"], clean[measure])
            print(f"  {shape:10s} {measure:20s} rho={rho:+.3f}  p={p:.4g}  n={len(clean)}  "
                  f"median_width={clean[measure].median():.2f}s")
