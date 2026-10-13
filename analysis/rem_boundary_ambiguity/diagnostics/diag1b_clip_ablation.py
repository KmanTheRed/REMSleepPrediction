"""
Follow-up to diag1: isolates whether trough-minimum window clipping is the
source of the spurious duration correlation found in the main invariance
test, by rerunning the identical synthetic episodes with clipping disabled
(fixed +-45s window regardless of duration/shape).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measures import moving_median
from run_pipeline import PRIMARY_SMOOTH, CP_WINDOW
from diag1_synthetic_invariance import simulate_flat_floor, simulate_ramp, fit_boundary, OUT_DIR


def run(durations=range(20, 251, 15), n_reps=20, seed0=0):
    rows = []
    rng_master = np.random.default_rng(seed0)
    for shape, sim_fn in [("flat_floor", simulate_flat_floor), ("ramp", simulate_ramp)]:
        for duration in durations:
            pad = CP_WINDOW + 5
            for rep in range(n_reps):
                rng = np.random.default_rng(rng_master.integers(0, 2**31))
                t, y = sim_fn(duration, pad, rng=rng)
                y_smooth_full = moving_median(y, PRIMARY_SMOOTH)
                in_ep = (t >= 0) & (t <= duration)
                time_of_min = float(t[in_ep][int(np.argmin(y_smooth_full[in_ep]))])

                onset_w = fit_boundary(t, y, 0.0, "rising_after", PRIMARY_SMOOTH, time_of_min,
                                        seed=rng.integers(0, 2**31), clip_at_min=False)
                offset_w = fit_boundary(t, y, duration, "flat_after", PRIMARY_SMOOTH, time_of_min,
                                         seed=rng.integers(0, 2**31), clip_at_min=False)
                rows.append(dict(shape=shape, duration=duration, rep=rep,
                                  onset_ambiguity_s=onset_w, offset_ambiguity_s=offset_w))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    df.to_csv(os.path.join(OUT_DIR, "diagnostics", "synthetic_invariance_no_clip.csv"), index=False)
    print(df.shape)
    print("\n--- Diagnostic 1b: same synthetic data, clipping DISABLED (fixed +-45s window) ---")
    for shape, grp in df.groupby("shape"):
        for measure in ["onset_ambiguity_s", "offset_ambiguity_s"]:
            clean = grp[["duration", measure]].replace([np.inf, -np.inf], np.nan).dropna()
            rho, p = spearmanr(clean["duration"], clean[measure])
            print(f"  {shape:10s} {measure:20s} rho={rho:+.3f}  p={p:.4g}  n={len(clean)}  "
                  f"median_width={clean[measure].median():.2f}s")
