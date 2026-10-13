"""
Diagnostic #1 (requested 2026-09-28): synthetic invariance test.

Simulate troughs with FIXED onset/offset transition slope, noise SD, and
AR(1) autocorrelation -- matched to real values measured by
diag_fit_and_truncation.py (median onset slope -2.2e-5, offset slope
-4.9e-5, noise SD ~1e-4, AR1 ~0.88-0.89) -- while episode duration varies
20-250 s. True ambiguity is constant by construction (the local dynamics at
each boundary never change). Run the EXACT real pipeline (same CP_WINDOW,
same clip-at-time_of_min, same smoothing, same bootstrap) on the synthetic
traces and check whether CI width still correlates with duration -- if it
does, that correlation is an artifact of the method, not a real duration
effect, and the primary result would need to be discounted accordingly.

Two shapes, both holding the LOCAL slope/noise/AR1 at each boundary fixed
regardless of total duration:
  - flat_floor: fixed-duration (15 s) ramp down, flat AR(1)-noise floor for
    the remainder, fixed-duration (15 s) ramp back up. Only the flat middle
    stretches with duration -- this is the shape for which "ambiguity should
    be exactly duration-invariant" is least ambiguous to construct.
  - ramp: a symmetric V (no flat floor) -- descent at the fixed slope for
    duration/2, then ascent at the fixed slope for duration/2. This is the
    adversarial case for the trough-minimum window clipping: short-duration
    ramps clip the onset/offset windows hardest.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measures import MIN_WINDOW_MARGIN, moving_median
from changepoint import changepoint_ci
from run_pipeline import CP_WINDOW, PRIMARY_SMOOTH

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# Real measured medians (diag_fit_and_truncation.py, 2026-09-28 run):
ONSET_SLOPE = -0.000022
OFFSET_SLOPE = -0.000049
NOISE_SD = 0.0001
AR1 = 0.88
RAMP_DURATION = 15.0  # seconds, matches real median descent/ascent 20-80% times


def ar1_noise(n, sd, rho, rng):
    """Generate AR(1) noise with the given marginal SD and lag-1 rho."""
    innovation_sd = sd * np.sqrt(1 - rho ** 2)
    e = rng.normal(0, innovation_sd, size=n)
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def simulate_flat_floor(duration, pad, fs=1.0, rng=None):
    """onset at t=0, offset at t=duration, in a window from -pad to duration+pad."""
    t = np.arange(-pad, duration + pad + 1, 1.0 / fs)
    y = np.zeros_like(t)
    depth = abs(ONSET_SLOPE) * RAMP_DURATION
    for i, ti in enumerate(t):
        if ti < 0:
            y[i] = 0.0
        elif ti < RAMP_DURATION:
            y[i] = ONSET_SLOPE * ti
        elif ti < duration - RAMP_DURATION:
            y[i] = -depth
        elif ti < duration:
            frac = (ti - (duration - RAMP_DURATION)) / RAMP_DURATION
            y[i] = -depth + depth * frac  # symmetric ascent back to baseline
        else:
            y[i] = 0.0
    y = y + ar1_noise(len(t), NOISE_SD, AR1, rng)
    return t, y


def simulate_ramp(duration, pad, fs=1.0, rng=None):
    """Symmetric V: no flat floor. Onset at t=0, offset at t=duration,
    apex (trough bottom) at duration/2. Local slope at EACH boundary is
    fixed at ONSET_SLOPE / OFFSET_SLOPE regardless of duration -- only how
    long the ramp continues before reversing changes."""
    t = np.arange(-pad, duration + pad + 1, 1.0 / fs)
    apex = duration / 2.0
    y = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < 0:
            y[i] = 0.0
        elif ti < apex:
            y[i] = ONSET_SLOPE * ti
        elif ti < duration:
            depth_at_apex = ONSET_SLOPE * apex
            y[i] = depth_at_apex - OFFSET_SLOPE * (ti - apex)  # ascend back at offset's slope
        else:
            y[i] = 0.0
    y = y + ar1_noise(len(t), NOISE_SD, AR1, rng)
    return t, y


def fit_boundary(t_full, y_full, boundary_t, kind, smoothing, time_of_min, seed, n_boot=100,
                  clip_at_min=True):
    """Replicates run_pipeline.process_episode's window construction exactly:
    +-CP_WINDOW around the boundary, clipped at time_of_min, primary
    smoothing, then changepoint_ci. clip_at_min=False uses a fixed +-CP_WINDOW
    window regardless of duration/shape, to isolate whether the clipping
    itself is the source of any duration artifact."""
    y_smooth = moving_median(y_full, smoothing)
    if kind == "rising_after":
        hi = min(boundary_t + CP_WINDOW, time_of_min) if clip_at_min else boundary_t + CP_WINDOW
        mask = (t_full >= boundary_t - CP_WINDOW) & (t_full <= hi)
    else:
        lo = max(boundary_t - CP_WINDOW, time_of_min) if clip_at_min else boundary_t - CP_WINDOW
        mask = (t_full >= lo) & (t_full <= boundary_t + CP_WINDOW)
    t_rel = t_full[mask] - boundary_t
    y_local = y_smooth[mask]
    if len(t_rel) < 2 * MIN_WINDOW_MARGIN + 4:
        return np.nan
    tau_grid = np.arange(t_rel.min() + MIN_WINDOW_MARGIN, t_rel.max() - MIN_WINDOW_MARGIN + 1)
    if len(tau_grid) < 2:
        return np.nan
    result = changepoint_ci(t_rel, y_local, kind, tau_grid, block_len=3 * smoothing,
                             n_boot=n_boot, seed=seed)
    return result["width"]


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

                onset_w = fit_boundary(t, y, 0.0, "rising_after", PRIMARY_SMOOTH,
                                        time_of_min, seed=rng.integers(0, 2**31))
                offset_w = fit_boundary(t, y, duration, "flat_after", PRIMARY_SMOOTH,
                                         time_of_min, seed=rng.integers(0, 2**31))
                rows.append(dict(shape=shape, duration=duration, rep=rep,
                                  onset_ambiguity_s=onset_w, offset_ambiguity_s=offset_w))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    os.makedirs(os.path.join(OUT_DIR, "diagnostics"), exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "diagnostics", "synthetic_invariance.csv"), index=False)
    print(df.shape)

    print("\n--- Diagnostic 1: Spearman(duration, CI width) under TRUE constant ambiguity ---")
    for shape, grp in df.groupby("shape"):
        for measure in ["onset_ambiguity_s", "offset_ambiguity_s"]:
            clean = grp[["duration", measure]].replace([np.inf, -np.inf], np.nan).dropna()
            rho, p = spearmanr(clean["duration"], clean[measure])
            print(f"  {shape:10s} {measure:20s} rho={rho:+.3f}  p={p:.4g}  n={len(clean)}  "
                  f"median_width={clean[measure].median():.2f}s")

    print("\nBy duration quartile (median width), for comparison with the real-data table:")
    df["duration_q"] = pd.qcut(df["duration"], 4)
    print(df.groupby(["shape", "duration_q"], observed=True)[["onset_ambiguity_s", "offset_ambiguity_s"]]
          .median().to_string())
