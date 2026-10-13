# Robustness diagnostics on the change-point CI duration effect (2026-09-28)

**Verdict: the primary result ("ambiguity increases with duration", rho≈0.40) does not
survive scrutiny. It is substantially a methodological artifact of the original change-point
model's fixed-size window colliding with variable episode duration. A more robust refit shows
a much weaker, direction-uncertain effect that does not reliably clear a proper random-slopes
test with only 4 animals.** Details below, in the order requested.

## 1. Synthetic invariance test

Simulated troughs with slope/noise SD/AR(1) fixed at real measured medians (onset slope
-2.2e-5, offset slope -4.9e-5, noise SD 1e-4, AR1 0.88 — from `diag_fit_and_truncation.py`),
duration swept 20-250s, two shapes (`flat_floor`: fixed-duration ramp, flat AR(1) floor,
fixed-duration ramp back — the physiologically realistic case; `ramp`: symmetric V, no floor).
True ambiguity is constant by construction. Ran the exact original pipeline
(`changepoint_ci`, ±45s window clipped at the local trough minimum, block_len=15 bootstrap):

| shape | measure | rho | p |
|---|---|---|---|
| flat_floor | onset | **+0.32** | 5e-9 |
| flat_floor | offset | **+0.33** | 2e-9 |
| ramp | onset | -0.32 | 3e-9 |
| ramp | offset | **+0.43** | 7e-16 |

**All four are significant despite zero true duration dependence.** The flat_floor case — the
one that best represents a genuine REM episode (departure, stable "in REM" period, return) —
produces a spurious *positive* correlation of comparable magnitude to a meaningful fraction of
the real rho≈0.40 finding, in the same direction. This is the core problem.

## 2. Fraction of CIs truncated at search-window bounds

From `diag_fit_and_truncation.py` on all 1,779 real episodes (original ±45s, clip-at-min
window; `EDGE_TOL=1s`):

| | tau at lo bound | tau at hi bound | CI lo at bound | CI hi at bound |
|---|---|---|---|---|
| onset | 5.5% | 0.4% | **26.1%** | 2.0% |
| offset | 4.0% | 3.8% | 12.0% | **23.2%** |

A quarter of onset CIs and offset CIs are **censored** at the window edge — the bootstrap
wanted to place the bound further out than the window allowed. Critically, **this censoring
rate itself scales sharply with duration**:

| duration quartile | onset CI-lo at bound | offset CI-hi at bound |
|---|---|---|
| shortest (18-44s) | 11.0% | 8.4% |
| | 22.5% | 8.7% |
| | 33.4% | 26.7% |
| longest (104-246s) | **37.8%** | **49.5%** |

Half of the longest-quartile offset CIs are pinned at the window ceiling. This directly
explains a large share of the "long episodes have wide CIs" pattern: many of the "widest" CIs
are not well-estimated wide intervals, they are saturated at the maximum value the window's
tau-grid permits (74s = the full ±37s grid span). Confirmed visually in diagnostic 5 below —
9 of the 10 widest long-quartile CIs are exactly 74.0s.

## 3. Fitted slope, residual SD, residual AR1, baseline slope by duration quartile

From the same real-episode fits (`output/diagnostics/slope_noise_ar1_by_quartile.csv`):

| duration quartile | onset slope | onset resid SD | onset resid AR1 | offset slope | offset resid SD | offset resid AR1 |
|---|---|---|---|---|---|---|
| 18-44s | -0.000026 | 0.000075 | 0.850 | -0.000053 | 0.000096 | 0.831 |
| 44-68s | -0.000022 | 0.000092 | 0.893 | -0.000053 | 0.000112 | 0.853 |
| 68-104s | -0.000019 | 0.000103 | 0.906 | -0.000037 | 0.000131 | 0.887 |
| 104-246s | -0.000020 | 0.000128 | 0.911 | -0.000032 | 0.000168 | 0.917 |

Two real (non-artifactual) trends worth noting: residual SD rises ~65-75% from shortest to
longest quartile, and residual AR1 rises from ~0.83-0.85 to ~0.91-0.92. Both would genuinely
widen ambiguity via `w ≈ Δk·σ/|slope|`, so *some* real duration-correlated signal-quality
change plausibly exists — but it's swamped by the artifact in (1)-(2), and the AR1 is far
higher than the original block_len=15 bootstrap (calibrated as "3x smoothing") was designed
for.

**Pre-onset baseline slope is ~0 at every quartile** (-2e-6 to -3e-6), ruling out systematic
pre-onset drift as an explanation for anything above.

## 4. Refit: fixed ±60s window, sloped-floor model, longer block bootstrap, motion covariate

New model (`changepoint.grid_search_tau_slopedfloor`): flat baseline, then ramp at slope s1
for a fixed 15s, then **continued decline at a second slope s2** ("sloped floor") instead of
either a hard plateau or an unboundedly steepening ramp — this tolerates continued movement
after departure without needing window clipping. Window fixed at ±60s regardless of duration
or trough location. Block bootstrap length raised to 45s (the original 15s was far too short
given the measured AR1≈0.88-0.92). Motion (z-scored `Motion Ang. Vector`) added as an extra
covariate at offset only.

**On synthetic data (does the artifact survive?):**

| shape | measure | rho | p |
|---|---|---|---|
| flat_floor | onset | +0.006 | 0.91 |
| flat_floor | offset | -0.015 | 0.79 |
| ramp | onset | +0.001 | 0.98 |
| ramp | offset | **-0.760** | 2e-61 |

**The artifact is gone for the realistic flat_floor shape** (both rho≈0, both n.s.) — this
refit is validated as trustworthy there. It is **not** gone for the floor-less "ramp" shape at
offset specifically (a genuine model-misspecification edge case: the sloped-floor model
assumes some stable floor exists to distinguish "recent ramp" from "older floor" — a shape
with no floor at all still confounds duration with how much of the fixed window the ascent
fills). Real short episodes (median ~30-56s) are close to or shorter than the 30s (2×15s)
floor the model expects, so some residual risk carries over to real short-episode offsets.

**On real data (600-episode sample, ~150/duration-quartile):**

| measure | rho | p |
|---|---|---|
| onset_ambiguity_s | **-0.196** | 1.3e-6 |
| offset_ambiguity_s | **-0.116** | 0.0043 |

| duration quartile | onset ambiguity (median) | offset ambiguity (median) |
|---|---|---|
| 20-45s | 9.1s | 6.3s |
| 45-69s | 7.0s | 4.0s |
| 69-104s | 8.0s | 5.0s |
| 104-246s | 6.0s | 5.0s |

**The duration effect does not persist — it reverses to weak-but-significant negative,** and
the absolute magnitudes collapse to a narrow, roughly flat 4-9s band across all quartiles,
nothing like the original 9s-to-21-29s spread.

## 5. Overlay plots

`output/diagnostics/overlays_diag5/`: 10 longest-quartile widest-CI episodes
(`longest_widest_*.png`) and 10 shortest-quartile typical-CI episodes (`shortest_typical_*.png`).
Visual confirmation of (2): 9/10 "widest" plots show the red (offset) or green (onset) CI
shading spanning the *entire* visible window — saturated, not a genuine localized-but-wide
estimate. The "typical" shortest-quartile plots show tight, visually well-localized shading
right at the actual transition.

## A5 exclusion — confirmed

`io_utils.ANIMALS` contains only `{A1, A3, A8, A11}`; `load_animal` raises `ValueError` for
any other key (verified live: `load_animal('A5')` raises). The only occurrences of the string
"A5" anywhere in this codebase are the exclusion guard itself and its docstring/comment — A5
is never read from disk by any script in this pipeline.

## Random slopes for duration by animal

Refit `measure ~ log(duration)` with `re_formula="~log(duration)"` (random slope + intercept
per animal) against the fixed-slope (random intercept only) model, on the **original**
per-episode results:

| measure | model | AIC | duration coef | p |
|---|---|---|---|---|
| onset_ambiguity_s | intercept-only | 13721.6 | 7.69 | 4e-55 |
| onset_ambiguity_s | + random slope | 13743.2 (worse) | 7.51 | **0.30** |
| offset_ambiguity_s | intercept-only | 14297.0 | 11.40 | 8e-86 |
| offset_ambiguity_s | + random slope | 14323.8 (worse) | 10.94 | **0.20** |

AIC prefers the simpler fixed-slope model (as expected — 4 groups barely identify a slope
variance component), but the fixed-effect duration coefficient's p-value **inflates by ~50
orders of magnitude** once between-animal slope heterogeneity is allowed for. With only 4
animals, a population-level duration effect and animal-specific idiosyncrasy are not
statistically distinguishable from each other. This is a second, independent reason (on top
of the artifact above) not to trust the original p-values at face value.

## Bottom line

The claim reported earlier this session — "REM boundary ambiguity increases with episode
duration" (Spearman rho≈0.40, p«1e-50) — **should be withdrawn as stated.** The evidence now
shows:
1. The measurement method produces a same-sign, comparable-magnitude spurious correlation
   even when true ambiguity is exactly constant (diagnostic 1).
2. A substantial, duration-correlated fraction of the original CIs are window-censored, not
   genuinely estimated (diagnostic 2, visually confirmed in diagnostic 5).
3. A refit designed to remove those two problems, and validated artifact-free on the
   realistic synthetic case, shows a much weaker, *negative*-leaning real-data relationship
   (diagnostic 4) — the opposite sign, an order of magnitude smaller in practical terms (a
   4-9s band, not a 9-29s spread).
4. Even the original positive finding's significance is not robust to modeling per-animal
   duration slopes, which only 4 animals can't reliably separate from a population effect.

The honest current answer to "does boundary ambiguity scale with duration": **not
demonstrated by this data/method combination.** If anything, the more defensible model points
slightly the other way, but that result itself carries a flagged residual risk for
floor-less short episodes and should not be over-interpreted as a confirmed effect either.
