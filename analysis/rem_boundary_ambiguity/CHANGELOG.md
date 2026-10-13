# Protocol change log

Every change to `protocol.md` after its initial commit is logged here with the date and the
reason, per the task's instruction not to change the protocol silently.

## 2026-09-28 — clip change-point fitting window at the local trough minimum

**What changed:** Section 5, Group A. The onset change-point fitting window
(`[onset-45, onset+45]` s) is additionally clipped so its post-`tau` side never extends past
`time_of_min` (the trough minimum). Symmetrically, the offset window's pre-`tau` side never
starts before `time_of_min`.

**Why:** `tests/test_changepoint.py::test_v_shaped_trough_onset_tau_near_departure_when_window_clipped_at_bottom`
constructed a V-shaped trough (linear descent, bottom at 10 s, linear ascent back) and found
that feeding the unclipped ±45 s window to the change-point model pulls `tau` more than 2 s
from the true onset, because the model assumes the post-onset segment keeps ramping in one
direction — once the trough bottoms out and reverses, that assumption is violated and the
fit is biased. This mainly affects short episodes, where the bottom (and possibly the
opposite boundary) falls inside what would otherwise be a fixed ±45 s window. Clipping at
`time_of_min` fixed it in the same test (tau recovered to within 2 s).

This was found while implementing Step 2 (unit tests), before any real data was scored, so no
previously-computed ambiguity numbers are affected.

## 2026-09-28 — A11 timestamp gap; widened gap-exclusion rule and row-lookup method

**What changed:** Section 2 (data), Section 3 (#4). Documented a genuine ~1044 s timestamp
discontinuity in `animal11_20230616B_trimmed.pq` at row 341980, and switched all elapsed-time
-> row-index lookups to go through a real `elapsed_sec` array (built from the file's actual
timestamps) instead of assuming `row_index == elapsed_seconds`. Widened the gap-exclusion
rule (#4) to explicitly cover timestamp discontinuities, not only flat/interpolated values.

**Why:** `io_utils.load_recording` originally asserted every recording was contiguous 1 Hz
and raised on A11. Investigating showed the gap is real (a segment removed when the file was
"trimmed"), not a bug in reading the file. Assuming row index == elapsed seconds would
silently misalign every manual-score bout after ~95 h into the A11 recording (the point of
the gap) by up to 1044 s, which would corrupt onset/offset localization for roughly the back
third of A11's bouts without raising any error. Found before any A11 episode was scored.

## 2026-09-28 — rebound_frac bug: was measuring net recovery, not a reversed rise

**Not a protocol change** (Section 5's definition, "mid-episode rebound (fraction of depth)",
is unchanged) — logged because the first implementation did not match it. The original
`rebound_frac` tracked the running minimum after the global trough and took the largest
subsequent rise from it, which is just "how far the signal has recovered by any later point"
— for an episode that simply descends and then ascends back to baseline (the normal case),
that measures the *entire final ascent*, not a rebound. On the real data this produced a
median of 0.79 and values up to 15.4 (i.e. "recovering" 15x the trough depth, from noise past
baseline) across 1,779 episodes — the tell that this was measuring the wrong thing.

**Fix:** `measures.morphology` now only counts a rise as a rebound if it is subsequently
*reversed* by a redescent back to at least 50% of the trough depth — i.e. a genuine
double-dip (dip -> partial recovery -> dip again), not the ordinary single final ascent out of
the episode. `tests/test_measures.py::test_morphology_ordinary_final_recovery_is_not_a_rebound`
checks the negative case explicitly. All `per_episode.csv`/sensitivity outputs were
regenerated after this fix; none of the Group A/B ambiguity numbers were affected (rebound is
a Group C covariate, computed independently).

## 2026-09-28 — overlay plots: sampled flagged episodes instead of "every" one

**What changed:** Section 9. "Every flagged episode" is now "up to 25 randomly sampled
episodes per flag criterion" (`plotting.py:make_overlays`, `max_per_flag=25`).

**Why:** measured on the real per-episode output (after the rebound_frac fix above):
`high_ambiguity` (>12 s) fires on 1,399 of 1,779 episodes (79%) and `far_departure` (>25 s) on
951 (53%) -- median onset departure precedes manual onset by ~27 s across the whole dataset
(see the main findings note). `weak_trough` (<5 SD, 161/1,779 = 9%) and `high_rebound` (>60%,
33/1,779 = 1.9%, post-fix) are true minorities and would have been plottable in full. But
`high_ambiguity` and `far_departure` alone describe the typical episode in this dataset, not
rare anomalies -- plotting every flagged episode would still be well over half the dataset,
which stops being useful for spot-checking and is really a substantive finding in its own
right ("ambiguity is large by default here"), not a QC problem. The required 30-random-episode
set is unaffected; `weak_trough` and `high_rebound` are plotted in full since both are under
the 25-per-flag cap anyway.

## 2026-09-28 — MAJOR: primary duration-effect finding retracted pending robustness diagnostics

**What happened:** the finding reported after the primary run ("ambiguity increases with
duration", Spearman rho≈0.40, p«1e-50) does not survive a battery of robustness diagnostics
requested and run the same day. Full detail in `diagnostics/DIAGNOSTICS_REPORT.md`; summary:

1. A synthetic invariance test (fixed slope/noise/AR1 matched to real medians, duration swept
   20-250s, true ambiguity constant by construction) shows the original method
   (`changepoint_ci`, ±45s window clipped at the trough minimum, block_len=15 bootstrap)
   produces a same-sign, comparable-magnitude spurious duration correlation (rho +0.32 to
   +0.43) on the physiologically realistic "flat floor" and "ramp" shapes even with zero true
   effect.
2. 26% of onset CIs and 23% of offset CIs are censored at the search-window edge on real data,
   and this censoring rate rises sharply with duration (11%->38% onset, 8%->50% offset from
   shortest to longest quartile) -- a large share of "long episodes have wide CIs" is really
   "long episodes hit the window ceiling," confirmed visually (9/10 of the widest
   long-quartile CIs are exactly 74.0s, the maximum the grid allows).
3. A refit (fixed ±60s window, a new "sloped floor" model tolerating continued decline instead
   of requiring window clipping, block_len=45 given a measured residual AR1 of 0.88-0.92, a
   motion covariate at offset) removes the synthetic artifact for the realistic shape (rho≈0,
   n.s.) and, applied to real data, shows the duration relationship reverses to weak-negative
   (rho=-0.20 onset, -0.12 offset) with a narrow ~4-9s range across all quartiles, not the
   original 9-29s spread.
4. Refitting the mixed model with random slopes for duration by animal (only 4 animals)
   inflates the duration coefficient's p-value from ~1e-55/1e-86 to 0.30/0.20 -- the original
   significance is not robust to per-animal slope heterogeneity.

**Why this matters for the protocol:** Section 5's primary change-point model (unbounded
post-tau ramp, clipped at the trough minimum, ±45s window, block_len=15) is not removed from
the pipeline -- `run_pipeline.py`/`per_episode.csv` are left as documented, since redesigning
the primary method mid-analysis without further validation would trade one unvalidated choice
for another. But the primary conclusion drawn from its output **is** retracted. Anyone using
`per_episode.csv`'s `onset_ambiguity_s`/`offset_ambiguity_s` for a duration claim must first
read `diagnostics/DIAGNOSTICS_REPORT.md`. The honest current state: whether boundary ambiguity
scales with duration is **not demonstrated** by this data/method combination.
`changepoint.grid_search_tau_slopedfloor`/`changepoint_ci_slopedfloor` (tested in
`tests/test_changepoint.py`) is the more defensible model found during this diagnostic pass;
it is not yet wired into the main pipeline as the new primary measure -- doing so on the full
dataset rather than the 600-episode diagnostic sample would be the natural next step.
