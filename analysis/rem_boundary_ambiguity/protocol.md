# Protocol: does REM boundary ambiguity (seconds) scale with episode duration?

Status: **frozen at first commit below.** Any change after that point must be logged in
[CHANGELOG.md](CHANGELOG.md) with a reason, per the task instructions. Do not silently edit
this file once analysis has started.

## 0. Question

Is the ambiguity of a REM onset/offset boundary — the width of the time window over which
a reasonable scorer could place that boundary — constant across episode durations, or does
it grow with duration? Onset and offset are analyzed **separately** throughout; they are
scored differently (offset is usually pinned by a motion spike, onset is not) and there is
no reason to expect the same answer for both.

## 1. What "ambiguity" means here, and why morphology measures alone don't answer it

Ambiguity is a property of **scoring**, not of the signal: how far apart two reasonable
placements of the same boundary could be. Descent time, ascent time, and trough depth are
signal-morphology measures. They are useful and are computed here, but none of them
directly measures ambiguity:

- **Descent/ascent time is contaminated by the in-REM ramp.** Most troughs keep deepening
  well after the animal is behaviorally in REM, so 20-80% descent time grows with episode
  duration by construction. That growth is not boundary ambiguity; it is the episode getting
  deeper. Any duration-vs-ambiguity claim built on descent time is confounded from the start.
- **Noise matters as much as slope.** A steep-but-noisy transition can be *harder* to place
  than a slow-but-clean one. A transition-duration number with no noise reference conflates
  these.
- **Scorers use more than IOS/CBV.** Offsets are usually pinned by a motion spike; a slow
  IOS recovery is not necessarily ambiguous if motion marks the exit cleanly. A signal-only
  measure that ignores motion can overstate offset ambiguity.
- **Some disagreement is conventional, not signal-driven.** One scorer marking "first
  departure" and another marking "steepest point" disagree even on a razor-sharp transition.
  No signal-shape measure captures this; only repeated scoring can.

Given that, this protocol uses three layers with distinct jobs, in increasing order of
validity as a measure of ambiguity itself:

| Layer | What it measures | Role here |
|---|---|---|
| Trough morphology (depth, 20-80% descent/ascent, time-of-min fraction, rebound) | Size and speed of the CBV/IOS excursion | **Explanatory covariates** — used to explain *why* ambiguity varies, never reported as ambiguity itself |
| Noise-band crossing (time from 1 SD to 3 SD departure / 3 SD to 1 SD return) | A normalized, threshold-based proxy for transition time | **Secondary measure** — cheap, matches what the original brief specified, kept for continuity and as a sensitivity check |
| Change-point interval width | A confidence interval, in seconds, on where the boundary actually is, from a model fit to the local signal | **Primary measure** — this is what "ambiguity in seconds" means in this analysis |

If a re-scoring set becomes available (Section 8), it validates layer 2: does interval width
predict actual scorer-to-scorer boundary disagreement? Until then, the change-point interval
is reported as the best available proxy, not as ground truth on scorer disagreement.

## 2. Data

- Animals: **A1, A3, A8, A11 only.** A5 and A6 are present in `ground_truth/` but must not
  be loaded (A5 is explicitly excluded by the task; A6 was never requested).
- Recordings (1 Hz, confirmed empirically — every file's row count equals its stated
  duration in hours × 3600, except A11 which is a "trimmed" file at 311.71 h and matches to
  the row):

  | Animal | Recording file | Manual score file | Duration (h) | Caffeine dose(s) |
  |---|---|---|---|---|
  | A1 | `recordings/COMBINED_CAFFEINE_A1_2022-04-08_to_2022-04-15.pq` | `ground_truth/manualscore1.txt` | 192.0 | 0, 0.3 |
  | A3 | `recordings/COMBINED_CAFFEINE_A3_2022-07-20_to_2022-07-27.pq` | `ground_truth/manualscore 3.txt` | 168.0 | 0, 0.3 |
  | A8 | `recordings/COMBINED_CAFFEINE_A8_2023-06-08_to_2023-06-22.pq` | `ground_truth/wowCombinedCaffeine8.txt` | 336.0 | 0, 0.6 |
  | A11 | `recordings/animal11_20230616B_trimmed.pq` | `ground_truth/A11ManualScore.txt` | 311.71 | 0, 0.6 |

  File-to-animal matching for the manual score `.txt` files is by filename plus a duration
  check: each file's last timestamp is within the corresponding recording's total duration
  (e.g. A1: last bout ends at 167:56:22 h, recording is 192.0 h long; A3: 166:36:36 vs
  168.0 h; A8: 332:00:36 vs 336.0 h; A11: 308:28:04 vs 311.71 h). `manualscore5.txt`
  (contains a "Caffeine Begins 253:34:14" marker, consistent with A5's known caffeine
  protocol) and `manualscore6.txt` are excluded from this analysis by construction — they
  are never opened.

- Columns used: `CBV`, `OIS` (confirmed `CBV = -OIS` to 1e-6, i.e. exactly the same signal,
  sign-flipped — verified via `corr(CBV, OIS) = -0.9999999999995989` on A1), `Motion Ang.
  Vector` (the motion indicator — see below), `caffeine` (0.0 = no caffeine, 0.3 or 0.6 =
  caffeine, per-animal dose), and the `t_idx` timestamp / row index for alignment.
- **Signal used for boundary work: OIS.** REM shows as a dip in OIS (a "trough"), matching
  the brief's language (depth, descent, ascent, trough). CBV is the same information
  sign-flipped and is not analyzed separately — reporting both would double every result for
  no new information.
- **Motion indicator: `Motion Ang. Vector`**, chosen empirically over `Motion Lin. Vector`:
  on A1, `Motion Ang. Vector` mean is 0.164 (NREM) / 0.178 (REM) vs 1.368 (awake) — a clean
  ~8x separation. `Motion Lin. Vector` only separates awake from sleep by ~1% (0.999 vs
  1.007) and is not usable as an offset marker.
- **Caffeine indicator:** the `caffeine` column is used directly (binarized: `> 0` = caffeine
  epoch). Episodes are stratified by the caffeine value in effect at the manual onset
  timestamp.
- Manual score files give elapsed time since recording start, `HH:MM:SS, HH:MM:SS`, where
  `HH` can exceed 24 (e.g. A11's last bout starts at `308:25:37`) — this is elapsed hours,
  not wall-clock, and is parsed as a `timedelta` added to the recording's first timestamp /
  first row.
- No NaNs and no flat/stuck runs ≥5 s were found in `CBV`/`OIS` for any of the 4 recordings
  used (checked empirically before writing this protocol).
- **`animal11_20230616B_trimmed.pq` has a genuine ~1044 s timestamp gap** (row 341980 →
  341981, real elapsed time jumps by 1044 s) — this is a real discontinuity from how the file
  was "trimmed", not a flat/stuck-value artifact, and row index is therefore **not**
  interchangeable with elapsed seconds for this file. All row lookups go through an
  `elapsed_sec` array built from the actual timestamps (`io_utils.sec_to_row`), never
  `row = int(elapsed_sec)`. Manual-score elapsed times are assumed to be real-time elapsed
  since recording start, so this mapping is also what correctly "skips" the missing time
  range the way the original scorer's clock would have. The gap-exclusion rule (Section 3,
  #4) is widened to catch this case explicitly, not just flat/interpolated values.

## 3. Episode construction and exclusions

Episodes = manual REM bouts read from the four `.txt` files above (start, end pairs).
Exclude a bout, with the reason logged, if:

1. Fewer than 25 s of clean pre-baseline available before onset (bounded by recording start
   or the end of the *previous* kept bout, whichever is closer).
2. Fewer than 20 s of clean post-baseline available after offset (bounded by recording end or
   the start of the *next* bout).
3. Another bout's boundary (start or end) falls within 30 s of this bout's start or end
   (neighbor-too-close exclusion — applied symmetrically, so if A excludes B, B also
   excludes A per this rule; both are logged).
4. A data gap falls inside either baseline window ([start-60, start-5] s or [end+10,
   end+60] s) or inside the local change-point fitting window (Section 5, Group A). A gap is
   either (a) ≥5 consecutive identical raw `OIS` samples (flat/interpolated segment), or
   (b) a real timestamp discontinuity in the recording (see A11 below) — both are logged with
   which kind fired.

Every excluded bout is written to `output/exclusions.csv` with animal, bout index, raw
onset/offset, and the specific reason (one row per reason if multiple apply).

## 4. Baselines and smoothing

- Smoothing: 5 s centered moving median on raw OIS, used for all downstream measures unless
  a sensitivity variant says otherwise (Section 7).
- Pre-baseline window: `[onset-60, onset-5]` s of raw OIS (baseline stats are computed on
  **raw**, not smoothed, signal, per the brief).
- Post-baseline window: `[offset+10, offset+60]` s of raw OIS.
- Baseline level: median of the window.
- Baseline noise SD: `1.4826 * MAD` of the **linearly detrended** window (`np.polyfit` degree
  1, residuals from that fit, MAD of residuals).
- Each episode gets its own pre-baseline (for onset measures) and post-baseline (for offset
  measures) — they are never pooled across episodes or animals.

## 5. Per-episode measures

Computed for every kept episode, split into three groups per Section 1.

**Group A — primary ambiguity (change-point interval width, seconds):**
- Fit, in a local window around the manual onset (`[onset-45, onset+45]` s of smoothed OIS,
  clipped to the episode and to available data), a 2-segment constant+ramp model: flat at
  the pre-baseline level up to change-point `tau`, then linear with slope `s` after `tau`.
  Fit by grid search over `tau` (1 s steps) with OLS for `(level, s)` at each `tau`, keep the
  minimum-SSE `tau`.
- **Window is additionally clipped at the local trough minimum**: the onset window's
  post-`tau` side stops at `time_of_min` (never extends past it), and the offset window's
  pre-`tau` side starts no earlier than `time_of_min`. The model assumes the post-onset
  (pre-offset) segment ramps monotonically; once the trough bottoms out and starts
  recovering, that assumption breaks and `tau` can be dragged away from the true boundary.
  (Found by the V-shaped-trough unit test in `tests/test_changepoint.py` — see
  [CHANGELOG.md](CHANGELOG.md) 2026-09-28.) This mainly matters for short episodes where the
  bottom falls inside the ±45 s window.
- Get a 95% interval on `tau` via **moving-block bootstrap** on the OLS residuals (block
  length = 15 s = 3x the smoothing window, to respect short-range autocorrelation left by
  the 5 s median filter), refit per resample, `n_boot=200` by default. Interval =
  [2.5th, 97.5th] percentile of the bootstrap `tau` estimates.
- `onset_ambiguity_s` = width of that interval (97.5th − 2.5th percentile). This is the
  primary onset measure.
- Same construction at the manual offset, ramp direction reversed (ramp back toward
  baseline), local window `[offset-45, offset+45]` s → `offset_ambiguity_s`.
- **Motion co-registration at offset (not folded into the interval, reported alongside it):**
  `motion_spike_offset` = 1 if `Motion Ang. Vector` exceeds 0.3 (roughly the empirical
  midpoint between sleep ~0.17 and awake ~1.37, see Section 2) at any point inside the
  offset's 95% CP interval, else 0; `motion_spike_time_s` = time of that spike relative to
  the manual offset (NaN if none). This is reported, not subtracted from
  `offset_ambiguity_s` — collapsing the two would hide exactly the "scorers use more than
  the trace" effect this protocol is trying to characterize. Whether `motion_spike_offset`
  predicts a *smaller* real disagreement is a question for the re-scoring validation
  (Section 8), not an assumption to bake in now.

**Group B — secondary/proxy measure (noise-band crossing, seconds), per the original brief:**
- `onset_ambiguity_nb_s` = time from leaving 1 SD of pre-baseline to reaching 3 SD below
  pre-baseline (first crossing each, in the departure direction).
- `offset_ambiguity_nb_s` = time from the last sample at ≥3 SD below post-baseline back to
  the first sample within 1 SD of post-baseline.
- These use the **same sign convention as depth** (SD units below baseline for OIS's dip).
- Kept because it's what the brief asked for, cheap, and gives a sensitivity check against
  Group A: if the two disagree about the duration relationship, that's worth knowing, not
  averaging away.

**Group C — explanatory/morphology covariates (not treated as ambiguity):**
- `depth_pct`, `depth_sd`: trough minimum vs. pre-baseline, in % and in pre-baseline SD units.
- `descent_20_80_s`, `ascent_20_80_s`: time between 20% and 80% of depth, on the way down
  and on the way back up.
- `min_time_frac`: time of the OIS minimum, as a fraction of episode duration (0=onset,
  1=offset).
- `rebound_frac`: size of the largest mid-episode partial recovery, as a fraction of depth
  (double-dip indicator).
- `motion_at_offset`: raw `Motion Ang. Vector` value at the manual offset timestamp.
- `episode_duration_s`: manual offset − manual onset. The predictor of interest.
- `caffeine`: binarized caffeine flag in effect at onset.
- `animal`: for the random-intercept term.

## 6. Stratification

All analyses (Section 7) are run twice: caffeine epochs and no-caffeine epochs, using the
`caffeine` column value at each episode's manual onset, binarized at `> 0`.

## 7. Statistics

Run separately for onset and offset, and separately for Group A (`*_ambiguity_s`) and Group
B (`*_ambiguity_nb_s`) measures, within each caffeine stratum:

- **Spearman(duration, measure).**
- **Mixed model** `measure ~ f(duration)` with a random intercept per animal
  (`statsmodels.MixedLM`, `groups=animal`), comparing `f` = constant (intercept-only),
  `log(duration)`, and linear `duration`, ranked by AIC.
- Group C covariates are reported descriptively and used only to *explain* Group A/B
  results (e.g. "duration correlates with ambiguity because deeper troughs are noisier",
  checked by adding `depth_sd` as a second fixed effect and seeing if the duration
  coefficient survives) — never presented as an ambiguity estimate on their own.

## 8. Sensitivity analyses

- Noise-band thresholds: 1→2 SD and 0.5→2 SD, in addition to the primary 1→3 SD.
- Smoothing: 3 s and 10 s moving median, in addition to the primary 5 s (re-run baseline SD,
  Group A and Group B measures under each).
- Alternative duration predictor: **core duration** = time from the 3 SD onset crossing to
  the 3 SD offset crossing (vs. label duration = manual offset − manual onset), re-run the
  Section 7 stats with this predictor in place of `episode_duration_s`.
- **Re-scoring validation hook (not run yet — no re-scoring data exists on this machine).**
  `validation.py` defines the expected schema for a re-scoring CSV
  (`episode_id, rescorer, onset_s, offset_s`) and a `compare_to_ambiguity()` function that,
  given such a file, computes per-boundary rescorer-vs-original disagreement and checks
  whether `*_ambiguity_s` (Group A) predicts it (Spearman + a simple regression). This
  validates whether the change-point interval is actually measuring what it claims to. Until
  a re-scoring set exists (recommended: 40-60 episodes from these 4 animals, stratified by
  duration and caffeine, rescored blind), Group A is reported as the best available proxy,
  explicitly not validated against real scorer disagreement.

## 9. Outputs

- `output/per_episode.csv` — one row per kept episode, all Group A/B/C measures + metadata.
- `output/exclusions.csv` — one row per excluded bout per reason.
- `output/summary_by_duration_bin_caffeine.csv` — summary table.
- `output/stats_results.csv` — Spearman + mixed-model AIC comparison, one row per
  (measure × stratum × sensitivity variant).
- `output/overlays/` — plots: 30 randomly sampled episodes, plus every flagged episode
  (weak trough <5 SD deep, onset or offset ambiguity >12 s, rebound >60%, or label-onset
  >25 s from first IOS departure from baseline).

## Change log

None yet. See [CHANGELOG.md](CHANGELOG.md).
