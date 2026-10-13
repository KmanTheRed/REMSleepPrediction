"""
Full pipeline: load 4 animals -> build episodes -> compute per-episode
measures (Groups A/B/C) under the primary settings and the Section 8
sensitivity variants -> write output/*.csv -> run stats -> make overlay plots.

See protocol.md for the full methodology; this module is the executable
version of it.
"""
import os
import time

import numpy as np
import pandas as pd

import io_utils as io
from episodes import build_episodes
from measures import (
    baseline_stats, moving_median, noise_band_onset, noise_band_offset,
    morphology, changepoint_measure, crossing_time_onset, crossing_time_offset,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
CP_WINDOW = 45          # seconds each side of a boundary for the change-point fit
PRE_BASELINE = (-60, -5)
POST_BASELINE = (10, 60)
MOTION_SPIKE_THRESH = 0.3  # Motion Ang. Vector; midpoint of sleep (~0.17) / awake (~1.37)

PRIMARY_SMOOTH = 5
SENSITIVITY_SMOOTHS = (3, 10)
PRIMARY_NB = (1.0, 3.0)
SENSITIVITY_NB = ((1.0, 2.0), (0.5, 2.0))
N_BOOT_PRIMARY = 200
N_BOOT_SENSITIVITY = 100  # lighter, since this multiplies the primary run 2x


def _extract_window(rec, sec_lo, sec_hi):
    i0 = io.sec_to_row(rec, sec_lo)
    i1 = io.sec_to_row(rec, sec_hi)
    raw = rec["ois"][i0:i1 + 1]
    t = rec["elapsed_sec"][i0:i1 + 1]
    return t, raw


def process_episode(rec, ep, smoothing, nb_lo_hi, n_boot, seed):
    """Compute all Groups A/B/C measures for one episode under one
    (smoothing, noise-band-threshold, n_boot) setting. Returns a flat dict."""
    onset, offset = ep.onset, ep.offset
    duration = offset - onset

    # --- baselines (raw signal, protocol Section 4) ---
    _, pre_raw = _extract_window(rec, onset + PRE_BASELINE[0], onset + PRE_BASELINE[1])
    _, post_raw = _extract_window(rec, offset + POST_BASELINE[0], offset + POST_BASELINE[1])
    pre_level, pre_sd = baseline_stats(pre_raw)
    post_level, post_sd = baseline_stats(post_raw)

    # --- full episode window, smoothed, for morphology + time-of-min ---
    t_full, y_full_raw = _extract_window(rec, onset - CP_WINDOW, offset + CP_WINDOW)
    y_full = moving_median(y_full_raw, smoothing)

    in_episode = (t_full >= onset) & (t_full <= offset)
    if not np.any(in_episode):
        return None
    t_ep, y_ep = t_full[in_episode], y_full[in_episode]
    time_of_min = float(t_ep[int(np.argmin(y_ep))])

    morph = morphology(t_ep, y_ep, pre_level, pre_sd, onset, offset)

    # --- onset window: [onset-45, onset+45], clipped at time_of_min ---
    onset_hi = min(onset + CP_WINDOW, time_of_min)
    on_mask = (t_full >= onset - CP_WINDOW) & (t_full <= onset_hi)
    t_on, y_on = t_full[on_mask], y_full[on_mask]
    t_on_rel = t_on - onset

    # --- offset window: [offset-45, offset+45], clipped at time_of_min ---
    offset_lo = max(offset - CP_WINDOW, time_of_min)
    off_mask = (t_full >= offset_lo) & (t_full <= offset + CP_WINDOW)
    t_off, y_off = t_full[off_mask], y_full[off_mask]
    t_off_rel = t_off - offset

    # Group A: change-point CI (primary ambiguity)
    cp_on = changepoint_measure(t_on_rel, y_on, "rising_after",
                                 block_len=3 * smoothing, n_boot=n_boot, seed=seed)
    cp_off = changepoint_measure(t_off_rel, y_off, "flat_after",
                                  block_len=3 * smoothing, n_boot=n_boot, seed=seed + 1)

    # Group B: noise-band crossing (secondary)
    lo_k, hi_k = nb_lo_hi
    nb_on = noise_band_onset(t_on_rel, y_on, pre_level, pre_sd, lo_k, hi_k)
    nb_off = noise_band_offset(t_off_rel, y_off, post_level, post_sd, lo_k, hi_k)

    # core duration (Section 8 alt. predictor): always at the primary 1-3 SD def
    onset_3sd_t = crossing_time_onset(t_on_rel, y_on, pre_level, pre_sd, k=3.0)
    offset_3sd_t = crossing_time_offset(t_off_rel, y_off, post_level, post_sd, k=3.0)
    core_duration = (
        (offset + offset_3sd_t) - (onset + onset_3sd_t)
        if np.isfinite(onset_3sd_t) and np.isfinite(offset_3sd_t) else np.nan
    )

    # onset departure offset (Section 9 flag: "label onset >25s from IOS
    # departure"): time from manual onset to the signal's 1-SD departure.
    # Reuses the already-extracted onset window -- no extra data pull.
    onset_departure_offset_s = crossing_time_onset(t_on_rel, y_on, pre_level, pre_sd, k=1.0)

    # motion co-registration at offset
    motion_at_offset = float(rec["motion"][io.sec_to_row(rec, offset)])
    lo_row = io.sec_to_row(rec, offset + cp_off["ci_lo"]) if np.isfinite(cp_off.get("ci_lo", np.nan)) else None
    hi_row = io.sec_to_row(rec, offset + cp_off["ci_hi"]) if np.isfinite(cp_off.get("ci_hi", np.nan)) else None
    motion_spike_offset, motion_spike_time_s = 0, np.nan
    if lo_row is not None and hi_row is not None and hi_row >= lo_row:
        seg = rec["motion"][lo_row:hi_row + 1]
        if np.any(seg > MOTION_SPIKE_THRESH):
            motion_spike_offset = 1
            spike_row = lo_row + int(np.argmax(seg > MOTION_SPIKE_THRESH))
            motion_spike_time_s = rec["elapsed_sec"][spike_row] - offset

    return dict(
        animal=ep.animal, bout_idx=ep.bout_idx, onset=onset, offset=offset,
        episode_duration_s=duration, core_duration_s=core_duration,
        onset_departure_offset_s=onset_departure_offset_s,
        caffeine=ep.caffeine,
        pre_baseline_level=pre_level, pre_baseline_sd=pre_sd,
        post_baseline_level=post_level, post_baseline_sd=post_sd,
        onset_ambiguity_s=cp_on["width"], onset_tau=cp_on.get("tau", np.nan),
        offset_ambiguity_s=cp_off["width"], offset_tau=cp_off.get("tau", np.nan),
        onset_ambiguity_nb_s=nb_on, offset_ambiguity_nb_s=nb_off,
        depth_pct=morph["depth_pct"], depth_sd=morph["depth_sd"],
        descent_20_80_s=morph["descent_20_80_s"], ascent_20_80_s=morph["ascent_20_80_s"],
        min_time_frac=morph["min_time_frac"], rebound_frac=morph["rebound_frac"],
        motion_at_offset=motion_at_offset,
        motion_spike_offset=motion_spike_offset, motion_spike_time_s=motion_spike_time_s,
    )


def process_episode_nb_only(rec, ep, smoothing, nb_lo_hi):
    """Cheap Group-B-only recompute for the noise-band-threshold sensitivity
    variants (Section 8) -- skips the change-point/bootstrap work entirely
    since Group A doesn't depend on the noise-band thresholds."""
    onset, offset = ep.onset, ep.offset
    _, pre_raw = _extract_window(rec, onset + PRE_BASELINE[0], onset + PRE_BASELINE[1])
    _, post_raw = _extract_window(rec, offset + POST_BASELINE[0], offset + POST_BASELINE[1])
    pre_level, pre_sd = baseline_stats(pre_raw)
    post_level, post_sd = baseline_stats(post_raw)

    t_full, y_full_raw = _extract_window(rec, onset - CP_WINDOW, offset + CP_WINDOW)
    y_full = moving_median(y_full_raw, smoothing)

    in_episode = (t_full >= onset) & (t_full <= offset)
    if not np.any(in_episode):
        return None
    t_ep, y_ep = t_full[in_episode], y_full[in_episode]
    time_of_min = float(t_ep[int(np.argmin(y_ep))])

    on_mask = (t_full >= onset - CP_WINDOW) & (t_full <= min(onset + CP_WINDOW, time_of_min))
    t_on, y_on = t_full[on_mask] - onset, y_full[on_mask]
    off_mask = (t_full >= max(offset - CP_WINDOW, time_of_min)) & (t_full <= offset + CP_WINDOW)
    t_off, y_off = t_full[off_mask] - offset, y_full[off_mask]

    lo_k, hi_k = nb_lo_hi
    nb_on = noise_band_onset(t_on, y_on, pre_level, pre_sd, lo_k, hi_k)
    nb_off = noise_band_offset(t_off, y_off, post_level, post_sd, lo_k, hi_k)
    return dict(animal=ep.animal, bout_idx=ep.bout_idx, episode_duration_s=offset - onset,
                caffeine=ep.caffeine, onset_ambiguity_nb_s=nb_on, offset_ambiguity_nb_s=nb_off)


def run_sensitivity(animals=("A1", "A3", "A8", "A11")):
    """Section 8 sensitivity analyses: smoothing 3s/10s (full Group A+B,
    lighter n_boot) and noise-band thresholds 1->2 / 0.5->2 SD (Group B
    only, primary smoothing). Returns (smoothing_df, noiseband_df)."""
    smoothing_rows, nb_rows = [], []
    for animal in animals:
        rec, bouts, skipped = io.load_animal(animal)
        duration_s = rec["elapsed_sec"][-1]
        eps, _ = build_episodes(animal, bouts, rec, duration_s)

        for smoothing in SENSITIVITY_SMOOTHS:
            t0 = time.time()
            for ep in eps:
                row = process_episode(rec, ep, smoothing, PRIMARY_NB, N_BOOT_SENSITIVITY,
                                       seed=hash((animal, ep.bout_idx, smoothing)) % (2 ** 31))
                if row is not None:
                    row["smoothing_s"] = smoothing
                    smoothing_rows.append(row)
            print(f"{animal} smoothing={smoothing}s: {len(eps)} episodes in {time.time()-t0:.1f}s")

        for lo_hi in SENSITIVITY_NB:
            for ep in eps:
                row = process_episode_nb_only(rec, ep, PRIMARY_SMOOTH, lo_hi)
                if row is not None:
                    row["nb_lo"], row["nb_hi"] = lo_hi
                    nb_rows.append(row)

    return pd.DataFrame(smoothing_rows), pd.DataFrame(nb_rows)


def run_all(animals=("A1", "A3", "A8", "A11"), n_boot=N_BOOT_PRIMARY, limit_per_animal=None,
            verbose=True):
    all_rows = []
    all_exclusions = []
    for animal in animals:
        rec, bouts, skipped = io.load_animal(animal)
        assert not skipped, f"{animal}: unparsed manual-score lines {skipped}"
        duration_s = rec["elapsed_sec"][-1]
        eps, exc = build_episodes(animal, bouts, rec, duration_s)
        all_exclusions.extend(exc)
        if limit_per_animal:
            eps = eps[:limit_per_animal]
        t0 = time.time()
        for k, ep in enumerate(eps):
            row = process_episode(rec, ep, PRIMARY_SMOOTH, PRIMARY_NB, n_boot,
                                   seed=hash((animal, ep.bout_idx)) % (2 ** 31))
            if row is not None:
                all_rows.append(row)
        if verbose:
            print(f"{animal}: {len(eps)} episodes in {time.time() - t0:.1f}s")
    per_episode = pd.DataFrame(all_rows)
    exclusions = pd.DataFrame([e.__dict__ for e in all_exclusions])
    return per_episode, exclusions


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    do_sensitivity = "--sensitivity" in args
    args = [a for a in args if a != "--sensitivity"]
    n_per_animal = int(args[0]) if args else None

    os.makedirs(OUT_DIR, exist_ok=True)
    per_episode, exclusions = run_all(limit_per_animal=n_per_animal)
    per_episode.to_csv(os.path.join(OUT_DIR, "per_episode.csv"), index=False)
    exclusions.to_csv(os.path.join(OUT_DIR, "exclusions.csv"), index=False)
    print("primary:", per_episode.shape, exclusions.shape)

    if do_sensitivity:
        smoothing_df, nb_df = run_sensitivity()
        smoothing_df.to_csv(os.path.join(OUT_DIR, "sensitivity_smoothing.csv"), index=False)
        nb_df.to_csv(os.path.join(OUT_DIR, "sensitivity_noiseband.csv"), index=False)
        print("sensitivity:", smoothing_df.shape, nb_df.shape)
