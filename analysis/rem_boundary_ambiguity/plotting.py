"""
Overlay plots (protocol.md Section 9): 30 random episodes, plus flagged
episodes (weak trough <5 SD, onset/offset ambiguity >12 s, rebound >60%,
label onset >25 s from IOS departure).

NOTE on volume (see CHANGELOG.md 2026-09-28 "flag thresholds fire on the
majority of episodes"): on real data, three of the four flag criteria each
fire on 50-80% of episodes -- they describe the typical episode here, not
rare outliers. Plotting "every" flagged episode would mean ~1750 of 1779
plots, which stops being a QC aid. This module plots the 30 required random
episodes, plus up to `max_per_flag` per flag criterion (default 25),
sampled from the flagged set, documented as a deviation from "every flagged
episode" for exactly this reason.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import io_utils as io
from run_pipeline import _extract_window, moving_median, CP_WINDOW, PRIMARY_SMOOTH

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OVERLAY_DIR = os.path.join(OUT_DIR, "overlays")


def plot_episode(rec, row, out_path):
    onset, offset = row["onset"], row["offset"]
    pad = max(CP_WINDOW, 30)
    t, y_raw = _extract_window(rec, onset - pad, offset + pad)
    y = moving_median(y_raw, PRIMARY_SMOOTH)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t - onset, y_raw, color="#bbbbbb", lw=0.6, label="raw OIS")
    ax.plot(t - onset, y, color="#1f5fa8", lw=1.4, label=f"{PRIMARY_SMOOTH}s median")
    ax.axvline(0, color="green", ls="--", lw=1, label="manual onset")
    ax.axvline(offset - onset, color="red", ls="--", lw=1, label="manual offset")

    on_lo, on_hi = row.get("onset_tau", np.nan), np.nan
    if np.isfinite(row.get("onset_ambiguity_s", np.nan)):
        tau = row["onset_tau"]
        half = row["onset_ambiguity_s"] / 2
        ax.axvspan(tau - half, tau + half, color="green", alpha=0.15)
    if np.isfinite(row.get("offset_ambiguity_s", np.nan)):
        tau = (offset - onset) + row["offset_tau"]
        half = row["offset_ambiguity_s"] / 2
        ax.axvspan(tau - half, tau + half, color="red", alpha=0.15)

    ax.set_xlabel("time relative to manual onset (s)")
    ax.set_ylabel("OIS")
    ax.set_title(
        f"{row['animal']} bout {row['bout_idx']}  dur={row['episode_duration_s']:.0f}s  "
        f"onset_amb={row.get('onset_ambiguity_s', np.nan):.1f}s  "
        f"offset_amb={row.get('offset_ambiguity_s', np.nan):.1f}s  "
        f"caffeine={row['caffeine']}"
    )
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def make_overlays(per_episode, max_random=30, max_per_flag=25, seed=0):
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)

    recs = {a: io.load_animal(a)[0] for a in per_episode["animal"].unique()}

    flags = {
        "weak_trough": per_episode["depth_sd"] < 5,
        "high_ambiguity": (per_episode["onset_ambiguity_s"] > 12) | (per_episode["offset_ambiguity_s"] > 12),
        "high_rebound": per_episode["rebound_frac"] > 0.6,
        "far_departure": per_episode.get("onset_departure_offset_s", pd.Series(dtype=float)).abs() > 25,
    }

    plotted = set()
    manifest = []

    random_idx = rng.choice(per_episode.index, size=min(max_random, len(per_episode)), replace=False)
    for idx in random_idx:
        row = per_episode.loc[idx]
        fname = f"random_{row['animal']}_{row['bout_idx']}.png"
        plot_episode(recs[row["animal"]], row, os.path.join(OVERLAY_DIR, fname))
        plotted.add((row["animal"], row["bout_idx"]))
        manifest.append(dict(file=fname, animal=row["animal"], bout_idx=row["bout_idx"], reason="random"))

    for flag_name, mask in flags.items():
        flagged = per_episode[mask.fillna(False)]
        if flagged.empty:
            continue
        sample = flagged.sample(n=min(max_per_flag, len(flagged)), random_state=seed)
        for idx, row in sample.iterrows():
            key = (row["animal"], row["bout_idx"])
            fname = f"{flag_name}_{row['animal']}_{row['bout_idx']}.png"
            plot_episode(recs[row["animal"]], row, os.path.join(OVERLAY_DIR, fname))
            manifest.append(dict(file=fname, animal=row["animal"], bout_idx=row["bout_idx"], reason=flag_name))

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(os.path.join(OVERLAY_DIR, "manifest.csv"), index=False)
    flag_counts = {k: int(v.fillna(False).sum()) for k, v in flags.items()}
    print("flag counts (of", len(per_episode), "episodes):", flag_counts)
    print("plots written:", len(manifest_df))
    return manifest_df, flag_counts


if __name__ == "__main__":
    per_episode = pd.read_csv(os.path.join(OUT_DIR, "per_episode.csv"))
    make_overlays(per_episode)
