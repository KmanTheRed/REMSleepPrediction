"""Build output/summary_by_duration_bin_caffeine.csv (protocol.md Section 9)."""
import os

import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

MEASURES = ["onset_ambiguity_s", "offset_ambiguity_s",
            "onset_ambiguity_nb_s", "offset_ambiguity_nb_s",
            "depth_pct", "depth_sd", "episode_duration_s"]


def build_summary(per_episode, n_bins=4):
    df = per_episode.copy()
    df["duration_bin"] = pd.qcut(df["episode_duration_s"], q=n_bins, duplicates="drop")
    rows = []
    for (dbin, caf), grp in df.groupby(["duration_bin", "caffeine"], observed=True):
        row = dict(duration_bin=str(dbin), caffeine=caf, n=len(grp))
        for m in MEASURES:
            if m in grp.columns:
                vals = grp[m].replace([np.inf, -np.inf], np.nan).dropna()
                row[f"{m}_median"] = vals.median() if len(vals) else np.nan
                row[f"{m}_iqr_lo"] = vals.quantile(0.25) if len(vals) else np.nan
                row[f"{m}_iqr_hi"] = vals.quantile(0.75) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["caffeine", "duration_bin"])


if __name__ == "__main__":
    per_episode = pd.read_csv(os.path.join(OUT_DIR, "per_episode.csv"))
    summary = build_summary(per_episode)
    summary.to_csv(os.path.join(OUT_DIR, "summary_by_duration_bin_caffeine.csv"), index=False)
    print(summary.to_string())
