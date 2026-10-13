"""
Re-scoring validation hook (protocol.md Section 8) -- NOT RUN. No re-scoring
data exists on this machine as of 2026-09-28. This module defines the
expected input schema and the comparison this analysis would need once a
re-scoring set is collected, so wiring it in later is a one-line change to
run_pipeline.py, not a redesign.

Recommended re-scoring set (per protocol.md Section 1): 40-60 episodes from
A1/A3/A8/A11, stratified by duration and caffeine, rescored blind -- ideally
by the same scorer weeks later AND by a second scorer.

Expected CSV schema for `rescoring_path`:
    episode_id   -- must match a (animal, bout_idx) pair in per_episode.csv,
                    written as "<animal>_<bout_idx>", e.g. "A1_17"
    rescorer     -- a scorer identifier (e.g. "same_scorer_delayed", "scorer2")
    onset_s      -- rescored onset, elapsed seconds since recording start
    offset_s     -- rescored offset, elapsed seconds since recording start
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_rescoring(path):
    df = pd.read_csv(path)
    required = {"episode_id", "rescorer", "onset_s", "offset_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rescoring file missing columns: {missing}")
    return df


def compare_to_ambiguity(per_episode, rescoring):
    """For each rescored episode, compute |rescored_onset - original_onset|
    and |rescored_offset - original_offset| (actual disagreement, seconds),
    then check whether Group A's onset_ambiguity_s / offset_ambiguity_s
    predicts that disagreement (Spearman + a simple OLS slope).

    This is THE validation check for whether the change-point interval width
    means what this analysis claims it means (protocol.md Section 1). Until
    it is run, Group A is reported as an unvalidated proxy.
    """
    per_episode = per_episode.copy()
    per_episode["episode_id"] = per_episode["animal"] + "_" + per_episode["bout_idx"].astype(str)
    merged = rescoring.merge(per_episode, on="episode_id", how="inner", suffixes=("_rescore", ""))
    if merged.empty:
        raise ValueError("no rescored episode_id matched per_episode.csv")

    merged["onset_disagreement_s"] = (merged["onset_s"] - merged["onset"]).abs()
    merged["offset_disagreement_s"] = (merged["offset_s"] - merged["offset"]).abs()

    out = {}
    for boundary, disag_col, ambig_col in [
        ("onset", "onset_disagreement_s", "onset_ambiguity_s"),
        ("offset", "offset_disagreement_s", "offset_ambiguity_s"),
    ]:
        clean = merged[[disag_col, ambig_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 3:
            out[boundary] = dict(n=len(clean), rho=np.nan, p=np.nan)
            continue
        rho, p = spearmanr(clean[ambig_col], clean[disag_col])
        out[boundary] = dict(n=len(clean), rho=rho, p=p)
    return out, merged


if __name__ == "__main__":
    print(__doc__)
    print("No re-scoring file configured -- this module is a hook, not run by run_pipeline.py.")
