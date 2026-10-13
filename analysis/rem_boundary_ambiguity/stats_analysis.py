"""
Statistics per protocol.md Section 7: Spearman(duration, measure) and a
mixed model measure ~ f(duration), random intercept per animal, comparing
f in {constant, log(duration), linear duration} by AIC.

Run separately for onset/offset, Group A/B, caffeine strata, and (when the
sensitivity CSVs exist) smoothing variants / noise-band-threshold variants /
core-duration-as-predictor.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

MEASURES_A = ["onset_ambiguity_s", "offset_ambiguity_s"]
MEASURES_B = ["onset_ambiguity_nb_s", "offset_ambiguity_nb_s"]


def _fit_mixedlm(df, measure, duration_col):
    """Fit measure ~ f(duration) with a random intercept per animal for
    f in {constant, log, linear}; return a DataFrame of one row per f with
    AIC, sorted best-first. Drops rows with non-finite measure/duration."""
    sub = df[[measure, duration_col, "animal"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 8 or sub["animal"].nunique() < 2:
        return pd.DataFrame([dict(f="insufficient_data", aic=np.nan, n=len(sub))])

    sub = sub.rename(columns={measure: "y", duration_col: "dur"})
    sub = sub[sub["dur"] > 0]  # log(duration) requires positive duration
    if len(sub) < 8:
        return pd.DataFrame([dict(f="insufficient_data", aic=np.nan, n=len(sub))])
    sub["log_dur"] = np.log(sub["dur"])

    rows = []
    specs = {
        "constant": "y ~ 1",
        "log": "y ~ log_dur",
        "linear": "y ~ dur",
    }
    for name, formula in specs.items():
        try:
            model = smf.mixedlm(formula, sub, groups=sub["animal"])
            result = model.fit(reml=False, disp=False)
            rows.append(dict(f=name, aic=result.aic, n=len(sub),
                              coef=result.params.get("dur", result.params.get("log_dur", np.nan)),
                              pvalue=result.pvalues.get("dur", result.pvalues.get("log_dur", np.nan))))
        except Exception as e:  # convergence failures on tiny/degenerate strata
            rows.append(dict(f=name, aic=np.nan, n=len(sub), error=str(e)))
    out = pd.DataFrame(rows).sort_values("aic", na_position="last").reset_index(drop=True)
    return out


def run_stats(per_episode, duration_col="episode_duration_s", label=""):
    """Returns a tidy DataFrame: one row per (stratum, measure, f) with
    Spearman rho/p (repeated across f rows for convenience) and AIC."""
    results = []
    strata = {"all": per_episode, "no_caffeine": per_episode[per_episode.caffeine == 0],
              "caffeine": per_episode[per_episode.caffeine == 1]}
    for stratum_name, sub in strata.items():
        for measure in MEASURES_A + MEASURES_B:
            if measure not in sub.columns:
                continue
            clean = sub[[measure, duration_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean) >= 3:
                rho, p = spearmanr(clean[duration_col], clean[measure])
            else:
                rho, p = np.nan, np.nan
            mm = _fit_mixedlm(sub, measure, duration_col)
            for _, r in mm.iterrows():
                results.append(dict(
                    variant=label, stratum=stratum_name, measure=measure,
                    duration_col=duration_col, n=len(clean),
                    spearman_rho=rho, spearman_p=p,
                    f=r.get("f"), aic=r.get("aic"), mm_n=r.get("n"),
                    coef=r.get("coef"), coef_pvalue=r.get("pvalue"),
                ))
    return pd.DataFrame(results)


if __name__ == "__main__":
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    per_episode = pd.read_csv(os.path.join(out_dir, "per_episode.csv"))

    all_results = [run_stats(per_episode, "episode_duration_s", label="primary_label_duration")]
    if "core_duration_s" in per_episode.columns:
        all_results.append(run_stats(per_episode, "core_duration_s", label="core_duration_alt"))

    smoothing_path = os.path.join(out_dir, "sensitivity_smoothing.csv")
    if os.path.exists(smoothing_path):
        sm = pd.read_csv(smoothing_path)
        for s, grp in sm.groupby("smoothing_s"):
            all_results.append(run_stats(grp, "episode_duration_s", label=f"smoothing_{s}s"))

    nb_path = os.path.join(out_dir, "sensitivity_noiseband.csv")
    if os.path.exists(nb_path):
        nb = pd.read_csv(nb_path)
        for (lo, hi), grp in nb.groupby(["nb_lo", "nb_hi"]):
            all_results.append(run_stats(grp, "episode_duration_s", label=f"nb_{lo}-{hi}SD"))

    stats_df = pd.concat(all_results, ignore_index=True)
    stats_df.to_csv(os.path.join(out_dir, "stats_results.csv"), index=False)
    print(stats_df.shape)
    print(stats_df[stats_df.variant == "primary_label_duration"].to_string())
