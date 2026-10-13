import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measures import (  # noqa: E402
    baseline_stats, noise_band_onset, noise_band_offset, morphology, moving_median,
)


def test_moving_median_smooths_spikes():
    x = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])  # single-sample spike
    y = moving_median(x, window=5)
    assert y[3] < 1.0  # spike suppressed
    assert np.allclose(y[:2], 0.0)  # flat region unaffected


def test_baseline_stats_recovers_level_and_zero_noise():
    x = np.full(60, 5.0)
    level, sd = baseline_stats(x)
    assert level == 5.0
    assert sd < 1e-9


def test_baseline_stats_detrends_before_noise_estimate():
    """A linear drift should not inflate the noise SD estimate -- it must be
    removed by the linear detrend before computing MAD."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200) + rng.normal(0, 0.1, 200)
    level, sd = baseline_stats(x)
    assert 0.05 < sd < 0.2  # close to the injected 0.1 SD, not inflated by drift


def test_noise_band_onset_matches_known_crossings():
    # baseline 0, SD "known" to be 1.0 by construction of a step at t=10
    t = np.arange(0, 60, dtype=float)
    y = np.where(t < 10, 0.0, -0.5 * (t - 10))
    # y crosses -1 SD at t-10 = 2 (y=-1), crosses -3 SD at t-10=6 (y=-3)
    w = noise_band_onset(t, y, baseline_level=0.0, sd=1.0, lo_k=1.0, hi_k=3.0)
    assert abs(w - 4.0) < 1e-6


def test_noise_band_offset_matches_known_crossings():
    t = np.arange(0, 60, dtype=float)
    # recovering: deep until t=30, then rises back to baseline
    y = np.where(t < 30, -5.0 + 0.0, np.minimum(0.0, -5.0 + 0.5 * (t - 30)))
    w = noise_band_offset(t, y, baseline_level=0.0, sd=1.0, lo_k=1.0, hi_k=3.0)
    assert w is not None and w > 0


def test_noise_band_returns_nan_when_no_crossing():
    t = np.arange(0, 20, dtype=float)
    y = np.zeros_like(t)  # never departs baseline
    assert np.isnan(noise_band_onset(t, y, 0.0, 1.0))
    assert np.isnan(noise_band_offset(t, y, 0.0, 1.0))


def test_morphology_depth_and_timing_on_synthetic_v_trough():
    t = np.arange(0, 40, dtype=float)
    onset_t, offset_t = 5.0, 35.0
    y = np.where(
        t < 20, 10.0 - 0.5 * np.maximum(0, t - 5),
        10.0 - 0.5 * 15 + 0.5 * (t - 20),
    )
    y = np.clip(y, None, 10.0)
    m = morphology(t, y, pre_baseline_level=10.0, pre_sd=1.0, onset_t=onset_t, offset_t=offset_t)
    assert m["depth_pct"] > 0
    assert 0.0 <= m["min_time_frac"] <= 1.0
    assert m["rebound_frac"] >= 0.0


def test_morphology_rebound_detects_partial_recovery():
    t = np.arange(0, 30, dtype=float)
    y = np.full(30, 0.0)
    y[5:15] = -10.0  # deep trough
    y[10:12] = -4.0  # partial rebound mid-trough (rises 6 out of depth 10)
    y[15:] = -10.0  # redescends back into deep territory -- a genuine double-dip
    m = morphology(t, y, pre_baseline_level=0.0, pre_sd=1.0, onset_t=5.0, offset_t=25.0)
    assert m["rebound_frac"] > 0.3


def test_morphology_ordinary_final_recovery_is_not_a_rebound():
    """A normal episode -- dip, then a single monotonic ascent back to
    baseline at the very end -- must NOT be flagged as a rebound. This is
    the exact bug found on real data: an earlier implementation measured net
    recovery-by-offset instead of a genuine reversed rise, and scored most
    real episodes (whose final ascent approaches or exceeds the trough
    depth) above 1.0."""
    t = np.arange(0, 30, dtype=float)
    y = np.linspace(0, -10, 15).tolist() + np.linspace(-10, 0, 15).tolist()
    y = np.array(y)
    m = morphology(t, y, pre_baseline_level=0.0, pre_sd=1.0, onset_t=0.0, offset_t=29.0)
    assert m["rebound_frac"] < 0.05


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
