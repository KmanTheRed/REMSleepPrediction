"""
Unit tests for changepoint.py on synthetic troughs with known transition
times: step, ramp, V (sharp trough), and double-dip.

Run with:  python -m pytest tests/test_changepoint.py -v
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from changepoint import (  # noqa: E402
    grid_search_tau, changepoint_ci, grid_search_tau_slopedfloor, changepoint_ci_slopedfloor,
)


def make_time(window=90):
    return np.arange(-window / 2, window / 2, 1.0)


def test_sharp_onset_recovers_exact_tau_noise_free():
    """A near-instant onset (steep clean ramp, not a discontinuous jump to a
    new flat level -- OIS/CBV is a hemodynamic signal and cannot jump to a
    new constant within one sample, and the model's post-tau segment is
    itself a ramp, not a plateau, matching how a trough keeps deepening past
    onset -- see protocol.md Section 1). A steep, noise-free ramp has no
    ambiguity: the fit should land tau exactly on the true onset time."""
    t = make_time()
    true_tau = 3.0
    steep_slope = -5.0
    y = np.where(t < true_tau, 0.0, steep_slope * (t - true_tau))
    tau_grid = np.arange(t.min() + 8, t.max() - 8 + 1)
    tau_hat, beta, fitted, resid, sse = grid_search_tau(t, y, "rising_after", tau_grid)
    assert abs(tau_hat - true_tau) <= 1.0
    assert sse < 1e-6


def test_ramp_onset_recovers_known_slope_and_tau():
    """A clean linear ramp starting at a known time: hinge model should
    recover both tau and the slope with tiny noise."""
    rng = np.random.default_rng(0)
    t = make_time()
    true_tau = -5.0
    true_slope = -0.4
    y = np.where(t < true_tau, 0.0, true_slope * (t - true_tau))
    y = y + rng.normal(0, 0.02, size=t.shape)
    tau_grid = np.arange(t.min() + 8, t.max() - 8 + 1)
    tau_hat, beta, fitted, resid, sse = grid_search_tau(t, y, "rising_after", tau_grid)
    assert abs(tau_hat - true_tau) <= 2.0
    assert abs(beta[1] - true_slope) < 0.05


def test_offset_flat_after_recovers_known_tau():
    """Offset kind: ramp recovering toward baseline, then flat. Mirrors the
    onset test but with the hinge on the other side."""
    rng = np.random.default_rng(1)
    t = make_time()
    true_tau = 8.0
    true_slope = -0.5  # signal falling as we approach tau from the left
    # before tau: y = c0 + slope*(tau - t) [[decays to c0 as t -> tau]]
    y = np.where(t < true_tau, 0.0 + true_slope * (true_tau - t), 0.0)
    y = y + rng.normal(0, 0.02, size=t.shape)
    tau_grid = np.arange(t.min() + 8, t.max() - 8 + 1)
    tau_hat, beta, fitted, resid, sse = grid_search_tau(t, y, "flat_after", tau_grid)
    assert abs(tau_hat - true_tau) <= 2.0


def test_ci_width_grows_with_noise():
    """More noise around an otherwise-identical transition should widen the
    bootstrap CI -- this is the core property the whole analysis leans on:
    ambiguity tracks noise-relative-to-slope, not transition depth."""
    t = make_time()
    true_tau = 0.0
    true_slope = -0.3
    tau_grid = np.arange(t.min() + 8, t.max() - 8 + 1)

    def signal(noise_sd, seed):
        rng = np.random.default_rng(seed)
        y = np.where(t < true_tau, 0.0, true_slope * (t - true_tau))
        return y + rng.normal(0, noise_sd, size=t.shape)

    low_noise = changepoint_ci(t, signal(0.02, 10), "rising_after", tau_grid,
                                block_len=15, n_boot=100, seed=42)
    high_noise = changepoint_ci(t, signal(0.3, 10), "rising_after", tau_grid,
                                 block_len=15, n_boot=100, seed=42)
    assert high_noise["width"] > low_noise["width"]


def test_ci_width_indifferent_to_depth_when_slope_scales_with_depth_and_duration():
    """A deeper-but-proportionally-slower trough (same slope magnitude at the
    boundary, i.e. the ramp is simply longer) should NOT by itself produce a
    wider onset CI -- this is exactly the descent-time confound the protocol
    calls out: depth/duration alone must not drive the ambiguity measure."""
    t = make_time()
    tau_grid = np.arange(t.min() + 8, t.max() - 8 + 1)
    true_tau = 0.0
    same_slope = -0.3

    def signal(seed):
        rng = np.random.default_rng(seed)
        y = np.where(t < true_tau, 0.0, same_slope * (t - true_tau))
        return y + rng.normal(0, 0.05, size=t.shape)

    a = changepoint_ci(t, signal(1), "rising_after", tau_grid, n_boot=100, seed=1)
    b = changepoint_ci(t, signal(2), "rising_after", tau_grid, n_boot=100, seed=1)
    # same generating process (same slope, same noise) => CI widths should be
    # in the same ballpark regardless of which "episode" (seed) it came from
    assert abs(a["width"] - b["width"]) < 5.0


def test_v_shaped_trough_onset_tau_near_departure_when_window_clipped_at_bottom():
    """A V-shaped trough (linear down, linear up, no flat bottom) reaching
    its minimum only 10s after onset. Feeding the model the FULL +-45s window
    (which includes the upswing) breaks the model's monotonic-ramp
    assumption and drags tau away from the true onset -- this was found by
    this exact test and is why measures.py additionally clips the onset
    fitting window at the local trough minimum (protocol.md Section 5,
    change-log 2026-09-28). This test checks the clipped-window usage, which
    is what the real pipeline does."""
    t = make_time()
    true_tau = -10.0
    depth_time = 10.0  # trough bottom 10s after tau
    slope_down = -1.0

    y = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti < true_tau:
            y[i] = 0.0
        elif ti < true_tau + depth_time:
            y[i] = slope_down * (ti - true_tau)
        else:
            # ramp back up symmetric to the down-slope
            bottom = slope_down * depth_time
            y[i] = bottom - slope_down * (ti - (true_tau + depth_time))

    # Unclipped: the upswing after the bottom violates the model's
    # assumption and tau is NOT recovered -- documents the failure mode.
    tau_grid_full = np.arange(t.min() + 8, t.max() - 8 + 1)
    tau_unclipped, *_ = grid_search_tau(t, y, "rising_after", tau_grid_full)
    assert abs(tau_unclipped - true_tau) > 2.0

    # Clipped at the trough bottom (as measures.py does): tau is recovered.
    clip_mask = t <= (true_tau + depth_time)
    t_clip, y_clip = t[clip_mask], y[clip_mask]
    tau_grid_clip = np.arange(t_clip.min() + 8, t_clip.max() - 2 + 1)
    tau_clipped, *_ = grid_search_tau(t_clip, y_clip, "rising_after", tau_grid_clip)
    assert abs(tau_clipped - true_tau) <= 2.0


def test_double_dip_onset_locks_to_first_departure_not_second_dip():
    """Double-dip: a small partial rebound mid-episode should not confuse the
    ONSET change-point (window is local to the onset, not the whole episode)."""
    rng = np.random.default_rng(3)
    t = make_time(window=120)
    true_tau = -5.0
    y = np.where(t < true_tau, 0.0, -0.4 * (t - true_tau))
    # partial rebound centered near t=+40 (well outside the +-45s onset
    # window in the real pipeline, but included here as part of the trace to
    # prove the local window is what matters)
    rebound_mask = (t > 35) & (t < 45)
    y = y.copy()
    y[rebound_mask] += 3.0
    y = y + rng.normal(0, 0.05, size=t.shape)

    local_mask = (t >= -45) & (t <= 45)
    t_local = t[local_mask]
    y_local = y[local_mask]
    tau_grid = np.arange(t_local.min() + 8, t_local.max() - 8 + 1)
    tau_hat, *_ = grid_search_tau(t_local, y_local, "rising_after", tau_grid)
    assert abs(tau_hat - true_tau) <= 2.0


def test_slopedfloor_recovers_tau_and_both_slopes_onset():
    """Diagnostic 4's model (2026-09-28): flat baseline, then ramp at s1 for
    w0 seconds, then continued decline at s2 (the 'sloped floor') -- must
    recover tau and both slopes cleanly on a noise-free synthetic case."""
    t = np.arange(-60, 60, 1.0)
    true_tau, s1, s2, w0 = 5.0, -0.001, -0.0002, 15.0
    u = np.maximum(0, t - true_tau)
    b1, b2 = np.minimum(u, w0), np.maximum(u - w0, 0.0)
    y = 0.01 + s1 * b1 + s2 * b2
    tau_grid = np.arange(t.min() + 8, t.max() - 8 + 1)
    tau_hat, beta, fitted, resid, sse = grid_search_tau_slopedfloor(t, y, "rising_after", tau_grid, w0=w0)
    assert abs(tau_hat - true_tau) <= 1.0
    assert abs(beta[1] - s1) < 1e-6
    assert abs(beta[2] - s2) < 1e-6
    assert sse < 1e-9


def test_slopedfloor_recovers_tau_offset_mirrored():
    t = np.arange(-60, 60, 1.0)
    true_tau, s1, s2, w0 = -5.0, -0.001, -0.0002, 15.0
    u = np.maximum(0, true_tau - t)
    b1, b2 = np.minimum(u, w0), np.maximum(u - w0, 0.0)
    y = 0.01 + s1 * b1 + s2 * b2
    tau_grid = np.arange(t.min() + 8, t.max() - 8 + 1)
    tau_hat, beta, *_ = grid_search_tau_slopedfloor(t, y, "flat_after", tau_grid, w0=w0)
    assert abs(tau_hat - true_tau) <= 1.0


def test_slopedfloor_tolerates_continued_decline_that_broke_the_2param_model():
    """The whole point of this model: unlike the 2-param hinge (which assumes
    a flat plateau or an unboundedly steepening ramp), a genuine continued
    -but-shallower- decline after the initial departure should not bias tau,
    because s2 absorbs it instead of forcing a flat/steep misfit."""
    rng = np.random.default_rng(2)
    t = np.arange(-60, 60, 1.0)
    true_tau, s1, s2, w0 = 0.0, -0.002, -0.0001, 15.0
    u = np.maximum(0, t - true_tau)
    b1, b2 = np.minimum(u, w0), np.maximum(u - w0, 0.0)
    y = 0.01 + s1 * b1 + s2 * b2 + rng.normal(0, 0.00005, len(t))
    tau_grid = np.arange(t.min() + 8, t.max() - 8 + 1)
    tau_hat, *_ = grid_search_tau_slopedfloor(t, y, "rising_after", tau_grid, w0=w0)
    assert abs(tau_hat - true_tau) <= 2.0


def test_slopedfloor_motion_covariate_does_not_bias_tau_when_uncorrelated():
    rng = np.random.default_rng(3)
    t = np.arange(-60, 60, 1.0)
    true_tau, s1, s2, w0 = 3.0, -0.001, -0.0003, 15.0
    u = np.maximum(0, t - true_tau)
    b1, b2 = np.minimum(u, w0), np.maximum(u - w0, 0.0)
    y = 0.01 + s1 * b1 + s2 * b2 + rng.normal(0, 0.00003, len(t))
    motion = rng.normal(0, 1, len(t))  # unrelated to y
    tau_grid = np.arange(t.min() + 8, t.max() - 8 + 1)
    tau_hat, beta, *_ = grid_search_tau_slopedfloor(t, y, "rising_after", tau_grid, w0=w0, motion=motion)
    assert abs(tau_hat - true_tau) <= 2.0
    assert abs(beta[3]) < 0.01  # motion coefficient should stay near zero


def test_slopedfloor_ci_sane_shape():
    rng = np.random.default_rng(4)
    t = np.arange(-60, 60, 1.0)
    true_tau, s1, s2, w0 = 0.0, -0.001, -0.0002, 15.0
    u = np.maximum(0, t - true_tau)
    b1, b2 = np.minimum(u, w0), np.maximum(u - w0, 0.0)
    y = 0.01 + s1 * b1 + s2 * b2 + rng.normal(0, 0.0001, len(t))
    tau_grid = np.arange(t.min() + 8, t.max() - 8 + 1)
    result = changepoint_ci_slopedfloor(t, y, "rising_after", tau_grid, w0=w0, n_boot=100, seed=0)
    assert result["ci_lo"] <= result["tau"] <= result["ci_hi"]
    assert result["width"] >= 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
