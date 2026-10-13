import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episodes import build_episodes  # noqa: E402


def make_rec(n, gaps=None):
    ois = np.sin(np.linspace(0, 50, n)) * 0.01 + 1.0
    return dict(
        ois=ois, cbv=-ois, motion=np.full(n, 0.2),
        caffeine=np.zeros(n, dtype=np.int8), caffeine_dose=np.zeros(n),
        n=n, elapsed_sec=np.arange(n, dtype=float), gaps=gaps or [],
    )


def test_keeps_well_separated_bout_with_enough_baseline():
    rec = make_rec(1000)
    bouts = [(200.0, 250.0)]
    eps, exc = build_episodes("T", bouts, rec, recording_duration_s=1000.0)
    assert len(eps) == 1
    assert exc == []


def test_excludes_bout_with_insufficient_pre_baseline():
    rec = make_rec(1000)
    bouts = [(10.0, 60.0)]  # only 10s before onset, need 25s
    eps, exc = build_episodes("T", bouts, rec, recording_duration_s=1000.0)
    assert len(eps) == 0
    assert any("pre-baseline" in e.reason for e in exc)


def test_excludes_bout_with_insufficient_post_baseline_at_recording_end():
    rec = make_rec(1000)
    bouts = [(500.0, 990.0)]  # only 10s left in the recording, need 20s
    eps, exc = build_episodes("T", bouts, rec, recording_duration_s=1000.0)
    assert len(eps) == 0
    assert any("post-baseline" in e.reason for e in exc)


def test_excludes_both_neighbors_symmetrically_when_too_close():
    rec = make_rec(1000)
    bouts = [(100.0, 150.0), (160.0, 200.0)]  # gap = 10s < 30s
    eps, exc = build_episodes("T", bouts, rec, recording_duration_s=1000.0)
    assert len(eps) == 0
    reasons_bout0 = [e.reason for e in exc if e.bout_idx == 0]
    reasons_bout1 = [e.reason for e in exc if e.bout_idx == 1]
    assert any("neighbor gap" in r for r in reasons_bout0)
    assert any("neighbor gap" in r for r in reasons_bout1)


def test_excludes_bout_with_timestamp_gap_in_baseline_window():
    rec = make_rec(1000, gaps=[dict(row_before=150, row_after=151,
                                     sec_before=150.0, sec_after=1200.0, gap_sec=1050.0)])
    bouts = [(200.0, 250.0)]  # pre-baseline window [140,195] overlaps the gap [150,1200]
    eps, exc = build_episodes("T", bouts, rec, recording_duration_s=2000.0)
    assert len(eps) == 0
    assert any("data gap" in e.reason for e in exc)


def test_excludes_bout_with_flat_run_in_baseline_window():
    rec = make_rec(1000)
    rec["ois"][140:150] = 1.0  # 10s flat run inside pre-baseline window [195-60,195-5]
    bouts = [(200.0, 250.0)]
    eps, exc = build_episodes("T", bouts, rec, recording_duration_s=1000.0)
    assert len(eps) == 0
    assert any("data gap" in e.reason for e in exc)


def test_caffeine_flag_read_at_onset():
    rec = make_rec(1000)
    rec["caffeine"][200:] = 1
    bouts = [(300.0, 350.0)]
    eps, exc = build_episodes("T", bouts, rec, recording_duration_s=1000.0)
    assert len(eps) == 1
    assert eps[0].caffeine == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
