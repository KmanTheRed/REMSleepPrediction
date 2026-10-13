"""
Episode construction: turn a list of raw (start, end) manual-score bouts for
one animal into kept episodes + logged exclusions, per protocol.md Section 3.
"""
from dataclasses import dataclass, field

import numpy as np

from io_utils import gap_in_range

PRE_BASELINE = (-60, -5)     # seconds relative to onset
POST_BASELINE = (10, 60)     # seconds relative to offset
MIN_PRE_BASELINE_S = 25
MIN_POST_BASELINE_S = 20
NEIGHBOR_MIN_GAP_S = 30
FLAT_RUN_MIN_S = 5


@dataclass
class Exclusion:
    animal: str
    bout_idx: int
    onset: float
    offset: float
    reason: str


@dataclass
class Episode:
    animal: str
    bout_idx: int
    onset: float
    offset: float
    caffeine: int
    extra: dict = field(default_factory=dict)


def _flat_run_gap(rec, sec_lo, sec_hi):
    """True if raw OIS has >=FLAT_RUN_MIN_S consecutive identical samples
    inside [sec_lo, sec_hi]."""
    from io_utils import sec_to_row

    i0 = sec_to_row(rec, sec_lo)
    i1 = sec_to_row(rec, sec_hi)
    if i1 <= i0:
        return False
    seg = rec["ois"][i0:i1 + 1]
    if len(seg) < FLAT_RUN_MIN_S:
        return False
    d = np.diff(seg)
    flat = (d == 0)
    if not np.any(flat):
        return False
    # longest run of True in `flat`
    run_id = np.cumsum(~flat)
    run_lengths = np.bincount(run_id[flat]) if np.any(flat) else np.array([0])
    return bool(run_lengths.max() >= FLAT_RUN_MIN_S - 1)


def build_episodes(animal, bouts, rec, recording_duration_s):
    """Apply protocol.md Section 3 exclusions. Returns (episodes, exclusions).

    `bouts` must be sorted by onset (io_utils.load_manual_scores already
    sorts). Neighbor-gap exclusion (#3) is evaluated against the RAW bout
    list (a bout excluded for another reason can still exclude its
    neighbor -- both directions are logged), matching "applied symmetrically"
    in the protocol.
    """
    exclusions = []
    n = len(bouts)

    def log(i, onset, offset, reason):
        exclusions.append(Exclusion(animal, i, onset, offset, reason))

    keep_mask = [True] * n
    reasons = [[] for _ in range(n)]

    for i, (onset, offset) in enumerate(bouts):
        # #1: pre-baseline availability
        prev_end = bouts[i - 1][1] if i > 0 else 0.0
        avail_pre = onset - max(prev_end, 0.0)
        if avail_pre < MIN_PRE_BASELINE_S:
            reasons[i].append(f"pre-baseline only {avail_pre:.1f}s (<{MIN_PRE_BASELINE_S}s)")

        # #2: post-baseline availability
        next_start = bouts[i + 1][0] if i < n - 1 else recording_duration_s
        avail_post = next_start - offset
        if avail_post < MIN_POST_BASELINE_S:
            reasons[i].append(f"post-baseline only {avail_post:.1f}s (<{MIN_POST_BASELINE_S}s)")

        # #3: neighbor too close (symmetric)
        if i > 0 and (onset - bouts[i - 1][1]) < NEIGHBOR_MIN_GAP_S:
            reasons[i].append(f"neighbor gap {onset - bouts[i - 1][1]:.1f}s (<{NEIGHBOR_MIN_GAP_S}s, prev bout)")
        if i < n - 1 and (bouts[i + 1][0] - offset) < NEIGHBOR_MIN_GAP_S:
            reasons[i].append(f"neighbor gap {bouts[i + 1][0] - offset:.1f}s (<{NEIGHBOR_MIN_GAP_S}s, next bout)")

        # #4: data gap in baseline windows (flat/interpolated OR timestamp gap)
        pre_lo, pre_hi = onset + PRE_BASELINE[0], onset + PRE_BASELINE[1]
        post_lo, post_hi = offset + POST_BASELINE[0], offset + POST_BASELINE[1]
        if gap_in_range(rec, pre_lo, pre_hi) or _flat_run_gap(rec, pre_lo, pre_hi):
            reasons[i].append("data gap in pre-baseline window")
        if gap_in_range(rec, post_lo, post_hi) or _flat_run_gap(rec, post_lo, post_hi):
            reasons[i].append("data gap in post-baseline window")

    episodes = []
    for i, (onset, offset) in enumerate(bouts):
        if reasons[i]:
            for r in reasons[i]:
                log(i, onset, offset, r)
            continue
        caffeine_row = None
        from io_utils import sec_to_row
        caffeine_row = int(rec["caffeine"][sec_to_row(rec, onset)])
        episodes.append(Episode(animal=animal, bout_idx=i, onset=onset, offset=offset,
                                 caffeine=caffeine_row))

    return episodes, exclusions
