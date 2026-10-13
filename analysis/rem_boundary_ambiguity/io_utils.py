"""
Loading recordings and manual-score files for the 4 animals used in this
analysis. See protocol.md Section 2 for the animal/file table and the checks
performed here (1 Hz, CBV = -OIS, no gaps in the files actually used).
"""
import os
import re

import numpy as np
import pandas as pd

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ANIMALS = {
    "A1": dict(
        recording="recordings/COMBINED_CAFFEINE_A1_2022-04-08_to_2022-04-15.pq",
        manual_score="ground_truth/manualscore1.txt",
    ),
    "A3": dict(
        recording="recordings/COMBINED_CAFFEINE_A3_2022-07-20_to_2022-07-27.pq",
        manual_score="ground_truth/manualscore 3.txt",
    ),
    "A8": dict(
        recording="recordings/COMBINED_CAFFEINE_A8_2023-06-08_to_2023-06-22.pq",
        manual_score="ground_truth/wowCombinedCaffeine8.txt",
    ),
    "A11": dict(
        recording="recordings/animal11_20230616B_trimmed.pq",
        manual_score="ground_truth/A11ManualScore.txt",
    ),
}

EXCLUDED_ANIMALS = ("A5", "A6")  # never loaded -- A5 per task instruction, A6 not requested

MOTION_COL = "Motion Ang. Vector"
NEEDED_COLS = ["OIS", "CBV", MOTION_COL, "caffeine"]

_TIME_RE = re.compile(r"^\s*(\d+):(\d{2}):(\d{2})\s*$")


def _parse_elapsed(tok):
    """Parse 'HH:MM:SS' (HH may exceed 24 -- elapsed time, not wall clock)
    into elapsed seconds. Returns None if the token doesn't match (e.g. the
    'Caffeine Begins ...' marker line in manualscore5.txt, which is never
    loaded by this pipeline but the parser should not choke if pointed at
    it)."""
    m = _TIME_RE.match(tok)
    if not m:
        return None
    h, mnt, s = (int(x) for x in m.groups())
    return h * 3600 + mnt * 60 + s


def load_manual_scores(path):
    """Return a list of (start_sec, end_sec) elapsed-time REM bouts, sorted."""
    bouts = []
    skipped = []
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            # format: "<idx>\tHH:MM:SS, HH:MM:SS"
            parts = line.split("\t")
            payload = parts[-1]
            halves = [h.strip() for h in payload.split(",")]
            if len(halves) != 2:
                skipped.append((lineno, line))
                continue
            start = _parse_elapsed(halves[0])
            end = _parse_elapsed(halves[1])
            if start is None or end is None:
                skipped.append((lineno, line))
                continue
            bouts.append((float(start), float(end)))
    bouts.sort()
    return bouts, skipped


def load_recording(path):
    """Load one recording and return a dict of numpy arrays plus an
    `elapsed_sec` array giving each row's true elapsed time (in seconds)
    since the recording's first sample, computed from the real timestamps
    rather than assumed from row index.

    This matters: row index and elapsed seconds are NOT interchangeable for
    every file. `animal11_20230616B_trimmed.pq` has a genuine ~1044 s
    timestamp discontinuity (a segment was cut when it was "trimmed") at row
    341980 -- real acquisition time jumps by 1044 s there with no rows in
    between. Manual-score elapsed times are real-time elapsed since
    recording start, so mapping them to rows must go through `elapsed_sec`
    (via `sec_to_row`), never `row = int(elapsed_sec)`. Any such gap is
    recorded in `gaps` so episodes/windows spanning it can be excluded
    (protocol.md Section 3, exclusion 4)."""
    df = pd.read_parquet(path, columns=NEEDED_COLS)

    if isinstance(df.index, pd.DatetimeIndex):
        t = df.index
    else:
        df2 = pd.read_parquet(path, columns=NEEDED_COLS + ["t_idx"])
        t = pd.DatetimeIndex(df2["t_idx"])
        df = df2.drop(columns=["t_idx"])

    t0 = t[0]
    elapsed_sec = (t.values - t0.to_datetime64()).astype("timedelta64[s]").astype(np.float64)

    diffs = np.diff(elapsed_sec)
    gap_idx = np.where(diffs != 1.0)[0]  # index i: gap is between row i and i+1
    gaps = [
        dict(row_before=int(i), row_after=int(i) + 1,
             sec_before=float(elapsed_sec[i]), sec_after=float(elapsed_sec[i + 1]),
             gap_sec=float(diffs[i]))
        for i in gap_idx
    ]

    caffeine_dose = df["caffeine"].to_numpy(dtype=np.float32)
    return dict(
        ois=df["OIS"].to_numpy(dtype=np.float64),
        cbv=df["CBV"].to_numpy(dtype=np.float64),
        motion=df[MOTION_COL].to_numpy(dtype=np.float64),
        caffeine_dose=caffeine_dose,
        caffeine=(caffeine_dose > 0).astype(np.int8),
        n=len(df),
        t0=t0,
        elapsed_sec=elapsed_sec,
        gaps=gaps,
    )


def sec_to_row(rec, sec):
    """Map an elapsed-seconds timestamp to the nearest row index, via the
    recording's real elapsed_sec array (robust to timestamp gaps)."""
    idx = int(np.searchsorted(rec["elapsed_sec"], sec))
    if idx <= 0:
        return 0
    if idx >= rec["n"]:
        return rec["n"] - 1
    # pick whichever neighbor is closer in real elapsed time
    before, after = rec["elapsed_sec"][idx - 1], rec["elapsed_sec"][idx]
    return idx - 1 if abs(before - sec) <= abs(after - sec) else idx


def gap_in_range(rec, sec_lo, sec_hi):
    """True if any recorded timestamp gap falls within [sec_lo, sec_hi]."""
    for g in rec["gaps"]:
        if g["sec_before"] < sec_hi and g["sec_after"] > sec_lo:
            return True
    return False


def load_animal(animal):
    """Convenience: load recording + manual scores for one of the 4 animals
    used in this analysis. Raises if `animal` is not one of them (in
    particular, raises for A5/A6 rather than silently loading them)."""
    if animal not in ANIMALS:
        raise ValueError(
            f"animal {animal!r} is not one of {sorted(ANIMALS)} "
            f"used by this analysis (A5/A6 are intentionally excluded)"
        )
    spec = ANIMALS[animal]
    rec = load_recording(os.path.join(DATA_ROOT, spec["recording"]))
    bouts, skipped = load_manual_scores(os.path.join(DATA_ROOT, spec["manual_score"]))
    return rec, bouts, skipped
