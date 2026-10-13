#!/usr/bin/env python3
"""
rem_eval.py  --  REM episode evaluation: conflate GT JSONs + greedy one-to-one IoU matching.

Two things this does:
  1) conflate : flatten per-frame labels/*.json into one flat ground_truth.txt
                (same 3-col format as the predictions file: image,start_sec,end_sec)
  2) compare  : greedy one-to-one IoU matching of predictions vs GT, with an IoU
                threshold sweep, boundary MAD, and symmetric merge->floor postprocessing.

GT can be supplied either as --labels DIR (raw JSONs) or --gt-txt FILE (already conflated).

Matching is WITHIN-FRAME: offsets are frame-relative seconds, so a predicted episode
in frame X can only match a GT episode in frame X.

Greedy one-to-one (the point of this vs a coverage-sum scorer):
  - compute IoU for every (pred, gt) pair within a frame
  - keep pairs with IoU >= threshold, sort best-first
  - walk down; assign a pair only if BOTH its pred and its gt are still free
  => each prediction matches <=1 GT and each GT matches <=1 prediction. No double-counting.

Author: for Kethan's animal11 REM eval / J Neurosci Methods manuscript.
"""
import argparse, csv, glob, json, os, sys
from collections import defaultdict


# ---------- IO ----------
def _stem(path):
    return os.path.basename(str(path).replace("\\", "/"))

def load_pred_txt(path):
    """Parse a 3-col txt (image,start_sec,end_sec). '#' comments and blanks ignored.
       Works for both the predictions file and a conflated ground_truth.txt."""
    d = defaultdict(list)
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split(",")
            if len(parts) != 3:
                continue
            img, s, e = parts
            d[_stem(img)].append((float(s), float(e)))
    return d

def load_labels_dir(labels_dir):
    """Read every *.json in labels_dir -> {frame_stem: [(s,e),...]}."""
    d = defaultdict(list)
    files = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
    if not files:
        sys.exit(f"[error] no *.json found in {labels_dir}")
    for fp in files:
        with open(fp) as f:
            obj = json.load(f)
        stem = _stem(obj.get("image", os.path.basename(fp).replace(".json", ".png")))
        for ep in obj.get("rem_episodes", []):
            d[stem].append((float(ep["start_offset_sec"]), float(ep["end_offset_sec"])))
        d.setdefault(stem, d[stem])  # ensure neg frames (0 episodes) are registered
    return d

def load_manifest_frames(path):
    """Return {frame_stem: (kind, count)} so negative frames are enumerated too."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[_stem(row["image"])] = (row["kind"].strip(), int(row["rem_episode_count"]))
    return out


# ---------- interval ops ----------
def iou(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0.0, e - s)
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0

def dedupe(intervals):
    return sorted(set((float(s), float(e)) for s, e in intervals))

def merge_then_floor(intervals, merge_gap, floor):
    """Merge intervals whose gap <= merge_gap, THEN drop those shorter than floor.
       Merge-before-floor so fragments can combine before the length test."""
    if not intervals:
        return []
    iv = sorted(intervals)
    merged = [list(iv[0])]
    for s, e in iv[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if (e - s) >= floor]


# ---------- matching ----------
def greedy_match(gt_iv, pred_iv, thr):
    """Greedy one-to-one by IoU within a single frame.
       Returns (matched_pairs, unmatched_pred_idx, unmatched_gt_idx)."""
    pairs = []
    for gi, g in enumerate(gt_iv):
        for pi, p in enumerate(pred_iv):
            v = iou(g, p)
            if v >= thr:
                pairs.append((v, gi, pi))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_g, used_p, matched = set(), set(), []
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi)
        matched.append((gi, pi, v))
    um_p = [i for i in range(len(pred_iv)) if i not in used_p]
    um_g = [i for i in range(len(gt_iv)) if i not in used_g]
    return matched, um_p, um_g


def evaluate(gt, pred, thr):
    TP = FP = FN = 0
    matched_pairs = []  # (gt_interval, pred_interval, iou)
    frames = set(gt) | set(pred)
    for fr in frames:
        g = gt.get(fr, []); p = pred.get(fr, [])
        m, um_p, um_g = greedy_match(g, p, thr)
        TP += len(m); FP += len(um_p); FN += len(um_g)
        for gi, pi, v in m:
            matched_pairs.append((g[gi], p[pi], v))
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec  = TP / (TP + FN) if TP + FN else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=TP, fp=FP, fn=FN, precision=prec, recall=rec, f1=f1,
                matched_pairs=matched_pairs)


def boundary_mad(matched_pairs):
    """start/end MAD in seconds over matched pairs (gt, pred)."""
    if not matched_pairs:
        return None, None, 0
    s = sum(abs(p[0] - g[0]) for g, p, _ in matched_pairs) / len(matched_pairs)
    e = sum(abs(p[1] - g[1]) for g, p, _ in matched_pairs) / len(matched_pairs)
    return s, e, len(matched_pairs)


# ---------- drivers ----------
def do_conflate(labels_dir, out_txt):
    gt = load_labels_dir(labels_dir)
    n = 0
    with open(out_txt, "w") as f:
        f.write("# image,start_offset_sec,end_offset_sec  (conflated ground truth)\n")
        for stem in sorted(gt):
            for s, e in sorted(gt[stem]):
                f.write(f"{stem},{int(s)},{int(e)}\n"); n += 1
    print(f"[conflate] wrote {n} GT episodes across {len(gt)} frames -> {out_txt}")
    return gt


def do_compare(gt, pred, args):
    # optional prep
    if args.dedupe_gt:
        gt = {k: dedupe(v) for k, v in gt.items()}
    if not args.no_postprocess:
        gt   = {k: merge_then_floor(v, args.merge_gap, args.floor) for k, v in gt.items()}
        pred = {k: merge_then_floor(v, args.merge_gap, args.floor) for k, v in pred.items()}
        note = f"merge<= {args.merge_gap}s, floor {args.floor}s (BOTH sides)"
    else:
        note = "none (raw intervals)"

    tot_gt = sum(len(v) for v in gt.values())
    tot_pr = sum(len(v) for v in pred.values())
    print("="*72)
    print("GREEDY ONE-TO-ONE IoU MATCHING")
    print("="*72)
    print(f"  postprocessing : {note}")
    print(f"  dedupe GT      : {args.dedupe_gt}")
    print(f"  GT episodes    : {tot_gt}   pred episodes: {tot_pr}\n")

    # sweep
    thr = args.iou_min
    print(f"  {'IoU>=':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'prec':>7} {'rec':>7} {'F1':>7}")
    thrs = []
    while thr <= args.iou_max + 1e-9:
        thrs.append(round(thr, 4)); thr += args.iou_step
    for t in thrs:
        r = evaluate(gt, pred, t)
        print(f"  {t:>6.2f} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
              f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}")

    # boundary MAD anchored at a fixed (loose) threshold, independent of the sweep
    anchor = evaluate(gt, pred, args.mad_anchor)
    smad, emad, npair = boundary_mad(anchor["matched_pairs"])
    print()
    print(f"  Boundary MAD (matched pairs at IoU>= {args.mad_anchor}):")
    if npair:
        print(f"     start MAD = {smad:.1f}s   end MAD = {emad:.1f}s   (n={npair} pairs)")
    else:
        print("     no matched pairs at anchor threshold")

    # F1 decay slope 0.3 -> 0.7  (your boundary-fidelity metric)
    f_lo = evaluate(gt, pred, 0.3)["f1"]
    f_hi = evaluate(gt, pred, 0.7)["f1"]
    print(f"\n  F1 decay 0.3->0.7 : {f_lo:.3f} -> {f_hi:.3f}  (Δ = {f_lo - f_hi:.3f})")


def main():
    ap = argparse.ArgumentParser(description="REM episode eval: conflate GT + greedy IoU compare")
    ap.add_argument("--mode", choices=["conflate", "compare", "both"], default="both")
    ap.add_argument("--labels", help="dir of per-frame GT *.json")
    ap.add_argument("--gt-txt", help="already-conflated GT txt (image,start,end)")
    ap.add_argument("--pred", help="predictions txt (image,start,end)")
    ap.add_argument("--manifest", help="manifest.csv (to enumerate negative frames too)")
    ap.add_argument("--out-gt", default="ground_truth.txt", help="output for conflate")
    # postprocessing
    ap.add_argument("--floor", type=float, default=30.0, help="duration floor sec (both sides)")
    ap.add_argument("--merge-gap", type=float, default=5.0, help="merge gap sec (both sides)")
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--dedupe-gt", action="store_true", help="drop identical duplicate GT lines")
    # sweep
    ap.add_argument("--iou-min", type=float, default=0.1)
    ap.add_argument("--iou-max", type=float, default=0.7)
    ap.add_argument("--iou-step", type=float, default=0.1)
    ap.add_argument("--mad-anchor", type=float, default=0.1, help="IoU thr at which MAD pairs are taken")
    args = ap.parse_args()

    gt = None
    if args.mode in ("conflate", "both") and args.labels:
        gt = do_conflate(args.labels, args.out_gt)

    if args.mode in ("compare", "both"):
        if gt is None:
            if args.gt_txt:
                gt = load_pred_txt(args.gt_txt)
            elif args.labels:
                gt = load_labels_dir(args.labels)
            else:
                sys.exit("[error] need --labels or --gt-txt for compare")
        if not args.pred:
            sys.exit("[error] need --pred for compare")
        pred = load_pred_txt(args.pred)
        # enumerate manifest frames so neg frames register (0-episode GT)
        if args.manifest:
            for stem in load_manifest_frames(args.manifest):
                gt.setdefault(stem, [])
        do_compare(gt, pred, args)


if __name__ == "__main__":
    main()
