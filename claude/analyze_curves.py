#!/usr/bin/env python
"""Summarize FastSAC debug training curves: first vs last mean_q / reward / etc.
Usage: analyze_curves.py <glob-substring e.g. dbg->"""
import csv, glob, sys, os

LOGROOT = "logs/fast_sac/g1_paper_fast_sac"
needle = sys.argv[1] if len(sys.argv) > 1 else "dbg-"


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def first_updated(rows, col):
    """first row where the metric column is populated (updates started)."""
    for r in rows:
        if fnum(r.get(col)) is not None:
            return r
    return None


dirs = sorted(glob.glob(f"{LOGROOT}/*{needle}*/"))
print(f"{'run':<26} {'rows':>4} {'mq0':>9} {'mqN':>9} {'dMq':>8} "
      f"{'rew0':>8} {'rewN':>8} {'tqN':>9} {'closs':>8} {'termN':>7} {'alphaN':>8}  verdict")
for d in dirs:
    csvp = os.path.join(d, "training_curve.csv")
    if not os.path.exists(csvp):
        continue
    rows = load(csvp)
    if not rows:
        continue
    name = os.path.basename(d.rstrip("/"))[20:]  # strip timestamp prefix
    r0 = first_updated(rows, "mean_q") or rows[0]
    rN = rows[-1]
    mq0, mqN = fnum(r0.get("mean_q")), fnum(rN.get("mean_q"))
    rew0, rewN = fnum(r0.get("reward_mean")), fnum(rN.get("reward_mean"))
    tqN = fnum(rN.get("target_q_mean"))
    closs = fnum(rN.get("critic_loss"))
    termN = fnum(rN.get("terminated_rate"))
    alphaN = fnum(rN.get("alpha"))
    dmq = (mqN - mq0) if (mqN is not None and mq0 is not None) else None
    # verdict per plan section 5
    verdict = "?"
    if mqN is not None and mq0 is not None and rew0 is not None and rewN is not None:
        if dmq >= 20 and rewN > rew0:
            verdict = "PASS"
        elif mqN <= -45 and abs(rewN - rew0) < 1:
            verdict = "FAIL"
        else:
            verdict = "mixed"

    def f(x, w=8, p=2):
        return ("{:>%d.%df}" % (w, p)).format(x) if x is not None else " " * (w - 3) + "n/a"
    print(f"{name:<26} {len(rows):>4} {f(mq0,9):>9} {f(mqN,9):>9} {f(dmq,8):>8} "
          f"{f(rew0):>8} {f(rewN):>8} {f(tqN,9):>9} {f(closs):>8} {f(termN,7,3):>7} {f(alphaN,8,4):>8}  {verdict}")
