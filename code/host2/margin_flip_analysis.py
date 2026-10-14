"""Decision margin and decision-change analysis.

For each audit result audit.npz:
- margin m = |host float64 logit - threshold|, board error e = board float32 logit - host logit;
- list cells with |e| > m (numerical error large enough to cross the threshold) and cells whose margin is below the max logit difference;
- randomly assign the measured signed errors of each output to its cells to estimate the expected number of decision changes and the probability of zero changes (Poisson approximation).

Usage: python host2/margin_flip_analysis.py [audit_dir ...]
Results are written to results/margin_flip/summary.json for the five full sets (no arguments), otherwise to
results/margin_flip/summary_<audit_dir>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = [f"{e}@{h}h" for e in ("hypo70", "hypo54", "hyper180", "hyper250") for h in (1, 2, 3, 4)]
DEFAULT = ["azt1d_full", "s20260829_full", "s20260830_full", "ohio_full", "devkitc_diatrend17_full"]


def analyse(tag):
    d = np.load(ROOT / "results/audit" / tag / "audit.npz")
    ok = d["status"] == 0
    idx = d["idx"][ok]
    ref = d["ref_logits"][ok]
    b = d["logits"][ok].astype(np.float64)
    thr = d["thr"].astype(np.float64)
    evt = d["evt"][ok]
    s = ref - thr
    m = np.abs(s)
    e = b - ref
    ae = np.abs(e)
    toward = np.sign(e) == -np.sign(s)
    emax = float(ae.max())

    expected = 0.0
    for j in range(16):
        ej = np.sort(e[:, j])
        n = len(ej)
        p_neg = np.searchsorted(ej, -m[:, j], side="left") / n
        p_pos = 1 - np.searchsorted(ej, m[:, j], side="left") / n
        expected += float(np.where(s[:, j] >= 0, p_neg, p_pos).sum())

    def cell(w, j):
        return {
            "idx": int(idx[w]), "output": NAMES[j], "thr": thr[j], "host": float(ref[w, j]),
            "board": float(b[w, j]), "margin": float(m[w, j]), "diff": float(e[w, j]),
            "toward_threshold": bool(toward[w, j]), "evt_board": int(evt[w, j]),
            "evt_host": int(ref[w, j] >= thr[j]),
            "diff_percentile": float((ae < ae[w, j]).mean()),
            "float32_spacing": float(abs(np.spacing(np.float32(b[w, j])))),
        }

    return {
        "windows": int(ok.sum()), "cells": int(ok.sum() * 16),
        "mismatch_cells": int((evt != (ref >= thr)).sum()),
        "logit_maxabs": emax, "min_margin": float(m.min()),
        "frac_toward_threshold": float(toward.mean()),
        "n_margin_lt_maxdiff": int((m < emax).sum()),
        "margin_lt_maxdiff": [cell(w, j) for w, j in np.argwhere(m < emax)],
        "diff_gt_margin": [cell(w, j) for w, j in np.argwhere(ae > m)],
        "expected_flips": expected, "p_zero_flips": float(np.exp(-expected)),
    }


def main():
    tags = sys.argv[1:] or DEFAULT
    out = {}
    for t in tags:
        if not (ROOT / "results/audit" / t / "audit.npz").exists():
            print(f"skip {t}: no audit.npz")
            continue
        r = analyse(t)
        out[t] = r
        print(f"{t}: cells {r['cells']}, mismatches {r['mismatch_cells']}, max logit diff {r['logit_maxabs']:.3e}, "
              f"min margin {r['min_margin']:.3e}, margin<max diff {r['n_margin_lt_maxdiff']}, "
              f"|e|>m {len(r['diff_gt_margin'])}, expected decision changes {r['expected_flips']:.3f}")
    total = sum(r["expected_flips"] for r in out.values())
    out["_total"] = {"expected_flips": total, "p_zero_flips": float(np.exp(-total))}
    print(f"total expected decision changes {total:.3f}, P(0 changes) {np.exp(-total):.3f}")
    dst = ROOT / "results/margin_flip"
    dst.mkdir(parents=True, exist_ok=True)
    name = "summary.json" if tags == DEFAULT else f"summary_{'_'.join(tags)}.json"
    (dst / name).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
