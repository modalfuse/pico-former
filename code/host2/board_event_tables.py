"""R2-4 / R2-5: compute counts for the 16 event outputs from the decision bits of the full-set board replay and compare them cell by cell with the host float32 tables (r2_4_r2_5_event_tables.py).

Labels y and participants come from results/r2_4_r2_5/<cohort>_scores.npz (written by the host script from the same cache in the same window order);
board decisions come from evt in results/audit/<board dir>/audit.npz (aligned by idx).
Output: results/r2_4_r2_5/board/<cohort>__<board dir>_table16.csv and board_vs_host.json.
Usage: python board_event_tables.py [cohort:board_dir ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
R = HERE / "results"
EVTS = ["hypo70", "hypo54", "hyper180", "hyper250"]
NAMES = [f"{e}_{h}h" for e in EVTS for h in (1, 2, 3, 4)]
DEFAULT = ["azt1d_s20260828:azt1d_full", "azt1d_s20260829:s20260829_full", "azt1d_s20260830:s20260830_full",
           "ohio:ohio_full", "diatrend17:devkitc_diatrend17_full"]


def counts(pred, y):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    return tp, fp, fn, tn


def main():
    pairs = sys.argv[1:] or DEFAULT
    out = R / "r2_4_r2_5" / "board"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for p in pairs:
        cohort, run = p.split(":")
        s = np.load(R / "r2_4_r2_5" / f"{cohort}_scores.npz")
        a = np.load(R / "audit" / run / "audit.npz")
        assert len(a["idx"]) == len(s["y"]) and np.array_equal(a["idx"], np.arange(len(s["y"]))), p
        y = s["y"].astype(bool)
        host = s["logits"] >= s["tau"][None, :]
        board = a["evt"].astype(bool)
        ok = a["status"] == 0
        rows = []
        for k, name in enumerate(NAMES):
            tp, fp, fn, tn = counts(board[:, k], y[:, k])
            htp, hfp, hfn, htn = counts(host[:, k], y[:, k])
            rows.append(dict(output=name, pos=tp + fn, neg=fp + tn, TP=tp, FP=fp, FN=fn, TN=tn,
                             sens=tp / max(tp + fn, 1), spec=tn / max(fp + tn, 1),
                             FNR=fn / max(tp + fn, 1), FPR=fp / max(fp + tn, 1),
                             same_as_host=(tp, fp, fn, tn) == (htp, hfp, hfn, htn)))
        pd.DataFrame(rows).to_csv(out / f"{cohort}__{run}_table16.csv", index=False)
        summary[p] = dict(windows=int(len(y)), status_ok=int(ok.sum()),
                          tau_equal=bool(np.allclose(s["tau"], a["thr"], rtol=0, atol=1e-6)),
                          cells_board_vs_host_f32=int((board != host).sum()),
                          cells_board_vs_ref_f64=int((board != (a["ref_logits"] >= a["thr"][None, :])).sum()),
                          rows_same_as_host=int(sum(r["same_as_host"] for r in rows)),
                          hypo70_4h=dict(FNR=rows[3]["FNR"], FPR=rows[3]["FPR"]),
                          hyper180_4h=dict(FNR=rows[11]["FNR"], FPR=rows[11]["FPR"]))
        print(p, summary[p], flush=True)
    (out / "board_vs_host.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
