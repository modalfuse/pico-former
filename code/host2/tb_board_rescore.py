#!/usr/bin/env python3
"""R2-17 T-B: re-score board/host agreement under the deployment conditions.

run_tb_board.py sends every injected window to the board and uses the status from r2_17_tb_host.py as the expected value. It differs from deployment in two respects:
  1. windows flagged GAP / STALE by the preprocessing layer (P_*) form no frame and are not sent to the node in deployment; tb_frames keeps clean frames for them,
     so the node returning OK is correct behaviour, and these windows are excluded;
  2. the host reference pf2.guard lacks the firmware output check M1 (0 <= yhat <= CEQT_YHAT_MAX and finite); it is added on the host side here.
Output: results/r2_17/tb_board/tb_board_rescored.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf2  # noqa: E402

RES = HERE.parent / "results" / "r2_17"
CKPT = pf2.BASE / "runs/20260830_s20260828/ckpt/ceqt.pt"
YHAT_MAX = 600.0


def main():
    _, _, _, thr = pf2.load_model(CKPT)
    rows, lmax = [], 0.0
    for f in sorted((RES / "tb_frames").glob("*.npz")):
        z, b = np.load(f), np.load(RES / "tb_board" / f.name)
        st = b["status"]
        m = len(st)
        es = z["exp_status"][:m].copy()
        ey, el = z["exp_yhat"][:m], z["exp_logits"][:m]
        sent = es == pf2.OK if f.stem.startswith("P_") else np.ones(m, bool)
        m1 = (es == pf2.OK) & ~(np.isfinite(ey) & (ey >= 0) & (ey <= YHAT_MAX)).all(1)
        es[m1] = pf2.FAULT_OUTPUT
        both = sent & (st == pf2.OK) & (es == pf2.OK)
        dis = ((b["logits"] >= thr) != (el >= thr)) & both[:, None]
        lm = float(np.abs(b["logits"][both] - el[both]).max()) if both.any() else None
        if lm is not None:
            lmax = max(lmax, lm)
        rows.append({"injection": f.stem, "n": m, "not_sent": int((~sent).sum()), "sent": int(sent.sum()),
                     "status_agree_sent": int((st[sent] == es[sent]).sum()), "host_m1": int(m1.sum()),
                     "board_fault_output": int((st == pf2.FAULT_OUTPUT).sum()),
                     "evt_cells_compared": int(both.sum() * 16), "evt_cells_disagree": int(dis.sum()),
                     "logit_maxabs": lm})
    keys = ("n", "not_sent", "sent", "status_agree_sent", "host_m1", "board_fault_output",
            "evt_cells_compared", "evt_cells_disagree")
    summary = {k: int(sum(r[k] for r in rows)) for k in keys}
    summary.update(injections=len(rows), logit_maxabs=lmax)
    (RES / "tb_board" / "tb_board_rescored.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for r in rows:
        if r["not_sent"] or r["host_m1"] or r["status_agree_sent"] != r["sent"]:
            print(r)


if __name__ == "__main__":
    main()
