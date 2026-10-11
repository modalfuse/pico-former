#!/usr/bin/env python3
"""R1-1: rebuild OhioT1DM (12 test participants) and DiaTrend (17 participants with basal-rate records) windows
with the data-loading and window-construction code of the RICE-Former release package.

Checks that window counts match the release tables P1_window_stats_*.csv (Ohio 8,619; DiaTrend17 115,934), and
recomputes 4 h hypo70 / hyper180 FNR/FPR on the host with the checkpoint (seed 20260828), compared with
main_ohio_event.csv and main_diatrend17_event.csv.

Outputs: results/external_cache/{ohio,diatrend17}_str15.pkl (dict, key test), results/external_cache/check.json
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "host2"))
import pf2  # noqa: E402
from ceqt.data.cache import load_diatrend_windows, load_ohio_windows  # noqa: E402
from ceqt.data.diatrend import BASAL_SUBJECTS  # noqa: E402

OUT = HERE / "results" / "external_cache"
TABLES = pf2.BASE / "rice-former/results/tables"
CKPT = pf2.BASE / "runs/20260830_s20260828/ckpt/ceqt.pt"


def host_rates(batch, net, y_mean, y_std, thr):
    fr = pf2.frames(batch)
    logits, _, _ = pf2.host_forward(net, y_mean, y_std, fr["y"], fr["feat"], fr["mask"], fr["n_marks"], fr["q"],
                                    fr["iner"], dtype=__import__("torch").float32)
    lab = pf2.labels(batch)
    ev = (logits >= thr[None, :]).astype(int)
    out = {}
    for name, col in (("hypo70_4h", 3), ("hyper180_4h", 11)):
        y, p = lab[:, col], ev[:, col]
        out[name] = {"fnr": float(((p == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
                     "fpr": float(((p == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))}
    out["n_trunc"] = int((fr["n_marks"] > pf2.N_MARK).sum())
    out["n_guard_drop"] = int(sum(pf2.guard(fr["y"][i], fr["feat"][i], fr["mask"][i], fr["q"][i], fr["flags"][i],
                                            fr["t0_s"][i], fr["t_last_s"][i])[-1] > 0 for i in range(len(fr["y"]))))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    net, y_mean, y_std, thr = pf2.load_model(CKPT)
    check = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "ckpt": str(CKPT)}
    for name, loader, stats in (("ohio", lambda: load_ohio_windows(), "P1_window_stats_ohio.csv"),
                                ("diatrend17", lambda: load_diatrend_windows(BASAL_SUBJECTS),
                                 "P1_window_stats_diatrend.csv")):
        t = time.time()
        path = OUT / f"{name}_str15.pkl"
        if path.exists():
            batch = pickle.loads(path.read_bytes())["test"]
        else:
            batch = loader()
            path.write_bytes(pickle.dumps({"test": batch}))
        ref = pd.read_csv(TABLES / stats)
        subj = np.asarray(batch.subject_id)
        per = {int(s): int((subj == s).sum()) for s in np.unique(subj)}
        ref_per = {int(r.subject): int(r.n_windows) for r in ref.itertuples()}
        r = host_rates(batch, net, y_mean, y_std, thr)
        check[name] = {"n_windows": len(batch), "n_windows_release": int(ref.n_windows.sum()),
                       "per_subject_match": per == ref_per, "per_subject": per, "build_s": time.time() - t, **r}
        print(name, json.dumps({k: v for k, v in check[name].items() if k != "per_subject"}), flush=True)
    (OUT / "check.json").write_text(json.dumps(check, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
