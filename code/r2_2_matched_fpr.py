#!/usr/bin/env python3
"""R2-2: event heads vs trajectory-threshold readout at matched FPR, and ROC and PR for the 16 outputs.

Model: RICE-Former checkpoint (seed 20260828), exact GELU, CPU float32.
Scores: event heads use the logit; the trajectory readout uses -min(y_hat) for hypo and max(y_hat) for hyper,
over (0, H].
Operating points (all selected on validation participants only; the held-out set is not used for selection):
- OP0 original submission: event heads use the frozen checkpoint thresholds tau; the trajectory readout uses the
  clinical thresholds themselves (70/54/180/250 mg/dL);
- OP1 matched FPR: the trajectory cut-off whose validation FPR is closest to the event head's validation FPR;
- OP2 same rule: both readouts selected by calibrate_threshold (highest recall with validation FPR <= 0.06).
Also reports FNR at matched FPR on the held-out set by ROC interpolation (descriptive), and
participant-level clustered bootstrap 95% CIs.
"""

from __future__ import annotations

import os
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

BASE = Path(os.environ.get("PF_WORKSPACE", "."))
SRC = BASE / "rice-former/src"
RUN = BASE / "runs/20260830_s20260828"
OUT = Path(__file__).resolve().parent / "results" / "r2_2"
N_BOOT = 2000
BOOT_SEED = 20261009

sys.path.insert(0, str(SRC))
from ceqt import EVENT_SPECS, PH_WINDOWS  # noqa: E402
from ceqt.metrics import calibrate_threshold, event_labels, prefix  # noqa: E402
from ceqt.models.ceqt import CEQT, CEQTNet  # noqa: E402
from ceqt.ops.ink import InertiaKernel  # noqa: E402


def load():
    blob = torch.load(RUN / "ckpt/ceqt.pt", map_location="cpu", weights_only=False)
    net = CEQTNet()
    net.load_state_dict(blob["state"])
    model = CEQT(net=net.eval(), ink=InertiaKernel(), y_mean=float(blob["y_mean"]), y_std=float(blob["y_std"]),
                 thresholds=np.asarray(blob["thresholds"], dtype=np.float64))
    with (RUN / "cache/azt1d_s20260828_str15.pkl").open("rb") as f:
        cache = pickle.load(f)
    return model, np.asarray(blob["thresholds"], dtype=np.float64), cache["val"], cache["test"]


def scores(model, batch):
    p = model.predict_batch(batch, device=torch.device("cpu"))
    yhat, logits = p["yhat"], p["logits"].astype(np.float64)
    traj = np.zeros_like(logits)
    k = 0
    for _name, thr, mode in EVENT_SPECS:
        for ph in PH_WINDOWS:
            path = prefix(yhat, batch.horizons, ph)
            traj[:, k] = -np.min(path, axis=1) if mode == "below" else np.max(path, axis=1)
            k += 1
    return logits, traj, event_labels(batch.y_future, batch.horizons).astype(int)


def rates(lab, pred):
    tp = int(((pred == 1) & (lab == 1)).sum()); fn = int(((pred == 0) & (lab == 1)).sum())
    fp = int(((pred == 1) & (lab == 0)).sum()); tn = int(((pred == 0) & (lab == 0)).sum())
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "fnr": fn / (tp + fn) if tp + fn else float("nan"),
            "fpr": fp / (fp + tn) if fp + tn else float("nan")}


def match_cut(score, lab, target_fpr):
    neg = np.sort(score[lab == 0])[::-1]
    n_fp = int(round(target_fpr * len(neg)))
    if n_fp <= 0:
        return float(neg[0] + 1e-6)
    return float(neg[min(n_fp, len(neg)) - 1])


def fnr_at_fpr(score, lab, fpr_target):
    fpr, tpr, _ = roc_curve(lab, score)
    return float(1.0 - np.interp(fpr_target, fpr, tpr))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    model, tau, val, test = load()
    lv, tv, yv = scores(model, val)
    lt, tt, yt = scores(model, test)
    subj = np.asarray(test.subject_id)
    subjects = np.unique(subj)
    rng = np.random.default_rng(BOOT_SEED)
    boots = [np.concatenate([np.where(subj == s)[0] for s in rng.choice(subjects, len(subjects))])
             for _ in range(N_BOOT)]

    rows = []
    k = 0
    for name, thr, mode in EVENT_SPECS:
        for ph in PH_WINDOWS:
            r = {"event": name, "ph_min": ph, "n_pos_val": int(yv[:, k].sum()), "n_val": len(yv),
                 "n_pos_test": int(yt[:, k].sum()), "n_test": len(yt)}
            head_val = rates(yv[:, k], (lv[:, k] >= tau[k]).astype(int))
            c1 = match_cut(tv[:, k], yv[:, k], head_val["fpr"])
            tau2_head = calibrate_threshold(lv[:, k], yv[:, k], 0.06)
            c2 = calibrate_threshold(tv[:, k], yv[:, k], 0.06)
            r["val_head_fpr"] = head_val["fpr"]
            r["OP0"] = {"head": rates(yt[:, k], (lt[:, k] >= tau[k]).astype(int)),
                        "traj": rates(yt[:, k], (tt[:, k] > (-thr if mode == "below" else thr)).astype(int))}
            r["OP1"] = {"traj_cut_mgdl": float(-c1 if mode == "below" else c1),
                        "traj_val_fpr": float((tv[yv[:, k] == 0, k] >= c1).mean()),
                        "head": r["OP0"]["head"],
                        "traj": rates(yt[:, k], (tt[:, k] >= c1).astype(int))}
            r["OP2"] = {"traj_cut_mgdl": float(-c2 if mode == "below" else c2),
                        "head_tau_check": float(tau2_head), "head_tau_ckpt": float(tau[k]),
                        "head": r["OP0"]["head"],
                        "traj": rates(yt[:, k], (tt[:, k] >= c2).astype(int))}
            if yt[:, k].sum() and (1 - yt[:, k]).sum():
                r["auroc"] = {"head": float(roc_auc_score(yt[:, k], lt[:, k])),
                              "traj": float(roc_auc_score(yt[:, k], tt[:, k]))}
                r["auprc"] = {"head": float(average_precision_score(yt[:, k], lt[:, k])),
                              "traj": float(average_precision_score(yt[:, k], tt[:, k]))}
                hf = r["OP0"]["head"]["fpr"]
                r["roc_fnr_at_head_fpr"] = {"head": r["OP0"]["head"]["fnr"],
                                            "traj": fnr_at_fpr(tt[:, k], yt[:, k], hf)}
                d_fnr, d_auc = [], []
                for idx in boots:
                    y = yt[idx, k]
                    if y.sum() == 0 or y.sum() == len(y):
                        continue
                    fh = rates(y, (lt[idx, k] >= tau[k]).astype(int))["fnr"]
                    ft = rates(y, (tt[idx, k] >= c1).astype(int))["fnr"]
                    d_fnr.append(ft - fh)
                    d_auc.append(roc_auc_score(y, lt[idx, k]) - roc_auc_score(y, tt[idx, k]))
                r["boot"] = {"n": len(d_fnr),
                             "OP1_fnr_traj_minus_head": [float(np.percentile(d_fnr, q)) for q in (2.5, 50, 97.5)],
                             "auroc_head_minus_traj": [float(np.percentile(d_auc, q)) for q in (2.5, 50, 97.5)]}
                fh_, th_, _ = roc_curve(yt[:, k], lt[:, k])
                ft_, tt_, _ = roc_curve(yt[:, k], tt[:, k])
                np.savez_compressed(OUT / f"roc_{name}_{ph}.npz", fpr_head=fh_, tpr_head=th_, fpr_traj=ft_, tpr_traj=tt_)
            rows.append(r)
            k += 1
    (OUT / "r2_2_matched_fpr.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    np.savez_compressed(OUT / "scores.npz", logits_val=lv, traj_val=tv, y_val=yv, logits_test=lt, traj_test=tt,
                        y_test=yt, subject_test=subj, tau=tau)

    hdr = "event ph | val_hFPR | OP0 head FNR/FPR | OP0 traj | OP1 cut traj FNR/FPR | OP2 cut traj | AUROC h/t | AUPRC h/t | dFNR CI | dAUC CI"
    print(hdr)
    for r in rows:
        f = lambda d: f"{d['fnr']:.3f}/{d['fpr']:.3f}"  # noqa: E731
        b = r.get("boot", {})
        print(f"{r['event']} {r['ph_min']} | {r['val_head_fpr']:.3f} | {f(r['OP0']['head'])} | {f(r['OP0']['traj'])} | "
              f"{r['OP1']['traj_cut_mgdl']:.1f} {f(r['OP1']['traj'])} | {r['OP2']['traj_cut_mgdl']:.1f} {f(r['OP2']['traj'])} | "
              f"{r['auroc']['head']:.3f}/{r['auroc']['traj']:.3f} | {r['auprc']['head']:.3f}/{r['auprc']['traj']:.3f} | "
              f"{np.round(b.get('OP1_fnr_traj_minus_head', []), 3)} | {np.round(b.get('auroc_head_minus_traj', []), 3)}")


if __name__ == "__main__":
    main()
