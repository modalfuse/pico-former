#!/usr/bin/env python3
"""R2-4 / R2-5 (also R1-1, R1-3): full count tables for the 16 event outputs, per-participant results,
participant-clustered CIs, and leave-one-participant-out analysis.

Cohorts (host CPU float32, frozen checkpoint thresholds tau, not reselected):
- azt1d_s20260828 / s20260829 / s20260830: each checkpoint with its own held-out participants;
- ohio, diatrend17: seed-20260828 checkpoint, external test (windows rebuilt by build_external_cache.py);
- ohio_s20260829 etc. (R1-3): checkpoints of the other two seeds tested on the two external cohorts;
  run only when listed in --cohorts.
Usage: python r2_4_r2_5_event_tables.py [--cohorts ohio_s20260829,diatrend17_s20260830,...]

Clustered CI: 2,000 participant-level resamples with replacement (seed 20261009); metrics are recomputed
from summed per-participant counts. Resamples with a zero denominator are excluded and the number of
valid resamples is recorded.
Outputs: results/r2_4_r2_5/<cohort>_table16.csv, <cohort>_per_subject.csv, <cohort>_scores.npz,
         azt1d_s20260828_loso.csv, summary.json
         (with --cohorts, the summary is written to summary_cohorts.json and summary.json is not overwritten)
"""

from __future__ import annotations

import os
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get("PF_WORKSPACE", "."))
SRC = BASE / "rice-former/src"
RUNS = BASE / "runs"
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "r2_4_r2_5"
EXT = HERE / "results" / "external_cache"
N_BOOT = 2000
BOOT_SEED = 20261009

sys.path.insert(0, str(SRC))
from ceqt import EVENT_SPECS, PH_WINDOWS  # noqa: E402
from ceqt.metrics import event_labels  # noqa: E402
from ceqt.models.ceqt import CEQT, CEQTNet  # noqa: E402
from ceqt.ops.ink import InertiaKernel  # noqa: E402

NAMES = [f"{e}_{ph // 60}h" for e, _, _ in EVENT_SPECS for ph in PH_WINDOWS]


def load_model(seed: int):
    blob = torch.load(RUNS / f"20260830_s{seed}/ckpt/ceqt.pt", map_location="cpu", weights_only=False)
    net = CEQTNet()
    net.load_state_dict(blob["state"])
    return CEQT(net=net.eval(), ink=InertiaKernel(), y_mean=float(blob["y_mean"]), y_std=float(blob["y_std"]),
                thresholds=np.asarray(blob["thresholds"], dtype=np.float64))


def cohorts():
    for seed in (20260828, 20260829, 20260830):
        yield f"azt1d_s{seed}", seed, RUNS / f"20260830_s{seed}/cache/azt1d_s{seed}_str15.pkl"
    yield "ohio", 20260828, EXT / "ohio_str15.pkl"
    yield "diatrend17", 20260828, EXT / "diatrend17_str15.pkl"


def extra_cohorts():
    for seed in (20260829, 20260830):
        yield f"ohio_s{seed}", seed, EXT / "ohio_str15.pkl"
        yield f"diatrend17_s{seed}", seed, EXT / "diatrend17_str15.pkl"


def rates(c):
    tp, fp, fn, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"sens": tp / (tp + fn), "spec": tn / (tn + fp), "ppv": tp / (tp + fp), "npv": tn / (tn + fn),
                "fnr": fn / (tp + fn), "fpr": fp / (fp + tn)}


def counts(y, p):
    return np.stack([((p == 1) & (y == 1)).sum(0), ((p == 1) & (y == 0)).sum(0),
                     ((p == 0) & (y == 1)).sum(0), ((p == 0) & (y == 0)).sum(0)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", default="")
    args = ap.parse_args()
    want = [c for c in args.cohorts.split(",") if c]
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    summary = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_boot": N_BOOT, "boot_seed": BOOT_SEED}
    todo = [c for c in (*cohorts(), *extra_cohorts()) if c[0] in want] if want else list(cohorts())
    for name, seed, cache in todo:
        if not cache.exists():
            print(f"skip {name}: {cache} missing", flush=True)
            continue
        t = time.time()
        batch = pickle.loads(cache.read_bytes())["test"]
        # R1 input rule: sensor-limit readings within 2 mg/dL outside 40-400 are clamped (targets unchanged)
        yh = np.asarray(batch.y_hist, np.float64)
        n_clamped = int(((yh < 40) | (yh > 400)).any(1).sum())
        batch.y_hist = np.clip(yh, 40.0, 400.0).astype(np.asarray(batch.y_hist).dtype)
        model = load_model(seed)
        p = model.predict_batch(batch, device=torch.device("cpu"))
        logits, yhat = p["logits"].astype(np.float64), np.asarray(p["yhat"], np.float64)
        tau = model.thresholds
        y = event_labels(batch.y_future, batch.horizons).astype(int)
        pred = (logits >= tau[None, :]).astype(int)
        subj = np.asarray(batch.subject_id)
        yf = np.asarray(batch.y_future, np.float64)
        hz = np.asarray(batch.horizons, float)
        ae = np.abs(yhat - yf)
        # MAE over the 0-1 h and 0-4 h prefixes, as in ceqt.eval.layer1_table
        err = np.stack([np.nanmean(ae[:, hz <= 60], axis=1), np.nanmean(ae[:, hz <= 240], axis=1)], 1)
        i1, i4 = 0, 1
        np.savez_compressed(OUT / f"{name}_scores.npz", logits=logits, y=y, tau=tau, subject=subj,
                            ae1=err[:, i1], ae4=err[:, i4])
        S = np.unique(subj)
        per_c = np.stack([counts(y[subj == s], pred[subj == s]) for s in S])  # (n_subj, 16, 4)
        tot = per_c.sum(0)
        rng = np.random.default_rng(BOOT_SEED)
        draws = rng.integers(0, len(S), size=(N_BOOT, len(S)))
        boot = per_c[draws].sum(1)  # (N_BOOT, 16, 4)
        r_tot, r_boot = rates(tot), rates(boot)
        rows = []
        for k, nm in enumerate(NAMES):
            row = {"output": nm, "n_pos": int(tot[k, 0] + tot[k, 2]), "n_neg": int(tot[k, 1] + tot[k, 3]),
                   "tp": int(tot[k, 0]), "fp": int(tot[k, 1]), "fn": int(tot[k, 2]), "tn": int(tot[k, 3]),
                   "tau": float(tau[k])}
            for m in ("sens", "spec", "ppv", "npv", "fnr", "fpr"):
                v = r_boot[m][:, k]
                v = v[np.isfinite(v)]
                row[m] = float(r_tot[m][k])
                row[f"{m}_lo"] = float(np.percentile(v, 2.5)) if v.size else np.nan
                row[f"{m}_hi"] = float(np.percentile(v, 97.5)) if v.size else np.nan
                row[f"{m}_nboot"] = int(v.size)
            row["auroc"] = float(roc_auc_score(y[:, k], logits[:, k])) if 0 < y[:, k].sum() < len(y) else np.nan
            row["n_subj_with_pos"] = int(((per_c[:, k, 0] + per_c[:, k, 2]) > 0).sum())
            rows.append(row)
        t16 = pd.DataFrame(rows)
        t16.to_csv(OUT / f"{name}_table16.csv", index=False)
        prow = []
        for j, s in enumerate(S):
            m = subj == s
            r = rates(per_c[j])
            d = {"subject": int(s) if str(s).isdigit() else str(s), "n_windows": int(m.sum()),
                 "mae_1h": float(np.nanmean(err[m, i1])), "mae_4h": float(np.nanmean(err[m, i4]))}
            for k, nm in enumerate(NAMES):
                d[f"{nm}_pos"] = int(per_c[j, k, 0] + per_c[j, k, 2])
                d[f"{nm}_fnr"] = float(r["fnr"][k])
                d[f"{nm}_fpr"] = float(r["fpr"][k])
            prow.append(d)
        ps = pd.DataFrame(prow)
        ps.to_csv(OUT / f"{name}_per_subject.csv", index=False)
        summ = {"n_windows": len(y), "n_windows_cgm_clamped": n_clamped, "n_subjects": len(S), "mae_1h": float(np.nanmean(err[:, i1])),
                "mae_4h": float(np.nanmean(err[:, i4])), "seconds": time.time() - t}
        for nm in ("hypo70_4h", "hyper180_4h", "hypo70_2h", "hyper180_2h"):
            r = t16[t16.output == nm].iloc[0]
            col = ps[f"{nm}_fnr"]
            summ[nm] = {"fnr": r.fnr, "fnr_ci": [r.fnr_lo, r.fnr_hi], "fpr": r.fpr, "fpr_ci": [r.fpr_lo, r.fpr_hi],
                        "ppv": r.ppv, "auroc": r.auroc, "subject_fnr_min": float(col.min()),
                        "subject_fnr_max": float(col.max()),
                        "subject_fpr_min": float(ps[f"{nm}_fpr"].min()), "subject_fpr_max": float(ps[f"{nm}_fpr"].max())}
        summary[name] = summ
        print(name, json.dumps(summ), flush=True)
        if name == "azt1d_s20260828":
            lo = []
            for j, s in enumerate(S):
                r = rates(np.delete(per_c, j, axis=0).sum(0))
                lo.append({"dropped": int(s), **{f"{nm}_{m}": float(r[m][k]) for k, nm in enumerate(NAMES)
                                                 for m in ("fnr", "fpr", "ppv")}})
            pd.DataFrame(lo).to_csv(OUT / f"{name}_loso.csv", index=False)
    (OUT / ("summary_cohorts.json" if want else "summary.json")).write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
