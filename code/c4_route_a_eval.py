"""Fixed released checkpoints and thresholds on origin-causal windows.

Predictive metrics use the CPU. Logits are stored as float64. The validation
curve cuts of Table S4b are chosen on the symmetric-resampling validation cache and then
read on the evaluated windows. The matched-FPR miss rate is a retrospective
read of the evaluated ROC.

  PF_RICE_SRC  directory that contains the ceqt package
  PF_RUNS      released training-run directory
  PF_EXT_CACHE symmetric-resampling external-cache directory
  PF_C3        C3 causal-window run directory
  PF_OUT_C4    C4 metric-output directory (created)
  PF_OUT_C5    C5 comparison-output directory (created)
  RICE_DATA_DIR is unused here; checkpoints and caches are opened directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, os.environ["PF_RICE_SRC"])

from ceqt import EVENT_SPECS, PH_WINDOWS  # noqa: E402
from ceqt.metrics import calibrate_threshold, event_labels, prefix  # noqa: E402
from ceqt.models.ceqt import CEQT, CEQTNet  # noqa: E402
from ceqt.ops.ink import InertiaKernel  # noqa: E402

SEEDS = (20260828, 20260829, 20260830)
CKPT_SHA = {
    20260828: "18175b249372da8e50338ddcbcc5482d1a6171ecc219c2fc8114ee3f90f35ba8",
    20260829: "801ef1cf50e8ba278d9baec2ea523f688a42f95c7c29279acd28d1518f67c8f9",
    20260830: "fd696e298552cb685ea362271897a370262bcd25d55d8d89ef960a717119d02d",
}
THR_SHA = {
    20260828: "2ab0c6f5448f885922bc463d362b86ee6fd6cf199b32ee736a534835f58f3217",
    20260829: "aca156b8eebdee615c9a2ef720b317b037b2c284fd02abbfa834f7d6e6f82630",
    20260830: "e8edf8bc60ce2f26385effe53970fdef5a548a09b71fb71f23b2fe0563dcf0af",
}
PAPER = {"auroc_head": 0.748, "auroc_curve": 0.671, "head_fnr": 0.669, "curve_fnr": 0.701}
N_BOOT = 2000
BOOT_SEED = 20261113
HYPO70_4H = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(runs: Path, seed: int):
    path = runs / f"20260830_s{seed}" / "ckpt" / "ceqt.pt"
    digest = sha256_file(path)
    if digest != CKPT_SHA[seed]:
        raise SystemExit(f"checkpoint hash mismatch for {seed}: {digest}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    tau = np.asarray(blob["thresholds"], dtype=np.float64)
    tau32 = np.asarray(blob["thresholds"], dtype="<f4")
    td = hashlib.sha256(tau32.tobytes()).hexdigest()
    if td != THR_SHA[seed]:
        raise SystemExit(f"threshold hash mismatch for {seed}: {td}")
    net = CEQTNet()
    net.load_state_dict(blob["state"])
    model = CEQT(net=net.eval(), ink=InertiaKernel(), y_mean=float(blob["y_mean"]), y_std=float(blob["y_std"]),
                 thresholds=tau)
    return model, tau, {"y_mean": float(blob["y_mean"]), "y_std": float(blob["y_std"]),
                        "checkpoint_sha256": digest, "threshold_sha256": td}


def scores(model, batch):
    pred = model.predict_batch(batch, device=torch.device("cpu"))
    yhat = pred["yhat"].astype(np.float64)
    logits = pred["logits"].astype(np.float64)
    curve = np.zeros_like(logits)
    k = 0
    for _name, _thr, mode in EVENT_SPECS:
        for ph in PH_WINDOWS:
            path = prefix(yhat, batch.horizons, ph)
            curve[:, k] = -np.min(path, axis=1) if mode == "below" else np.max(path, axis=1)
            k += 1
    labels = event_labels(batch.y_future, batch.horizons).astype(int)
    return yhat, logits, curve, labels


def rates(lab, pred):
    tp = int(((pred == 1) & (lab == 1)).sum())
    fn = int(((pred == 0) & (lab == 1)).sum())
    fp = int(((pred == 1) & (lab == 0)).sum())
    tn = int(((pred == 0) & (lab == 0)).sum())
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "n_pos": tp + fn, "n": int(len(lab)),
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "ppv": tp / (tp + fp) if tp + fp else None,
        "npv": tn / (tn + fn) if tn + fn else None,
        "fnr": fn / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
    }


def fnr_at_fpr(score, lab, fpr_target):
    fpr, tpr, _ = roc_curve(lab, score)
    return float(1.0 - np.interp(fpr_target, fpr, tpr))


def mae_hours(y_true, yhat, horizons):
    out = {}
    for ph in PH_WINDOWS:
        m = horizons <= ph + 1e-6
        out[str(int(ph))] = float(np.mean(np.abs(yhat[:, m] - y_true[:, m])))
    return out


def match_cut(score, lab, target_fpr):
    neg = np.sort(score[lab == 0])[::-1]
    n_fp = int(round(target_fpr * len(neg)))
    if n_fp <= 0:
        return float(neg[0] + 1e-6)
    return float(neg[min(n_fp, len(neg)) - 1])


def output_rows(logits, curve, labels, tau, boots):
    rows = []
    k = 0
    for name, _thr, _mode in EVENT_SPECS:
        for ph in PH_WINDOWS:
            y = labels[:, k]
            row = {"event": name, "ph_min": int(ph), "n_pos": int(y.sum()), "n": int(len(y))}
            if 0 < y.sum() < len(y):
                head = rates(y, (logits[:, k] >= tau[k]).astype(int))
                row.update({
                    "auroc_head": float(roc_auc_score(y, logits[:, k])),
                    "auroc_curve": float(roc_auc_score(y, curve[:, k])),
                    "auprc_head": float(average_precision_score(y, logits[:, k])),
                    "auprc_curve": float(average_precision_score(y, curve[:, k])),
                    "head": {kk: head[kk] for kk in ("sensitivity", "specificity", "ppv", "npv", "fnr", "fpr", "tp", "fn", "fp", "tn")},
                    "curve_fnr_at_head_fpr": fnr_at_fpr(curve[:, k], y, head["fpr"]),
                })
                d_auc, d_fnr = [], []
                for idx in boots:
                    yb = y[idx]
                    if yb.sum() == 0 or yb.sum() == len(yb):
                        continue
                    st = rates(yb, (logits[idx, k] >= tau[k]).astype(int))
                    d_auc.append(roc_auc_score(yb, logits[idx, k]) - roc_auc_score(yb, curve[idx, k]))
                    d_fnr.append(fnr_at_fpr(curve[idx, k], yb, st["fpr"]) - st["fnr"])
                row["boot"] = {
                    "n_valid": len(d_auc),
                    "n_drawn": len(boots),
                    "auroc_head_minus_curve": [float(np.percentile(d_auc, q)) for q in (2.5, 50, 97.5)] if d_auc else None,
                    "fnr_curve_minus_head": [float(np.percentile(d_fnr, q)) for q in (2.5, 50, 97.5)] if d_fnr else None,
                }
            rows.append(row)
            k += 1
    return rows


def bootstrap_indices(subject_id):
    subj = np.asarray(subject_id)
    subjects = np.unique(subj)
    rng = np.random.default_rng(BOOT_SEED)
    groups = {int(s): np.where(subj == s)[0] for s in subjects}
    keys = [int(s) for s in subjects]
    return [np.concatenate([groups[int(s)] for s in rng.choice(keys, len(keys))]) for _ in range(N_BOOT)]


def fpr_over_budget(rows):
    n = 0
    for row in rows:
        fpr = (row.get("head") or {}).get("fpr")
        if fpr is not None and fpr > 0.06:
            n += 1
    return n


def _nominal_as_batch(batch, nominal):
    """Expose nominal origins in t0 so key_mask can join them to the cache."""
    clone = type(batch)(
        subject_id=batch.subject_id,
        t0=np.asarray(nominal, float),
        y_hist=batch.y_hist,
        t_hist=batch.t_hist,
        y_future=batch.y_future,
        horizons=batch.horizons,
        mark_t=batch.mark_t,
        mark_a=batch.mark_a,
        mark_e=batch.mark_e,
        day_index=batch.day_index,
        n_marks=batch.n_marks,
    )
    return clone


def subset(batch, nominal, mask):
    from ceqt.data.windows import WindowBatch
    idx = np.flatnonzero(mask)
    sl = idx
    return WindowBatch(
        subject_id=batch.subject_id[sl],
        t0=batch.t0[sl],
        y_hist=batch.y_hist[sl],
        t_hist=batch.t_hist[sl],
        y_future=batch.y_future[sl],
        horizons=batch.horizons,
        mark_t=[batch.mark_t[i] for i in sl],
        mark_a=[batch.mark_a[i] for i in sl],
        mark_e=[batch.mark_e[i] for i in sl],
        day_index=batch.day_index[sl],
        n_marks=batch.n_marks[sl],
    ), nominal[sl]


def key_mask(batch, nominal, other_batch):
    want = set(zip(np.asarray(other_batch.subject_id, int).tolist(), np.round(np.asarray(other_batch.t0, float), 6).tolist()))
    keys = list(zip(np.asarray(batch.subject_id, int).tolist(), np.round(np.asarray(nominal, float), 6).tolist()))
    return np.array([k in want for k in keys], dtype=bool)


def s4b(model, tau, val_batch, yhat_te, curve_te, labels_te):
    yhat_v, logits_v, curve_v, labels_v = scores(model, val_batch)
    rows = []
    k = 0
    for name, thr, mode in EVENT_SPECS:
        for ph in PH_WINDOWS:
            yv, yt = labels_v[:, k], labels_te[:, k]
            head_val = rates(yv, (logits_v[:, k] >= tau[k]).astype(int))
            c1 = match_cut(curve_v[:, k], yv, head_val["fpr"])
            c2 = calibrate_threshold(curve_v[:, k], yv, 0.06)
            clinical = -thr if mode == "below" else thr
            rows.append({
                "event": name, "ph_min": int(ph),
                "val_head_fpr": head_val["fpr"],
                "OP0_curve_fnr_fpr": rates(yt, (curve_te[:, k] > clinical).astype(int)),
                "OP1_cut_mgdl": float(-c1 if mode == "below" else c1),
                "OP1_curve_fnr_fpr": rates(yt, (curve_te[:, k] >= c1).astype(int)),
                "OP2_cut_mgdl": float(-c2 if mode == "below" else c2),
                "OP2_curve_fnr_fpr": rates(yt, (curve_te[:, k] >= c2).astype(int)),
            })
            k += 1
    return rows, yhat_v


def per_participant(batch, yhat, labels, logits, tau):
    rows = []
    k4 = HYPO70_4H
    for sid in np.unique(batch.subject_id):
        m = batch.subject_id == sid
        mae = mae_hours(batch.y_future[m], yhat[m], batch.horizons)
        head = rates(labels[m, k4], (logits[m, k4] >= tau[k4]).astype(int))
        rows.append({"subject_id": int(sid), "windows": int(m.sum()), "mae_0_to_h": mae, "hypo70_4h": head})
    return rows


def leave_one_out(batch, labels, logits, tau):
    y = labels[:, HYPO70_4H]
    pred = (logits[:, HYPO70_4H] >= tau[HYPO70_4H]).astype(int)
    rows = []
    for sid in np.unique(batch.subject_id):
        m = batch.subject_id != sid
        st = rates(y[m], pred[m])
        rows.append({"left_out": int(sid), "fnr": st["fnr"], "fpr": st["fpr"], "n_pos": st["n_pos"]})
    return rows


def paper_check(row):
    got = {
        "auroc_head": row["auroc_head"],
        "auroc_curve": row["auroc_curve"],
        "head_fnr": row["head"]["fnr"],
        "curve_fnr": row["curve_fnr_at_head_fpr"],
    }
    delta = {k: got[k] - PAPER[k] for k in PAPER}
    ok = all(abs(v) < 5e-4 for v in delta.values())
    return ok, got, delta


def main():
    t_start = time.time()
    runs = Path(os.environ["PF_RUNS"])
    ext = Path(os.environ["PF_EXT_CACHE"])
    c3 = Path(os.environ["PF_C3"])
    out_c4 = Path(os.environ["PF_OUT_C4"])
    out_c5 = Path(os.environ["PF_OUT_C5"])
    out_c4.mkdir(parents=True, exist_ok=True)
    out_c5.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)

    models = {}
    identity = {}
    for seed in SEEDS:
        models[seed], tau, identity[seed] = load_model(runs, seed)
        models[seed] = (models[seed], tau)
    if round(identity[20260828]["y_mean"], 2) != 149.10 or round(identity[20260828]["y_std"], 2) != 49.30:
        raise SystemExit(f"deployed normalization is not 149.10 / 49.30: {identity[20260828]}")

    with (runs / "20260830_s20260828" / "cache" / "azt1d_s20260828_str15.pkl").open("rb") as f:
        hist = pickle.load(f)
    model, tau = models[20260828]
    _yhat, logits, curve, labels = scores(model, hist["test"])
    hist_rows = output_rows(logits, curve, labels, tau, [])
    ok, got, delta = paper_check(hist_rows[HYPO70_4H])
    manifest = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "boot_seed": BOOT_SEED,
        "n_boot": N_BOOT,
        "c3_run": str(c3),
        "checkpoints": identity,
        "historical_4h_hypo70": got,
        "historical_minus_paper": delta,
        "historical_matches_paper": ok,
    }
    (out_c4 / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if not ok:
        raise SystemExit(f"historical 4 h hypo70 does not match the paper: {delta}")

    sets = []
    # AZT1D tests from the causal caches, external tests, deployed S4b on historical val.
    def one(name, seed, batch, nominal, historical):
        model, tau = models[seed]
        yhat, logits, curve, labels = scores(model, batch)
        boots = bootstrap_indices(batch.subject_id)
        rows = output_rows(logits, curve, labels, tau, boots)
        entry = {
            "set": name,
            "seed": seed,
            "windows": int(len(batch)),
            "participants": int(len(np.unique(batch.subject_id))),
            "mae_h": mae_hours(batch.y_future, yhat, batch.horizons),
            "fpr_above_0.06": fpr_over_budget(rows),
            "outputs": rows,
        }
        if historical is not None and nominal is not None:
            mask = key_mask(batch, nominal, historical)
            entry["shared_nominal_windows"] = int(mask.sum())
            if mask.any():
                sub, nom_s = subset(batch, nominal, mask)
                yhat_s, logits_s, curve_s, labels_s = yhat[mask], logits[mask], curve[mask], labels[mask]
                paired = output_rows(logits_s, curve_s, labels_s, tau, [])
                entry["paired_on_shared_nominal_origin"] = {
                    "windows": int(mask.sum()),
                    "mae_h": mae_hours(sub.y_future, yhat_s, sub.horizons),
                    "hypo70_4h": paired[HYPO70_4H],
                }
                # Historical t0 is the nominal origin. Join on that, not on the shifted t0'.
                hmask = key_mask(historical, historical.t0, _nominal_as_batch(sub, nom_s))
                hist_sub, _hn = subset(historical, historical.t0, hmask)
                hy, hl, hc, hlab = scores(model, hist_sub)
                hh = output_rows(hl, hc, hlab, tau, [])
                entry["historical_on_shared_nominal_origin"] = {
                    "windows": int(hmask.sum()),
                    "mae_h": mae_hours(hist_sub.y_future, hy, hist_sub.horizons),
                    "hypo70_4h_point": {
                        "auroc_head": hh[HYPO70_4H].get("auroc_head"),
                        "auroc_curve": hh[HYPO70_4H].get("auroc_curve"),
                        "head_fnr": (hh[HYPO70_4H].get("head") or {}).get("fnr"),
                        "curve_fnr_at_head_fpr": hh[HYPO70_4H].get("curve_fnr_at_head_fpr"),
                    },
                }
        if seed == 20260828 and name.startswith("azt1d"):
            entry["per_participant"] = per_participant(batch, yhat, labels, logits, tau)
            entry["leave_one_out_hypo70_4h"] = leave_one_out(batch, labels, logits, tau)
            entry["s4b_cuts_from_submitted_validation"] = s4b(model, tau, hist["val"], yhat, curve, labels)[0]
            if name.endswith("test"):
                for ev, _thr, _mode in EVENT_SPECS:
                    # column of this event at 240 min
                    pass
                save_roc(out_c5, logits, curve, labels)
        sets.append(entry)
        print(name, "windows", len(batch), "4h hypo70", rows[HYPO70_4H].get("auroc_head"), flush=True)
        (out_c4 / "route_a_metrics.json").write_text(json.dumps({"sets": sets}, indent=1))

    def save_roc(out_dir, logits, curve, labels):
        k = 0
        for name, _thr, _mode in EVENT_SPECS:
            for ph in PH_WINDOWS:
                if ph == 240:
                    y = labels[:, k]
                    fpr_h, tpr_h, _ = roc_curve(y, logits[:, k])
                    fpr_c, tpr_c, _ = roc_curve(y, curve[:, k])
                    prec_h, rec_h, _ = precision_recall_curve(y, logits[:, k])
                    prec_c, rec_c, _ = precision_recall_curve(y, curve[:, k])
                    np.savez_compressed(
                        out_dir / f"roc_pr_{name}_240.npz",
                        fpr_head=fpr_h, tpr_head=tpr_h, fpr_curve=fpr_c, tpr_curve=tpr_c,
                        precision_head=prec_h, recall_head=rec_h, precision_curve=prec_c, recall_curve=rec_c,
                    )
                k += 1

    for seed in SEEDS:
        with (c3 / f"azt1d_s{seed}.pkl").open("rb") as f:
            blob = pickle.load(f)
        with (runs / f"20260830_s{seed}" / "cache" / f"azt1d_s{seed}_str15.pkl").open("rb") as f:
            cache = pickle.load(f)
        one(f"azt1d_s{seed}_test", seed, blob["batch"]["test"], blob["nominal_t0"]["test"], cache["test"])

    with (c3 / "ohio.pkl").open("rb") as f:
        ohio = pickle.load(f)
    with (ext / "ohio_str15.pkl").open("rb") as f:
        ohio_hist = pickle.load(f)["test"]
    one("ohiot1dm", 20260828, ohio["batch"]["test"], ohio["nominal_t0"]["test"], ohio_hist)

    with (c3 / "diatrend.pkl").open("rb") as f:
        dia = pickle.load(f)
    with (ext / "diatrend17_str15.pkl").open("rb") as f:
        dia_hist = pickle.load(f)["test"]
    one("diatrend", 20260828, dia["batch"]["test"], dia["nominal_t0"]["test"], dia_hist)

    payload = {"sets": sets}
    (out_c4 / "route_a_metrics.json").write_text(json.dumps(payload, indent=1))
    # C5 keeps the paired endpoint block, including bootstrap intervals.
    c5 = []
    for entry in sets:
        c5.append({
            "set": entry["set"],
            "windows": entry["windows"],
            "participants": entry["participants"],
            "outputs": [{k: row.get(k) for k in ("event", "ph_min", "n_pos", "n", "auroc_head", "auroc_curve", "curve_fnr_at_head_fpr", "head", "boot")} for row in entry["outputs"]],
            "shared_nominal_windows": entry.get("shared_nominal_windows"),
            "paired_on_shared_nominal_origin": entry.get("paired_on_shared_nominal_origin"),
            "historical_on_shared_nominal_origin": entry.get("historical_on_shared_nominal_origin"),
        })
    (out_c5 / "event_vs_curve.json").write_text(json.dumps({"n_boot": N_BOOT, "boot_seed": BOOT_SEED, "sets": c5}, indent=1))
    manifest["elapsed_s"] = round(time.time() - t_start, 1)
    (out_c4 / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out_c5 / "manifest.json").write_text(json.dumps({
        "c4_manifest": "see ../c4 manifest",
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "elapsed_s": manifest["elapsed_s"],
        "historical_matches_paper": ok,
    }, indent=1))
    print("wrote", out_c4, out_c5, flush=True)


if __name__ == "__main__":
    main()
