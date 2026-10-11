#!/usr/bin/env python3
"""R2-15 design space: host training and evaluation of 15 architecture configurations x 3 initialisation seeds.

The training protocol matches RICE-Former stage P3 (fit_ceqt: AdamW 1e-3, batch 128, 10 epochs, best epoch by
validation loss; event thresholds chosen on the validation set as the highest recall with FPR <= 0.06). The data
split is fixed to the cache built with seed 20260828, so all configurations are compared on the same held-out set;
the seed changes only initialisation and batch order.
Token-count configurations only truncate the model input: the most recent 36 (3 h) or 72 (6 h) CGM points and the
most recent 12 or 24 marks; inputs to the inertial kernel and the residual gate are unchanged.

Usage: python train_grid.py [--configs base,d32_l1,...] [--seeds 20260828,...] [--out <dir>]
Output: code/results/design_space/<config>_s<seed>.json (written to the --out directory if given, used for
repeated training with the same seed)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(os.environ.get("PF_WORKSPACE", "."))
sys.path.insert(0, str(BASE / "rice-former/src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from ceqt.eval import calibrate_event_thresholds  # noqa: E402
from ceqt.metrics import event_labels  # noqa: E402
from ceqt.models.ceqt import MAX_MARKS, N_EVENT, CEQTNet, pack_marks  # noqa: E402
from ceqt.ops.gate import residual_gate  # noqa: E402
from ceqt.ops.ink import InertiaKernel, ink_stack  # noqa: E402

CACHE = BASE / "runs/20260830_s20260828/cache/azt1d_s20260828_str15.pkl"
OUT = Path(__file__).resolve().parents[1] / "results" / "design_space"

# name: (d, layers, heads, n_cgm, max_marks)
CONFIGS = {
    "d32_l1": (32, 1, 4, 72, 24), "d32_l2": (32, 2, 4, 72, 24), "d32_l3": (32, 3, 4, 72, 24),
    "d48_l1": (48, 1, 4, 72, 24), "base": (48, 2, 4, 72, 24), "d48_l3": (48, 3, 4, 72, 24),
    "d64_l1": (64, 1, 4, 72, 24), "d64_l2": (64, 2, 4, 72, 24), "d64_l3": (64, 3, 4, 72, 24),
    "d96_l2": (96, 2, 4, 72, 24),
    "L48": (48, 2, 4, 36, 12), "L60": (48, 2, 4, 36, 24), "L84": (48, 2, 4, 72, 12),
    "h2": (48, 2, 2, 72, 24), "h8": (48, 2, 8, 72, 24),
}


class Net(CEQTNet):
    def __init__(self, d, layers, heads, n_cgm, max_marks):
        super().__init__(n_cgm=n_cgm, d=d, n_heads=heads, max_marks=max_marks)
        if layers == 2:
            return
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, dim_feedforward=d * 2, batch_first=True, dropout=0.1),
            num_layers=layers,
        )


def act_bytes(d: int, L: int) -> int:
    return 4 * (8 * L * d + L * L + 64 * d)


def pack(batch, y_mean, y_std, n_cgm, max_marks):
    y = (batch.y_hist[:, -n_cgm:].astype(np.float32) - y_mean) / y_std
    marks, mask = pack_marks(batch, max_marks=max_marks)
    iner = ink_stack(InertiaKernel(), batch.t_hist, batch.y_hist, batch.t0, batch.horizons).astype(np.float32)
    tgt = (batch.y_future.astype(np.float32) - iner) / y_std
    lab = event_labels(batch.y_future, batch.horizons)
    g = (batch.n_marks > 0).astype(np.float32)
    return y, marks, mask, tgt, lab, g, iner


def predict(net, dev, y, marks, mask, chunk=2048):
    net.eval()
    ds, ls = [], []
    with torch.no_grad():
        for s in range(0, len(y), chunk):
            d, lg = net(*(torch.from_numpy(a[s:s + chunk]).to(dev) for a in (y, marks, mask)))
            ds.append(d.cpu().numpy())
            ls.append(lg.cpu().numpy())
    return np.concatenate(ds), np.concatenate(ls)


def fnr_fpr(lab, pred):
    tp = ((pred == 1) & (lab == 1)).sum(); fn = ((pred == 0) & (lab == 1)).sum()
    fp = ((pred == 1) & (lab == 0)).sum(); tn = ((pred == 0) & (lab == 0)).sum()
    return float(fn / max(tp + fn, 1)), float(fp / max(fp + tn, 1))


def run(name, seed, data, dev, epochs=10, out=OUT):
    d, layers, heads, n_cgm, max_marks = CONFIGS[name]
    train, val, test = data["train"], data["val"], data["test"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    y_mean = float(np.nanmean(train.y_hist))
    y_std = float(max(np.nanstd(train.y_hist), 8.0))
    y, mk, ms, tg, lb, g, _ = pack(train, y_mean, y_std, n_cgm, max_marks)
    yv, mv, kv, tv, lv, gv, _ = pack(val, y_mean, y_std, n_cgm, max_marks)
    net = Net(d, layers, heads, n_cgm, max_marks).to(dev)
    n_param = int(sum(p.numel() for p in net.parameters()))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(*(torch.from_numpy(a) for a in (y, mk, ms, tg, lb, g))), batch_size=128,
                        shuffle=True)
    w = np.ones(tg.shape[1], dtype=np.float32); w[:12] = 2.0; w[12:24] = 1.2
    wt = torch.from_numpy(w).to(dev)
    best, best_state, best_ep = 1e9, None, -1
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        for yb, mb, kb, tb, lbb, gb in loader:
            yb, mb, kb, tb, lbb, gb = (x.to(dev) for x in (yb, mb, kb, tb, lbb, gb))
            opt.zero_grad()
            delta, logits = net(yb, mb, kb)
            delta = delta * gb.unsqueeze(1)
            loss = (F.huber_loss(delta, tb, reduction="none", delta=1.0) * wt).mean() + 0.6 * \
                F.binary_cross_entropy_with_logits(logits, lbb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        dv, lgv = predict(net, dev, yv, mv, kv)
        dv = dv * gv[:, None]
        vl = float(F.huber_loss(torch.from_numpy(dv), torch.from_numpy(tv)).item()
                   + 0.6 * F.binary_cross_entropy_with_logits(torch.from_numpy(lgv), torch.from_numpy(lv)).item())
        if vl < best:
            best, best_ep = vl, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"  {name} s{seed} epoch {ep + 1}/{epochs} val={vl:.4f} best={best:.4f} t={time.time() - t0:.0f}s",
              flush=True)
    train_s = time.time() - t0
    net.load_state_dict(best_state)

    def evaluate(batch):
        yy, mm, kk, _, ll, _, iner = pack(batch, y_mean, y_std, n_cgm, max_marks)
        dd, lg = predict(net, dev, yy, mm, kk)
        gate = residual_gate(batch.n_marks, batch.mark_a, batch.mark_e)
        yhat = iner + gate[:, None] * dd * y_std
        return yhat, lg, ll

    _, lgv, llv = evaluate(val)
    thr = calibrate_event_thresholds(lgv, val, fpr_budget=0.06)
    yhat, lgt, llt = evaluate(test)
    err = yhat - test.y_future
    res = {"config": name, "seed": seed, "d": d, "layers": layers, "heads": heads, "n_cgm": n_cgm,
           "max_marks": max_marks, "L": n_cgm + max_marks, "act_bytes": act_bytes(d, n_cgm + max_marks),
           "params": n_param, "weight_bytes_fp32": 4 * n_param, "epochs": epochs, "best_epoch": best_ep,
           "best_val": best, "train_s": train_s, "device": str(dev), "thresholds": thr.tolist()}
    for h, i in (("1h", 11), ("2h", 23), ("4h", 47)):
        e = err[:, i]
        e = e[np.isfinite(e)]
        res[f"mae_{h}"] = float(np.mean(np.abs(e)))
        res[f"rmse_{h}"] = float(np.sqrt(np.mean(e ** 2)))
    pred = (lgt >= thr[None, :]).astype(int)
    k = 0
    from ceqt import EVENT_SPECS, PH_WINDOWS
    for ev, _thr, _mode in EVENT_SPECS:
        for ph in PH_WINDOWS:
            fnr, fpr = fnr_fpr(llt[:, k].astype(int), pred[:, k])
            res[f"{ev}_{ph // 60}h_fnr"], res[f"{ev}_{ph // 60}h_fpr"] = fnr, fpr
            k += 1
    assert k == N_EVENT
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_s{seed}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    torch.save({"state": best_state, "y_mean": y_mean, "y_std": y_std, "cfg": CONFIGS[name], "thresholds": thr.tolist()},
               out / f"{name}_s{seed}.pt")
    print(f"DONE {name} s{seed} params={n_param} train={train_s:.0f}s mae1h={res['mae_1h']:.2f} "
          f"mae4h={res['mae_4h']:.2f} hypo70_4h_fnr={res['hypo70_4h_fnr']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--seeds", default="20260828,20260829,20260830")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    with CACHE.open("rb") as f:
        data = pickle.load(f)
    dev = torch.device(args.device)
    out = Path(args.out)
    for seed in map(int, args.seeds.split(",")):
        for name in args.configs.split(","):
            if (out / f"{name}_s{seed}.json").exists():
                print(f"skip {name} s{seed}", flush=True)
                continue
            run(name, seed, data, dev, out=out)


if __name__ == "__main__":
    main()
