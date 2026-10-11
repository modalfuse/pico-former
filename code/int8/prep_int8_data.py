#!/usr/bin/env python3
"""R2-14 step 1 (streamlit3.13 environment): prepare tokens and the FP32 reference of the same checkpoint for INT8 quantization.

Token construction (CGM normalization, positional encoding, marker encoding) is done in FP32 by C code at deployment; the quantized graph takes the 96×48 token matrix,
so tokens are computed here in float32 and the FP32 reference outputs in float64 (same as pf2.host_forward).
Calibration set: 1,024 windows drawn from the training split, stratified by participant (seed 20261009); the validation and held-out splits are excluded.
Output in code/results/int8/: state_s20260828.pt (plain-tensor weights with y_mean, y_std and frozen thresholds),
  tok_<split>.npy (float32, N×96×48), side_<split>.npz (labels, inertial baseline, residual gate, FP32 reference logits and delta).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "host2"))
import pf2  # noqa: E402

RUN = pf2.BASE / "runs/20260830_s20260828"
OUT = HERE.parent / "results" / "int8"
SEED, N_CALIB = 20261009, 1024


@torch.no_grad()
def tokens(net, y_mean, y_std, fr):
    y = torch.from_numpy(((fr["y"].astype(np.float32) - np.float32(y_mean)) / np.float32(y_std)).astype(np.float32))
    marks = torch.from_numpy(fr["feat"])
    mask = torch.from_numpy(fr["mask"])
    h = net.cgm_proj(y.unsqueeze(-1)) + net.pos.unsqueeze(0)
    typ = marks[..., 0].long().clamp(0, 3)
    mtok = (net.type_emb(typ) + net.mark_mlp(marks[..., 1:])) * mask.unsqueeze(-1)
    return torch.cat([h, mtok], dim=1).numpy().astype(np.float32)


def calib_idx(batch, rng):
    sid = np.asarray(batch.subject_id)
    subs = np.unique(sid)
    per = N_CALIB // len(subs)
    out = [rng.choice(np.flatnonzero(sid == s), per, replace=False) for s in subs]
    rest = np.setdiff1d(np.arange(len(sid)), np.concatenate(out))
    out.append(rng.choice(rest, N_CALIB - per * len(subs), replace=False))
    return np.sort(np.concatenate(out))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    net, y_mean, y_std, thr = pf2.load_model(RUN / "ckpt/ceqt.pt")
    torch.save({"state": {k: v.clone() for k, v in net.state_dict().items()}, "y_mean": y_mean, "y_std": y_std,
                "thresholds": thr}, OUT / "state_s20260828.pt")
    cache = pf2.load_cache(RUN / "cache/azt1d_s20260828_str15.pkl")
    rng = np.random.RandomState(SEED)
    for split, src in (("calib", "train"), ("val", "val"), ("test", "test")):
        batch = cache[src]
        fr = pf2.frames(batch)
        idx = calib_idx(batch, rng) if split == "calib" else np.arange(len(batch))
        sub = {k: v[idx] for k, v in fr.items()}
        tok = tokens(net, y_mean, y_std, sub)
        np.save(OUT / f"tok_{split}.npy", tok)
        side = {"idx": idx, "subject": np.asarray(batch.subject_id)[idx], "n_marks": sub["n_marks"], "q": sub["q"],
                "iner": sub["iner"]}
        if split != "calib":
            l, yh, g = pf2.host_forward(net, y_mean, y_std, sub["y"], sub["feat"], sub["mask"], sub["n_marks"],
                                        sub["q"], sub["iner"])
            side.update(logits_fp32=l, yhat_fp32=yh, g=g, labels=pf2.labels(batch)[idx],
                        y_future=np.asarray(batch.y_future, np.float64)[idx], horizons=np.asarray(batch.horizons))
        np.savez_compressed(OUT / f"side_{split}.npz", **side)
        print(split, tok.shape, flush=True)


if __name__ == "__main__":
    main()
