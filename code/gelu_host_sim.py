#!/usr/bin/env python3
"""Host simulation: effect of exact GELU (erf) vs tanh-approximated GELU on event decisions on the AZT1D held-out set.

Compared variants:
- ref_mps32: host reference of original-submission Phase B (MPS, float32, exact GELU);
- ref_cpu64: CPU float64, exact GELU, used as the numerical ground truth;
- tanh_cpu32: CPU float32, tanh-approximated GELU, simulating the original-submission firmware;
- erf_cpu32: CPU float32, exact GELU, simulating the revised firmware.
Board archives: rice-s3/eval/validation/phase_b_ckpt.npz (event decisions for 19,152 windows),
sample512.npz (logits for 512 windows).
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path(os.environ.get("PF_WORKSPACE", "."))
SRC = BASE / "rice-former/src"
RUN = BASE / "runs/20260830_s20260828"
CKPT = RUN / "ckpt/ceqt.pt"
CACHE = RUN / "cache/azt1d_s20260828_str15.pkl"
BOARD = BASE / "validation"
OUT = Path(__file__).resolve().parent / "results" / "gelu"

sys.path.insert(0, str(SRC))
from ceqt.models.ceqt import CEQT, CEQTNet  # noqa: E402
from ceqt.ops.ink import InertiaKernel  # noqa: E402


def set_gelu(net: nn.Module, approximate: str) -> nn.Module:
    for name, mod in net.named_children():
        if isinstance(mod, nn.GELU):
            setattr(net, name, nn.GELU(approximate=approximate))
        else:
            set_gelu(mod, approximate)
    return net


def build(blob, approximate: str, dtype: torch.dtype) -> CEQT:
    net = CEQTNet()
    net.load_state_dict(blob["state"])
    net = set_gelu(net, approximate).to(dtype).eval()
    return CEQT(net=net, ink=InertiaKernel(), y_mean=float(blob["y_mean"]), y_std=float(blob["y_std"]),
                thresholds=np.asarray(blob["thresholds"], dtype=np.float64))


def predict(model: CEQT, test, device: str, dtype: torch.dtype) -> np.ndarray:
    model.net.to(device)
    if dtype == torch.float64:
        orig = model._tensors

        def tensors64(batch, dev):
            return tuple(t.to(torch.float64) for t in orig(batch, dev))

        model._tensors = tensors64
    return model.predict_batch(test, device=torch.device(device))["logits"].astype(np.float64)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(os.cpu_count() or 1, 1))
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    thr = np.asarray(blob["thresholds"], dtype=np.float64)
    with CACHE.open("rb") as f:
        test = pickle.load(f)["test"]
    n = len(test)

    runs = {}
    if torch.backends.mps.is_available():
        runs["ref_mps32"] = predict(build(blob, "none", torch.float32), test, "mps", torch.float32)
    runs["ref_cpu64"] = predict(build(blob, "none", torch.float64), test, "cpu", torch.float64)
    runs["tanh_cpu32"] = predict(build(blob, "tanh", torch.float32), test, "cpu", torch.float32)
    runs["erf_cpu32"] = predict(build(blob, "none", torch.float32), test, "cpu", torch.float32)
    runs["tanh_cpu64"] = predict(build(blob, "tanh", torch.float64), test, "cpu", torch.float64)

    evt = {k: (v >= thr[None, :]).astype(np.int32) for k, v in runs.items()}
    board = np.load(BOARD / "phase_b_ckpt.npz")["evt_board"]
    assert board.shape == (n, 16)
    evt["board_fw_tanh"] = board

    def cmp(a, b):
        d = evt[a] != evt[b]
        return {"cells": int(d.sum()), "windows": int(d.any(axis=1).sum()),
                "window_agree": int(n - d.any(axis=1).sum())}

    names = list(evt)
    pairs = {f"{a}|{b}": cmp(a, b) for i, a in enumerate(names) for b in names[i + 1:]}

    logit_diff = {}
    for a in runs:
        for b in runs:
            if a < b:
                logit_diff[f"{a}|{b}"] = float(np.max(np.abs(runs[a] - runs[b])))

    s = np.load(BOARD / "sample512.npz")
    idx = s["idx"]
    lb = s["logits_board"].astype(np.float64)
    sample = {k: {"max": float(np.max(np.abs(lb - v[idx]))), "median_of_window_max":
                  float(np.median(np.max(np.abs(lb - v[idx]), axis=1)))} for k, v in runs.items()}

    flips = []
    d = np.argwhere(board != evt["ref_mps32" if "ref_mps32" in evt else "erf_cpu32"])
    for i, c in d:
        flips.append({"idx": int(i), "cell": int(c), "board": int(board[i, c]),
                      **{k: int(evt[k][i, c]) for k in runs},
                      **{f"margin_{k}": float(runs[k][i, c] - thr[c]) for k in runs}})

    rep = {"n_windows": n, "thresholds_from": str(CKPT), "decision_pairs": pairs,
           "logit_maxabs_pairs": logit_diff, "board_vs_host_logits_sample512": sample, "flips_vs_ref": flips}
    (OUT / "gelu_host_sim.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUT / "gelu_host_logits.npz", thresholds=thr, **runs)
    print(json.dumps({"pairs": pairs, "sample512": sample}, indent=1))
    for f in flips:
        print(f)


if __name__ == "__main__":
    main()
