"""Export CEQT weights + one AZT1D holdout window for S3 firmware. Requires the public RICE-Former source tree and an authorized checkpoint."""

from __future__ import annotations

import json
import pickle
import sys
import os
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RELEASE_SRC = Path(os.environ.get("RICE_FORMER_SRC", "")).expanduser()
if not str(RELEASE_SRC):
    raise SystemExit("Set RICE_FORMER_SRC to the rice-former src directory (the folder that contains ceqt/).")
CKPT = Path(os.environ.get("RICE_CKPT", "")).expanduser()
CACHE = Path(os.environ.get("RICE_CACHE", "")).expanduser()
if not CKPT.is_file() or not CACHE.is_file():
    raise SystemExit("Set RICE_CKPT and RICE_CACHE to the authorized checkpoint and AZT1D cache.")
FW = Path(os.environ.get("PICO_FIRMWARE_INCLUDE", ROOT / "firmware" / "include")).expanduser()
GOLD_DIR = Path(os.environ.get("PICO_GOLD_DIR", ROOT / "gold")).expanduser()

sys.path.insert(0, str(RELEASE_SRC))

from ceqt.data.windows import WindowBatch  # noqa: E402
from ceqt.metrics import event_labels  # noqa: E402
from ceqt.models.ceqt import CEQT, CEQTNet, pack_marks  # noqa: E402
from ceqt.ops.ink import InertiaKernel  # noqa: E402


def c_float(x: float) -> str:
    s = f"{float(x):.8g}"
    if "e" not in s and "E" not in s and "." not in s:
        s += ".0"
    return s + "f"


def c_array(name: str, arr: np.ndarray) -> str:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    body = ",\n".join(
        "  " + ", ".join(c_float(x) for x in flat[i : i + 8]) for i in range(0, flat.size, 8)
    )
    return f"static const float {name}[{flat.size}] = {{\n{body}\n}};\n"


def main() -> None:
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = CEQTNet()
    net.load_state_dict(blob["state"])
    net.eval()
    y_mean = float(blob["y_mean"])
    y_std = float(blob["y_std"])
    thresholds = np.asarray(blob["thresholds"], dtype=np.float32)
    model = CEQT(
        net=net,
        ink=InertiaKernel(),
        y_mean=y_mean,
        y_std=y_std,
        use_g=True,
        use_quality=True,
        use_type=True,
        thresholds=thresholds.astype(np.float64),
    )

    with CACHE.open("rb") as f:
        data = pickle.load(f)
    test = data["test"]
    idx = 0
    for i in range(len(test)):
        if int(test.n_marks[i]) >= 1:
            idx = i
            break
    sl = slice(idx, idx + 1)
    one = WindowBatch(
        subject_id=test.subject_id[sl],
        t0=test.t0[sl],
        y_hist=test.y_hist[sl],
        t_hist=test.t_hist[sl],
        y_future=test.y_future[sl],
        horizons=test.horizons,
        mark_t=test.mark_t[idx : idx + 1],
        mark_a=test.mark_a[idx : idx + 1],
        mark_e=test.mark_e[idx : idx + 1],
        day_index=test.day_index[sl],
        n_marks=test.n_marks[sl],
    )
    out = model.predict_batch(one, device=torch.device("cpu"))
    marks, mask = pack_marks(one)
    y_norm = (one.y_hist.astype(np.float32) - y_mean) / y_std

    gold = {
        "ckpt": str(CKPT),
        "subject_id": int(one.subject_id[0]),
        "t0": float(one.t0[0]),
        "n_marks": int(one.n_marks[0]),
        "y_mean": y_mean,
        "y_std": y_std,
        "thresholds": thresholds.tolist(),
        "y_hist": one.y_hist[0].astype(float).tolist(),
        "t_hist": one.t_hist[0].astype(float).tolist(),
        "marks": marks[0].astype(float).tolist(),
        "mask": mask[0].astype(float).tolist(),
        "y_norm": y_norm[0].astype(float).tolist(),
        "delta": out["delta"][0].astype(float).tolist(),
        "logits": out["logits"][0].astype(float).tolist(),
        "y_iner": out["y_iner"][0].astype(float).tolist(),
        "g": float(out["g"][0]),
        "yhat": out["yhat"][0].astype(float).tolist(),
        "y_future": one.y_future[0].astype(float).tolist(),
        "horizons": one.horizons.astype(float).tolist(),
    }
    gold_path = GOLD_DIR / "window0.json"
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_text(json.dumps(gold, indent=2), encoding="utf-8")

    n = min(32, len(test))
    sub = WindowBatch(
        subject_id=test.subject_id[:n],
        t0=test.t0[:n],
        y_hist=test.y_hist[:n],
        t_hist=test.t_hist[:n],
        y_future=test.y_future[:n],
        horizons=test.horizons,
        mark_t=test.mark_t[:n],
        mark_a=test.mark_a[:n],
        mark_e=test.mark_e[:n],
        day_index=test.day_index[:n],
        n_marks=test.n_marks[:n],
    )
    batch_out = model.predict_batch(sub, device=torch.device("cpu"))
    mae_1h = float(np.mean(np.abs(batch_out["yhat"][:, 11] - sub.y_future[:, 11])))
    mae_4h = float(np.mean(np.abs(batch_out["yhat"][:, 47] - sub.y_future[:, 47])))
    pred_evt = (batch_out["logits"] >= thresholds[None, :]).astype(np.float32)
    lab = event_labels(sub.y_future, sub.horizons)
    tp = ((pred_evt[:, 3] == 1) & (lab[:, 3] == 1)).sum()
    fn = ((pred_evt[:, 3] == 0) & (lab[:, 3] == 1)).sum()
    fp = ((pred_evt[:, 3] == 1) & (lab[:, 3] == 0)).sum()
    tn = ((pred_evt[:, 3] == 0) & (lab[:, 3] == 0)).sum()
    fnr = float(fn / (tp + fn)) if (tp + fn) else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    metrics = {
        "n": n,
        "mae_1h": mae_1h,
        "mae_4h": mae_4h,
        "hypo70_4h_fnr": fnr,
        "hypo70_4h_fpr": fpr,
        "note": "AZT1D test-split first 32 windows; not the full-paper table",
    }
    (GOLD_DIR / "pc_metrics_32.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    FW.mkdir(parents=True, exist_ok=True)
    parts = [
        "#pragma once\n",
        "// generated by host/export_golden.py; do not edit\n",
        f"#define CEQT_Y_MEAN {c_float(y_mean)}\n",
        f"#define CEQT_Y_STD {c_float(y_std)}\n",
        f"#define CEQT_G {c_float(float(out['g'][0]))}\n",
        f"#define CEQT_SUBJECT {int(one.subject_id[0])}\n",
        f"#define CEQT_N_MARKS {int(one.n_marks[0])}\n",
    ]
    names = [
        ("W_POS", net.pos.detach().cpu().numpy()),
        ("W_CURVE_Q", net.curve_q.detach().cpu().numpy()),
        ("W_EVENT_Q", net.event_q.detach().cpu().numpy()),
        ("W_CGM", net.cgm_proj.weight.detach().cpu().numpy()),
        ("B_CGM", net.cgm_proj.bias.detach().cpu().numpy()),
        ("W_TYPE", net.type_emb.weight.detach().cpu().numpy()),
        ("W_MARK1", net.mark_mlp[0].weight.detach().cpu().numpy()),
        ("B_MARK1", net.mark_mlp[0].bias.detach().cpu().numpy()),
        ("W_MARK2", net.mark_mlp[2].weight.detach().cpu().numpy()),
        ("B_MARK2", net.mark_mlp[2].bias.detach().cpu().numpy()),
        ("W_DELTA1", net.delta_head[0].weight.detach().cpu().numpy()),
        ("B_DELTA1", net.delta_head[0].bias.detach().cpu().numpy()),
        ("W_DELTA2", net.delta_head[2].weight.detach().cpu().numpy()),
        ("B_DELTA2", net.delta_head[2].bias.detach().cpu().numpy()),
        ("W_EVT1", net.evt_head[0].weight.detach().cpu().numpy()),
        ("B_EVT1", net.evt_head[0].bias.detach().cpu().numpy()),
        ("W_EVT2", net.evt_head[2].weight.detach().cpu().numpy()),
        ("B_EVT2", net.evt_head[2].bias.detach().cpu().numpy()),
        ("W_CROSS_IN", net.cross.in_proj_weight.detach().cpu().numpy()),
        ("B_CROSS_IN", net.cross.in_proj_bias.detach().cpu().numpy()),
        ("W_CROSS_OUT", net.cross.out_proj.weight.detach().cpu().numpy()),
        ("B_CROSS_OUT", net.cross.out_proj.bias.detach().cpu().numpy()),
        ("W_THR", thresholds),
        ("G_Y_HIST", one.y_hist[0]),
        ("G_Y_NORM", y_norm[0]),
        ("G_MARKS", marks[0]),
        ("G_MASK", mask[0]),
        ("G_DELTA", out["delta"][0]),
        ("G_LOGITS", out["logits"][0]),
        ("G_YHAT", out["yhat"][0]),
        ("G_YINER", out["y_iner"][0]),
        ("G_YFUT", one.y_future[0]),
    ]
    for li in range(2):
        layer = net.enc.layers[li]
        names += [
            (f"W_E{li}_IN", layer.self_attn.in_proj_weight.detach().cpu().numpy()),
            (f"B_E{li}_IN", layer.self_attn.in_proj_bias.detach().cpu().numpy()),
            (f"W_E{li}_OUT", layer.self_attn.out_proj.weight.detach().cpu().numpy()),
            (f"B_E{li}_OUT", layer.self_attn.out_proj.bias.detach().cpu().numpy()),
            (f"W_E{li}_L1", layer.linear1.weight.detach().cpu().numpy()),
            (f"B_E{li}_L1", layer.linear1.bias.detach().cpu().numpy()),
            (f"W_E{li}_L2", layer.linear2.weight.detach().cpu().numpy()),
            (f"B_E{li}_L2", layer.linear2.bias.detach().cpu().numpy()),
            (f"W_E{li}_N1", layer.norm1.weight.detach().cpu().numpy()),
            (f"B_E{li}_N1", layer.norm1.bias.detach().cpu().numpy()),
            (f"W_E{li}_N2", layer.norm2.weight.detach().cpu().numpy()),
            (f"B_E{li}_N2", layer.norm2.bias.detach().cpu().numpy()),
        ]
    (FW / "ceqt_weights.h").write_text("".join(parts + [c_array(n, a) for n, a in names]), encoding="utf-8")
    print("wrote", gold_path)
    print("wrote", FW / "ceqt_weights.h")
    print("metrics32", metrics)
    print("g", gold["g"], "yhat1h", gold["yhat"][11], "yhat4h", gold["yhat"][47])


if __name__ == "__main__":
    main()
