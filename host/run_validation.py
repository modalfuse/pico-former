#!/usr/bin/env python3
"""Pico-Former on-board validation: sample, full holdout, burn, Wi-Fi jitter."""

from __future__ import annotations

import argparse
import json
import pickle
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import serial
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
OUT = Path(os.environ.get("PICO_OUT", ROOT / "results" / "validation")).expanduser()
PORT_DEFAULT = os.environ.get("PICO_PORT", "/dev/cu.usbserial-0001")
BAUD = 115200

sys.path.insert(0, str(RELEASE_SRC))

from ceqt.data.windows import WindowBatch  # noqa: E402
from ceqt.metrics import event_labels  # noqa: E402
from ceqt.models.ceqt import CEQT, CEQTNet, pack_marks  # noqa: E402
from ceqt.ops.gate import residual_gate  # noqa: E402
from ceqt.ops.ink import InertiaKernel  # noqa: E402

MAGIC_WIN = b"PFW1"
MAGIC_RES = b"PFR1"
MAGIC_BURN = b"PFB1"
MAGIC_PING = b"PFP1"
N_CGM, N_MARK, N_OUT, N_EVT = 72, 24, 48, 16
WIN_PAYLOAD = 2 + 1 + 1 + N_CGM * 4 + N_MARK * 3 * 4 + N_MARK * 4
RES_BODY = 4 + 1 + 1 + N_EVT + 4 + 16 + N_OUT * 4 + N_EVT * 4 + 4  # after magic
RES_FRAME = 4 + RES_BODY + 1


def xor_bytes(data: bytes) -> int:
    x = 0
    for b in data:
        x ^= b
    return x


def pack_window(seq: int, n_marks: int, quality_ok: int, y_hist, marks, mask) -> bytes:
    payload = struct.pack("<HBB", int(n_marks), int(quality_ok), 0)
    payload += np.asarray(y_hist, dtype="<f4").reshape(-1).tobytes()
    payload += np.asarray(marks, dtype="<f4").reshape(-1).tobytes()
    payload += np.asarray(mask, dtype="<f4").reshape(-1).tobytes()
    if len(payload) != WIN_PAYLOAD:
        raise RuntimeError(f"payload {len(payload)} != {WIN_PAYLOAD}")
    body = struct.pack("<I", int(seq)) + payload
    return MAGIC_WIN + body + bytes([xor_bytes(body)])


def parse_result(frame: bytes) -> dict:
    if len(frame) != RES_FRAME or frame[:4] != MAGIC_RES:
        raise ValueError("bad result frame")
    body = frame[4:-1]
    if xor_bytes(body) != frame[-1]:
        raise ValueError("result xor mismatch")
    seq, status, core = struct.unpack_from("<IBB", body, 0)
    evt = list(body[6 : 6 + N_EVT])
    off = 6 + N_EVT
    (g,) = struct.unpack_from("<f", body, off)
    off += 4
    us_total, us_enc, us_cross, us_heads = struct.unpack_from("<IIII", body, off)
    off += 16
    yhat = np.frombuffer(body[off : off + N_OUT * 4], dtype="<f4").copy()
    off += N_OUT * 4
    logits = np.frombuffer(body[off : off + N_EVT * 4], dtype="<f4").copy()
    off += N_EVT * 4
    (heap,) = struct.unpack_from("<I", body, off)
    return {
        "seq": seq,
        "status": status,
        "core": core,
        "evt": np.asarray(evt, dtype=np.int32),
        "g": float(g),
        "us_total": us_total,
        "us_enc": us_enc,
        "us_cross": us_cross,
        "us_heads": us_heads,
        "yhat": yhat,
        "logits": logits,
        "heap_free": heap,
    }


class Board:
    def __init__(self, port: str, reset: bool = False):
        self.ser = serial.Serial(port, BAUD, timeout=0.2, dsrdtr=False, rtscts=False)
        self.ser.dtr = False
        self.ser.rts = False
        time.sleep(0.05)
        if reset:
            self.ser.rts = True
            time.sleep(0.08)
            self.ser.rts = False
            self.ser.dtr = False
        self.ser.reset_input_buffer()
        self.ip = ""
        self.banner = ""

    def close(self) -> None:
        self.ser.close()

    def wait_ready(self, timeout: float = 25.0) -> str:
        deadline = time.time() + timeout
        chunks: list[bytes] = []
        while time.time() < deadline:
            data = self.ser.read(1024)
            if data:
                chunks.append(data)
                text = b"".join(chunks).decode("utf-8", errors="replace")
                if "Pico-Former ready" in text or "alloc FAIL" in text:
                    self.banner = text
                    for line in text.splitlines():
                        if line.startswith("wifi "):
                            self.ip = line.split()[1]
                    return text
        self.banner = b"".join(chunks).decode("utf-8", errors="replace")
        return self.banner

    def ping(self, timeout: float = 3.0) -> bool:
        self.ser.reset_input_buffer()
        self.ser.write(MAGIC_PING)
        self.ser.flush()
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline:
            buf += self.ser.read(16)
            if MAGIC_PING in buf:
                return True
        return False

    def infer(self, seq: int, n_marks, quality_ok, y_hist, marks, mask, timeout: float = 8.0) -> dict:
        self.ser.reset_input_buffer()
        self.ser.write(pack_window(seq, n_marks, quality_ok, y_hist, marks, mask))
        self.ser.flush()
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline:
            buf += self.ser.read(512)
            i = buf.find(MAGIC_RES)
            if i >= 0 and len(buf) >= i + RES_FRAME:
                return parse_result(buf[i : i + RES_FRAME])
        raise TimeoutError(f"no result for seq={seq} got={len(buf)} bytes")

    def burn(self, n: int, timeout: float | None = None) -> dict:
        timeout = timeout or (n * 1.2 + 5)
        self.ser.reset_input_buffer()
        self.ser.write(MAGIC_BURN + struct.pack("<H", int(n)))
        self.ser.flush()
        deadline = time.time() + timeout
        text = ""
        while time.time() < deadline:
            chunk = self.ser.read(256)
            if chunk:
                text += chunk.decode("utf-8", errors="replace")
                if "BURN done" in text:
                    break
        row = {"raw": text, "n": n}
        for line in text.splitlines():
            if line.startswith("BURN done"):
                parts = dict(p.split("=", 1) for p in line.split()[2:])
                row.update({k: int(v) for k, v in parts.items()})
        return row


def load_model():
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = CEQTNet()
    net.load_state_dict(blob["state"])
    net.eval()
    thresholds = np.asarray(blob["thresholds"], dtype=np.float32)
    model = CEQT(
        net=net,
        ink=InertiaKernel(),
        y_mean=float(blob["y_mean"]),
        y_std=float(blob["y_std"]),
        use_g=True,
        use_quality=True,
        use_type=True,
        thresholds=thresholds.astype(np.float64),
    )
    return model, thresholds


def load_test():
    with CACHE.open("rb") as f:
        return pickle.load(f)["test"]


def take(batch: WindowBatch, idx) -> WindowBatch:
    idx = np.asarray(idx, dtype=int)
    return WindowBatch(
        subject_id=batch.subject_id[idx],
        t0=batch.t0[idx],
        y_hist=batch.y_hist[idx],
        t_hist=batch.t_hist[idx],
        y_future=batch.y_future[idx],
        horizons=batch.horizons,
        mark_t=[batch.mark_t[int(i)] for i in idx],
        mark_a=[batch.mark_a[int(i)] for i in idx],
        mark_e=[batch.mark_e[int(i)] for i in idx],
        day_index=batch.day_index[idx],
        n_marks=batch.n_marks[idx],
    )


def quality_flag(mark_a) -> int:
    a = np.asarray(mark_a, dtype=float)
    if a.size == 0:
        return 0
    return int(np.any(np.isfinite(a) & (a > 0)))


def pair_stats(lab: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (lab == 1)).sum())
    fn = int(((pred == 0) & (lab == 1)).sum())
    fp = int(((pred == 1) & (lab == 0)).sum())
    tn = int(((pred == 0) & (lab == 0)).sum())
    fnr = float(fn / (tp + fn)) if (tp + fn) else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "fnr": fnr, "fpr": fpr}


def pct(xs: np.ndarray, q: float) -> float:
    if xs.size == 0:
        return float("nan")
    return float(np.quantile(xs, q))


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predict(model: CEQT, batch: WindowBatch) -> dict:
    dev = device()
    model.net.to(dev)
    return model.predict_batch(batch, device=dev)


def stratified_idx(test: WindowBatch, n: int, seed: int) -> np.ndarray:
    lab = event_labels(test.y_future, test.horizons)
    g = residual_gate(test.n_marks, test.mark_a, test.mark_e, use_g=True, use_quality=True)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i in range(len(test)):
        key = (int(lab[i, 3]), int(lab[i, 11]), int(g[i] > 0), int(test.n_marks[i] > 0))
        groups[key].append(i)
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    leftover = []
    for key, ids in groups.items():
        ids = np.array(ids)
        rng.shuffle(ids)
        take_n = min(len(ids), 20)
        picked.extend(ids[:take_n].tolist())
        leftover.extend(ids[take_n:].tolist())
    leftover = np.array(leftover)
    rng.shuffle(leftover)
    need = max(0, n - len(picked))
    picked.extend(leftover[:need].tolist())
    out = np.array(sorted(set(picked)), dtype=int)
    if out.size < n and leftover.size:
        extra = [i for i in leftover if i not in set(out.tolist())]
        out = np.array(sorted(out.tolist() + extra[: n - out.size]), dtype=int)
    return out[:n]


def extra_golden_idx(test: WindowBatch) -> dict[str, int]:
    first_mark = next(i for i in range(len(test)) if int(test.n_marks[i]) >= 1)
    first_empty = next(i for i in range(len(test)) if int(test.n_marks[i]) == 0)
    sid0 = int(test.subject_id[first_mark])
    other = next(
        i
        for i in range(len(test))
        if int(test.n_marks[i]) >= 1 and int(test.subject_id[i]) != sid0
    )
    return {"empty": first_empty, "golden": first_mark, "other_subject": other}


def send_one(board: Board, seq: int, test: WindowBatch, i: int, mark_row, mask_row) -> dict:
    q = quality_flag(test.mark_a[i])
    last_err = None
    for _attempt in range(3):
        try:
            return board.infer(seq, int(test.n_marks[i]), q, test.y_hist[i], mark_row, mask_row)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.15)
    raise RuntimeError(f"window {i} failed: {last_err}")


def summarize_batch(name: str, idx, test, pc, board_rows, thresholds) -> dict:
    yhat_b = np.stack([r["yhat"] for r in board_rows])
    log_b = np.stack([r["logits"] for r in board_rows])
    evt_b = np.stack([r["evt"] for r in board_rows])
    us = np.array([r["us_total"] for r in board_rows], dtype=np.float64)
    heap = np.array([r["heap_free"] for r in board_rows], dtype=np.int64)
    cores = [int(r["core"]) for r in board_rows]
    evt_pc = (pc["logits"] >= thresholds[None, :]).astype(np.int32)
    lab = event_labels(test.y_future, test.horizons)
    agree = evt_b == evt_pc
    d_y = np.max(np.abs(yhat_b - pc["yhat"]), axis=1)
    d_l = np.max(np.abs(log_b - pc["logits"]), axis=1)
    d_g = np.abs(np.array([r["g"] for r in board_rows]) - pc["g"])
    out = {
        "name": name,
        "n": int(len(idx)),
        "evt_agree": float(agree.mean()),
        "evt_mismatch": int((~agree).sum()),
        "yhat_maxabs_median": float(np.median(d_y)),
        "yhat_maxabs_max": float(np.max(d_y)),
        "logit_maxabs_median": float(np.median(d_l)),
        "logit_maxabs_max": float(np.max(d_l)),
        "g_maxabs_max": float(np.max(d_g)),
        "us_min": float(us.min() / 1000.0),
        "us_median": float(np.median(us) / 1000.0),
        "us_p95": pct(us, 0.95) / 1000.0,
        "us_max": float(us.max() / 1000.0),
        "heap_min": int(heap.min()),
        "cores": cores[:8],
        "core_all_1": all(c == 1 for c in cores),
        "hypo70_4h_board": pair_stats(lab[:, 3], evt_b[:, 3]),
        "hypo70_4h_pc": pair_stats(lab[:, 3], evt_pc[:, 3]),
        "hyper180_4h_board": pair_stats(lab[:, 11], evt_b[:, 11]),
        "hyper180_4h_pc": pair_stats(lab[:, 11], evt_pc[:, 11]),
        "pass_agree": float(agree.mean()) >= 0.999,
        "pass_pair": (
            abs(pair_stats(lab[:, 3], evt_b[:, 3])["fnr"] - pair_stats(lab[:, 3], evt_pc[:, 3])["fnr"]) <= 0.001
            and abs(pair_stats(lab[:, 3], evt_b[:, 3])["fpr"] - pair_stats(lab[:, 3], evt_pc[:, 3])["fpr"]) <= 0.001
            and abs(pair_stats(lab[:, 11], evt_b[:, 11])["fnr"] - pair_stats(lab[:, 11], evt_pc[:, 11])["fnr"]) <= 0.001
            and abs(pair_stats(lab[:, 11], evt_b[:, 11])["fpr"] - pair_stats(lab[:, 11], evt_pc[:, 11])["fpr"]) <= 0.001
        ),
        "pass_numeric": float(np.median(d_y)) <= 0.05 and float(np.median(d_l)) <= 1e-3,
    }
    return out


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def run_indexed(board: Board, model, test, idx, thresholds, tag: str, http_ip: str = "") -> dict:
    sub = take(test, idx)
    marks, mask = pack_marks(sub)
    pc = predict(model, sub)
    rows = []
    t0 = time.time()
    http_ok = 0
    http_fail = 0
    stop_http = threading.Event()

    def hammer():
        nonlocal http_ok, http_fail
        if not http_ip:
            return
        url = f"http://{http_ip}/"
        while not stop_http.is_set():
            try:
                with urllib.request.urlopen(url, timeout=1.5) as resp:
                    if resp.status == 200:
                        http_ok += 1
                    else:
                        http_fail += 1
            except (urllib.error.URLError, TimeoutError, OSError):
                http_fail += 1
            time.sleep(0.08)

    th = threading.Thread(target=hammer, daemon=True)
    if http_ip:
        th.start()
    for k, i in enumerate(idx):
        row = send_one(board, k + 1, test, int(i), marks[k], mask[k])
        if row["status"] != 0:
            row = send_one(board, k + 100000, test, int(i), marks[k], mask[k])
        rows.append(row)
        if (k + 1) % 32 == 0:
            print(f"  {tag} {k+1}/{len(idx)}  {row['us_total']/1000:.1f} ms", flush=True)
    stop_http.set()
    if http_ip:
        th.join(timeout=2)
    summary = summarize_batch(tag, idx, sub, pc, rows, thresholds)
    summary["wall_s"] = time.time() - t0
    summary["http_ok"] = http_ok
    summary["http_fail"] = http_fail
    summary["idx"] = [int(x) for x in idx]
    return summary, rows, pc, marks, mask


def phase_a(board, model, test, thresholds, n: int) -> dict:
    idx = stratified_idx(test, n, SEED)
    print(f"phase A n={len(idx)}", flush=True)
    summary, rows, pc, marks, mask = run_indexed(board, model, test, idx, thresholds, "A")
    np.savez(
        OUT / "sample512.npz",
        idx=idx,
        yhat_pc=pc["yhat"],
        logits_pc=pc["logits"],
        yhat_board=np.stack([r["yhat"] for r in rows]),
        logits_board=np.stack([r["logits"] for r in rows]),
        evt_board=np.stack([r["evt"] for r in rows]),
        us=np.array([r["us_total"] for r in rows]),
    )
    dump(OUT / "phase_a.json", summary)
    extras = extra_golden_idx(test)
    extra_out = {}
    for name, i in extras.items():
        sub = take(test, [i])
        m, mk = pack_marks(sub)
        pc1 = predict(model, sub)
        row = send_one(board, 9000 + i, test, i, m[0], mk[0])
        extra_out[name] = {
            "idx": int(i),
            "subject": int(test.subject_id[i]),
            "n_marks": int(test.n_marks[i]),
            "now": float(test.y_hist[i, -1]),
            "pred_1h_board": float(row["yhat"][11]),
            "pred_4h_board": float(row["yhat"][47]),
            "pred_1h_pc": float(pc1["yhat"][0, 11]),
            "pred_4h_pc": float(pc1["yhat"][0, 47]),
            "yhat_maxabs": float(np.max(np.abs(row["yhat"] - pc1["yhat"][0]))),
            "logit_maxabs": float(np.max(np.abs(row["logits"] - pc1["logits"][0]))),
            "g_board": row["g"],
            "g_pc": float(pc1["g"][0]),
            "us_ms": row["us_total"] / 1000.0,
            "evt_agree": bool(np.array_equal(row["evt"], (pc1["logits"][0] >= thresholds).astype(np.int32))),
        }
        print(f"  extra {name} idx={i} yMax={extra_out[name]['yhat_maxabs']:.4f}", flush=True)
    dump(OUT / "golden_extra.json", extra_out)
    summary["extras"] = extra_out
    return summary


def phase_d(board, model, test, thresholds, n: int, ip: str) -> dict:
    rng = np.random.default_rng(SEED + 1)
    idx = rng.choice(len(test), size=min(n, len(test)), replace=False)
    print(f"phase D jitter n={len(idx)} ip={ip}", flush=True)
    summary, rows, pc, _, _ = run_indexed(board, model, test, idx, thresholds, "D", http_ip=ip)
    dump(OUT / "jitter.json", summary)
    return summary


def phase_c(board, n: int = 80) -> dict:
    print(f"phase C burn n={n}", flush=True)
    row = board.burn(n)
    us = row.get("last_us")
    wall = row.get("wall_us")
    out = {
        "n": n,
        "last_us": us,
        "wall_us": wall,
        "wall_ms_per": (wall / n / 1000.0) if wall else None,
        "current_source": "none",
        "note": "无电流计：只记录墙钟。电流与 mJ 待仪表。",
    }
    dump(OUT / "energy.json", out)
    return out


def phase_b(board, model, test, thresholds, start: int = 0) -> dict:
    n = len(test)
    marks, mask = pack_marks(test)
    print(f"phase B PC predict n={n}", flush=True)
    pc = predict(model, test)
    evt_pc = (pc["logits"] >= thresholds[None, :]).astype(np.int32)
    lab = event_labels(test.y_future, test.horizons)
    agree_n = 0
    mismatch = 0
    d_y_max = 0.0
    d_l_max = 0.0
    us_all = []
    heap_min = 1 << 30
    fails = []
    mismatches = []
    ckpt_path = OUT / "phase_b_ckpt.npz"
    t0 = time.time()
    evt_b_all = np.zeros((n, N_EVT), dtype=np.int32)
    for i in range(start, n):
        try:
            row = send_one(board, i + 1, test, i, marks[i], mask[i])
        except Exception as exc:  # noqa: BLE001
            fails.append({"i": int(i), "err": str(exc)})
            continue
        evt_b_all[i] = row["evt"]
        same = np.array_equal(row["evt"], evt_pc[i])
        agree_n += int(same)
        n_bad = int(np.sum(row["evt"] != evt_pc[i]))
        mismatch += n_bad
        if n_bad:
            cells = np.where(row["evt"] != evt_pc[i])[0]
            mismatches.append(
                {
                    "idx": int(i),
                    "cells": [int(c) for c in cells],
                    "board": [int(x) for x in row["evt"][cells]],
                    "pc": [int(x) for x in evt_pc[i, cells]],
                    "logit_pc": [float(pc["logits"][i, c]) for c in cells],
                    "thr": [float(thresholds[c]) for c in cells],
                    "margin_pc": [float(pc["logits"][i, c] - thresholds[c]) for c in cells],
                }
            )
        d_y_max = max(d_y_max, float(np.max(np.abs(row["yhat"] - pc["yhat"][i]))))
        d_l_max = max(d_l_max, float(np.max(np.abs(row["logits"] - pc["logits"][i]))))
        us_all.append(row["us_total"])
        heap_min = min(heap_min, int(row["heap_free"]))
        if (i + 1) % 256 == 0 or i + 1 == n:
            np.savez(ckpt_path, done=i + 1, evt_board=evt_b_all[: i + 1], us=np.array(us_all))
            print(f"  B {i+1}/{n} agree_win={agree_n}/{i+1-start-len(fails)} heap={heap_min}", flush=True)
            dump(
                OUT / "phase_b.json",
                {
                    "done": i + 1,
                    "n": n,
                    "window_agree": agree_n,
                    "evt_mismatch_cells": mismatch,
                    "yhat_maxabs_max": d_y_max,
                    "logit_maxabs_max": d_l_max,
                    "us_median_ms": float(np.median(us_all) / 1000.0) if us_all else None,
                    "heap_min": heap_min,
                    "fails": fails,
                    "mismatches": mismatches,
                    "wall_s": time.time() - t0,
                    "hypo70_4h_board": pair_stats(lab[: i + 1, 3], evt_b_all[: i + 1, 3]),
                    "hypo70_4h_pc": pair_stats(lab[: i + 1, 3], evt_pc[: i + 1, 3]),
                    "hyper180_4h_board": pair_stats(lab[: i + 1, 11], evt_b_all[: i + 1, 11]),
                    "hyper180_4h_pc": pair_stats(lab[: i + 1, 11], evt_pc[: i + 1, 11]),
                },
            )
    done = n
    summary = {
        "done": done,
        "n": n,
        "window_agree": agree_n,
        "window_agree_rate": agree_n / max(done - len(fails), 1),
        "evt_agree": 1.0 - mismatch / max((done - len(fails)) * N_EVT, 1),
        "evt_mismatch_cells": mismatch,
        "yhat_maxabs_max": d_y_max,
        "logit_maxabs_max": d_l_max,
        "us_median_ms": float(np.median(us_all) / 1000.0) if us_all else None,
        "us_p95_ms": pct(np.array(us_all, dtype=float), 0.95) / 1000.0 if us_all else None,
        "us_max_ms": float(np.max(us_all) / 1000.0) if us_all else None,
        "heap_min": heap_min,
        "fails": fails,
        "mismatches": mismatches,
        "wall_s": time.time() - t0,
        "hypo70_4h_board": pair_stats(lab[:, 3], evt_b_all[:, 3]),
        "hypo70_4h_pc": pair_stats(lab[:, 3], evt_pc[:, 3]),
        "hyper180_4h_board": pair_stats(lab[:, 11], evt_b_all[:, 11]),
        "hyper180_4h_pc": pair_stats(lab[:, 11], evt_pc[:, 11]),
    }
    dump(OUT / "phase_b.json", summary)
    return summary


def attach(port: str, reset: bool) -> Board:
    board = Board(port, reset=reset)
    if reset:
        text = board.wait_ready()
        print(text[-800:], flush=True)
    elif not board.ping():
        print("ping miss; waiting ready after implicit chatter", flush=True)
        text = board.wait_ready(timeout=8)
        if "ready" not in text:
            board.close()
            board = Board(port, reset=True)
            text = board.wait_ready()
            print(text[-800:], flush=True)
    else:
        print("ping ok", flush=True)
    return board


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=PORT_DEFAULT)
    ap.add_argument("--phases", default="A,D,C,B", help="comma list: A,D,C,B")
    ap.add_argument("--n-sample", type=int, default=512)
    ap.add_argument("--n-jitter", type=int, default=128)
    ap.add_argument("--n-burn", type=int, default=80)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--ip", default="")
    ap.add_argument("--b-start", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("load model/cache", flush=True)
    model, thresholds = load_model()
    test = load_test()
    board = attach(args.port, reset=args.reset)
    ip = args.ip or board.ip
    phases = [p.strip().upper() for p in args.phases.split(",") if p.strip()]
    report = {"ip": ip, "phases": phases}
    try:
        if "A" in phases:
            report["A"] = phase_a(board, model, test, thresholds, args.n_sample)
        if "D" in phases:
            report["D"] = phase_d(board, model, test, thresholds, args.n_jitter, ip)
        if "C" in phases:
            report["C"] = phase_c(board, args.n_burn)
        if "B" in phases:
            report["B"] = phase_b(board, model, test, thresholds, start=args.b_start)
    finally:
        board.close()
    dump(OUT / "report.json", report)
    print(json.dumps({k: (v if k != "A" else {kk: v[kk] for kk in v if kk != "extras"}) for k, v in report.items() if k != "ip"}, indent=2)[:2000])


if __name__ == "__main__":
    main()
