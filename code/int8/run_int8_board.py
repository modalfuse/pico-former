#!/usr/bin/env python3
"""R2-14 board: replay the tokens of 512 held-out windows on the full-model ESP-DL quantized firmware (firmware_espdl_full) and compare with the host ESP-PPQ executor.

Usage: python run_int8_board.py --port <port> --bits 8 [--n 512] [--tag q8]
Flash firmware_espdl_full first (model: results/int8/espdl/ceqt_int<bits>.espdl). The FP32-B firmware (firmware_fp32_idf)
uses the same serial protocol; with --tag fp32 it is compared with the host fp32_test.npz (exact GELU, float32).
Output: code/results/int8/board_<tag>/{board.json, board.npz, boot.log}; tag defaults to q<bits>
"""

from __future__ import annotations

import os
import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import serial

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "int8"
SAMPLE512 = Path(os.environ.get("PF_WORKSPACE", ".")) / "validation/sample512.npz"
REPLY = 4 + 4 + 4 + 4 * 64 + 12 + 1


def xor_of(b: bytes) -> int:
    x = 0
    for v in b:
        x ^= v
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--tag", default="")
    ap.add_argument("--baud", type=int, default=115200,
                    help="the console baud rate in the ESP-IDF sdkconfig is 115200 (460800 in sdkconfig.defaults does not take effect)")
    args = ap.parse_args()
    tag = args.tag or f"q{args.bits}"
    out = RES / f"board_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(np.load(SAMPLE512)["idx"], int)[:args.n]
    tok = np.load(RES / "tok_test.npy", mmap_mode="r")
    host = np.load(RES / f"{tag}_test.npz")
    ref = np.load(RES / "side_test.npz")

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    ser.dtr = False
    ser.rts = True
    time.sleep(0.1)
    ser.rts = False
    banner = b""
    t_end = time.time() + 30
    while b"QTK ready" not in banner and time.time() < t_end:
        banner += ser.read(4096)
    (out / "boot.log").write_bytes(banner)
    if b"QTK ready" not in banner:
        raise SystemExit("no QTK ready banner")
    print(banner.decode(errors="replace")[-1500:], flush=True)
    ser.reset_input_buffer()

    n = len(idx)
    us = np.zeros(n, np.int64)
    dn = np.zeros((n, 48), np.float32)
    lg = np.zeros((n, 16), np.float32)
    mem = np.zeros((n, 3), np.int64)
    retries = 0
    for k, i in enumerate(idx):
        payload = struct.pack("<I", k + 1) + np.ascontiguousarray(tok[i], np.float32).tobytes()
        for attempt in range(3):
            ser.write(b"QTK1" + payload + bytes([xor_of(payload)]))
            buf = b""
            t1 = time.time()
            while time.time() - t1 < 5:
                buf += ser.read(REPLY)
                j = buf.find(b"QRS1")
                if j >= 0 and len(buf) - j >= REPLY:
                    buf = buf[j:j + REPLY]
                    break
            ok = buf.startswith(b"QRS1") and len(buf) == REPLY and xor_of(buf[4:-1]) == buf[-1] \
                and struct.unpack("<I", buf[4:8])[0] == k + 1
            if ok:
                break
            retries += 1
            ser.reset_input_buffer()
        if not ok:
            raise RuntimeError(f"window {k} failed")
        us[k] = struct.unpack("<I", buf[8:12])[0]
        dn[k] = np.frombuffer(buf[12:12 + 192], np.float32)
        lg[k] = np.frombuffer(buf[204:268], np.float32)
        mem[k] = struct.unpack("<3I", buf[268:280])
        if (k + 1) % 64 == 0:
            print(f"  {k + 1}/{n} {us[k] / 1000:.1f} ms", flush=True)
    ser.close()

    hl, hd = host["logits"][idx], host["delta_norm"][idx]
    res = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "bits": args.bits, "tag": tag, "n": n, "serial_retries": retries,
           "logit_maxabs_board_vs_host_q": float(np.abs(lg - hl).max()),
           "delta_norm_maxabs_board_vs_host_q": float(np.abs(dn - hd).max()),
           "windows_bit_identical_logits": int((lg == hl.astype(np.float32)).all(1).sum()),
           "logit_maxabs_board_vs_fp32": float(np.abs(lg - ref["logits_fp32"][idx]).max()),
           "latency_ms": {"median": float(np.median(us) / 1000), "p95": float(np.percentile(us, 95) / 1000),
                          "p99": float(np.percentile(us, 99) / 1000), "max": float(us.max() / 1000),
                          "first": float(us[0] / 1000)},
           "int_free_min": int(mem[:, 0].min()), "int_min_free_min": int(mem[:, 1].min()),
           "psram_free_min": int(mem[:, 2].min())}
    (out / "board.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / "board.npz", idx=idx, us=us, delta_norm=dn, logits=lg, mem=mem)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
