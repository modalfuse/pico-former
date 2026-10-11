#!/usr/bin/env python3
"""R2-15 board latency: flash the latency_fw firmware of each architecture configuration in turn and read the median
per-window latency and on-chip memory.

Per configuration: pio upload (no rebuild) -> reset -> read serial until BENCH_DONE -> parse BENCH lines.
Usage: python run_latency_bench.py --port <port> [--configs base,d32_l1,...]
Output: code/results/design_space/latency_board.json, latency_board.log
"""

from __future__ import annotations

import os
import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import serial

HERE = Path(__file__).resolve().parent
FW = HERE / "latency_fw"
OUT = HERE.parent / "results" / "design_space"
ORDER = ["base", "d32_l1", "d32_l2", "d32_l3", "d48_l1", "d48_l3", "d64_l1", "d64_l2", "d64_l3", "d96_l2",
         "L48", "L60", "L84", "h2", "h8"]
PIO = shutil.which("pio") or str(Path.home() / ".platformio/penv/bin/pio")


def read_bench(port, timeout_s=600):
    with serial.Serial(port, 460800, timeout=1) as ser:
        ser.setDTR(False)
        ser.setRTS(True)
        time.sleep(0.1)
        ser.setRTS(False)
        lines, t0 = [], time.time()
        while time.time() - t0 < timeout_s:
            s = ser.readline().decode("utf-8", "replace").strip()
            if not s:
                continue
            lines.append(s)
            if s.startswith("BENCH_DONE"):
                break
    return lines


def parse(lines):
    r = {}
    for s in lines:
        if not s.startswith("BENCH"):
            continue
        m = re.search(r"before free=(\d+) largest=(\d+) after free=(\d+) largest=(\d+) alloc_ok=(\d)", s)
        if m:
            for k, v in zip(("before_free", "before_largest", "after_free", "after_largest", "alloc_ok"), m.groups()):
                r[k] = int(v)
            continue
        for k, v in re.findall(r"(\w+)=([-\w.]+)", s):
            try:
                r[k] = float(v) if "." in v else int(v)
            except ValueError:
                r[k] = v
        if s.startswith("BENCH_DONE"):
            r["done"] = s.split()[1] if len(s.split()) > 1 else ""
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--configs", default=",".join(ORDER))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "latency_board.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    log = (OUT / "latency_board.log").open("a", encoding="utf-8")
    for cfg in args.configs.split(","):
        t0 = time.time()
        p = subprocess.run([PIO, "run", "-d", str(FW), "-e", cfg, "-t", "nobuild", "-t", "upload", "--upload-port",
                            args.port], capture_output=True, text=True)
        if p.returncode != 0:
            log.write(f"{cfg} flash failed\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}\n")
            log.flush()
            print(f"{cfg} flash failed", flush=True)
            continue
        time.sleep(1.0)
        lines = read_bench(args.port)
        log.write(f"== {cfg} {time.strftime('%F %T')}\n" + "\n".join(lines) + "\n")
        log.flush()
        r = parse(lines)
        r["wall_s"] = time.time() - t0
        res[cfg] = r
        res_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"{cfg} done={r.get('done')} total_us={r.get('total')} enc={r.get('enc')} cross={r.get('cross')} "
              f"heads={r.get('heads')} int_free_after={r.get('after_free')}", flush=True)


if __name__ == "__main__":
    main()
