#!/usr/bin/env python3
"""HTTP command diagnostics for the R2-9 energy test build: keep the serial port open to log node output, send mode commands one by one over HTTP, and record the raw response of each GET /api/energy.
Usage: python energy_http_diag.py --port <serial port> --ip <node IP>
Results: code/results/energy/http_diag/{serial.log, http.log}
"""

from __future__ import annotations

import argparse
import http.client
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf2  # noqa: E402
from run_energy_modes import OUT  # noqa: E402
from run_wireless import token  # noqa: E402

STEPS = [("4", "3"), ("12", str(2 | (5 << 8))), ("0", "0"), ("11", "0"), ("4", "2")]


def raw(ip, tok, method, path):
    conn = http.client.HTTPConnection(ip, 80, timeout=3.0)
    try:
        conn.request(method, path, headers={"Authorization": f"Bearer {tok}"})
        r = conn.getresponse()
        return f"{r.status} {r.read().decode('utf-8', errors='replace')}"
    except (OSError, http.client.HTTPException) as exc:
        return f"ERR {exc!r}"
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--ip", required=True)
    args = ap.parse_args()
    out = OUT / "http_diag"
    out.mkdir(parents=True, exist_ok=True)
    board = pf2.attach(args.port, reset=True, timeout=60)
    ser_log = [board.banner]
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            b = board._read()
            if b:
                ser_log.append(b.decode("utf-8", errors="replace"))

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    tok = token()
    log = []
    t0 = time.time()
    for mode, param in STEPS:
        log.append(f"{time.time() - t0:7.1f} POST mode={mode} param={param} -> "
                   f"{raw(args.ip, tok, 'POST', f'/api/energy?mode={mode}&param={param}')}")
        for _ in range(25):
            time.sleep(1.0)
            r = raw(args.ip, tok, "GET", "/api/energy")
            log.append(f"{time.time() - t0:7.1f} GET -> {r[:400]}")
            if r.startswith("200") and '"busy":0' in r:
                break
    stop.set()
    th.join(timeout=1)
    board.close()
    (out / "http.log").write_text("\n".join(log), encoding="utf-8")
    (out / "serial.log").write_text("".join(ser_log), encoding="utf-8")
    print("\n".join(log[-12:]))


if __name__ == "__main__":
    main()
