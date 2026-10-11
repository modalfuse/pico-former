#!/usr/bin/env python3
"""Handshake diagnostics for the internal-RAM TLS build (s3_tls_int: all mbedTLS allocations in internal RAM).

After reset, keep the serial port open and log the node output, read /api/status over plain HTTP, then attempt --n full handshakes,
recording for each the outcome, exception type, and internal free heap and largest free internal block before and after the handshake.
Used to determine whether handshake failures come from internal-RAM allocation.
The access token is read from firmware/wifi.yaml at run time; it is neither printed nor written to the results.
Usage: python tls_int_diag.py --port <serial port> [--n 5]
Results: code/results/https/tls_int/diag.json, serial.log
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf2  # noqa: E402
from run_https import ctx, one, status  # noqa: E402
from run_wireless import route_iface, token  # noqa: E402

OUT = HERE.parent / "results" / "https" / "tls_int"
KEYS = ("int_free", "int_min", "int_largest", "spiram_free", "https_req", "https_fail", "https_stack_hw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    b = pf2.attach(args.port, reset=True, timeout=60)
    ip = b.ip
    lines = [f"{0.0:8.3f} {ln}" for ln in b.banner.splitlines()]
    t0 = time.time()
    stop = threading.Event()

    def reader():
        buf = b""
        while not stop.is_set():
            buf += b._read()
            while b"\n" in buf:
                ln, buf = buf.split(b"\n", 1)
                lines.append(f"{time.time() - t0:8.3f} {ln.decode('utf-8', errors='replace').rstrip()}")

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    time.sleep(3)
    if route_iface(ip).startswith("utun"):
        raise SystemExit("route to board goes through utun")
    tok = token()
    c = ctx()

    def st():
        try:
            s = status(ip, tok)
            return {k: s.get(k) for k in KEYS}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

    res = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "ip": ip, "status_before": st(), "attempts": []}
    for i in range(args.n):
        rec = {"i": i, "t_s": round(time.time() - t0, 3)}
        try:
            s, r = one(ip, c, tok)
            s.close()
            rec.update(ok=True, tls_ms=r["tls_ms"], status=r["status"])
        except Exception as exc:  # noqa: BLE001
            rec.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        time.sleep(1.0)
        rec["status_after"] = st()
        res["attempts"].append(rec)
        print(json.dumps(rec), flush=True)
    time.sleep(3)
    stop.set()
    th.join(1)
    b.close()
    res["n_ok"] = sum(a["ok"] for a in res["attempts"])
    (OUT / "diag.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (OUT / "serial.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("status_before", "n_ok")}), flush=True)


if __name__ == "__main__":
    main()
