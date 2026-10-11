#!/usr/bin/env python3
"""R2-19 on-board check of the deployment build s3_deploy (-DDEPLOY_BUILD=1).

Reads the boot messages after reset and checks:
  D1 the boot message is `WEB http=0 https=1 ok=1` and `Pico-Former ready` appears;
  D2 port 80 refuses connections (plaintext HTTP disabled);
  D3 HTTPS requests to /api/last, /api/history and /api/status without a token all return 401;
  D4 HTTPS requests to the three paths with a token all return 200, and sha_ok in /api/status is 1;
  D5 the serial port does not respond to audit commands (no `STATUS ` line within 3 s after debug command 4).
The token is read from firmware/wifi.yaml at run time; it is neither printed nor written to the results.
Usage: python deploy_check.py --port <port>
Results: code/results/security/deploy_check/summary.json, boot.log
"""

from __future__ import annotations

import os
import argparse
import http.client
import json
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf2  # noqa: E402
from run_https import ctx  # noqa: E402
from run_wireless import route_iface, token  # noqa: E402

OUT = HERE.parent / "results" / "security" / "deploy_check"
PATHS = ("/api/last", "/api/history", "/api/status")


def https_get(ip, c, path, tok):
    raw = socket.create_connection((ip, 443), timeout=10)
    s = c.wrap_socket(raw, server_hostname=None)
    try:
        hdr = f"GET {path} HTTP/1.1\r\nHost: {ip}\r\nConnection: close\r\n"
        if tok:
            hdr += f"Authorization: Bearer {tok}\r\n"
        s.sendall((hdr + "\r\n").encode())
        resp = http.client.HTTPResponse(s)
        resp.begin()
        return resp.status, resp.read()
    finally:
        s.close()


def port80(ip):
    try:
        socket.create_connection((ip, 80), timeout=3).close()
        return "open"
    except ConnectionRefusedError:
        return "refused"
    except OSError as exc:
        return f"error:{type(exc).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    board = pf2.attach(args.port, reset=True, timeout=60)
    (OUT / "boot.log").write_text(board.banner, encoding="utf-8")
    ip, tok = board.ip, token()
    res = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "ip": ip}
    res["D1_web_line"] = next((l.strip() for l in board.banner.splitlines() if l.startswith("WEB ")), "")
    res["D1_ready"] = "Pico-Former ready" in board.banner
    res["D1_ok"] = res["D1_web_line"] == "WEB http=0 https=1 ok=1" and res["D1_ready"]
    if not ip or route_iface(ip).startswith("utun"):
        raise SystemExit(f"board ip {ip!r} missing or routed through utun")
    time.sleep(2)

    res["D2_port80"] = port80(ip)
    res["D2_ok"] = res["D2_port80"] == "refused"

    c = ctx()
    res["D3_status"] = {p: https_get(ip, c, p, "")[0] for p in PATHS}
    res["D3_ok"] = all(v == 401 for v in res["D3_status"].values())
    d4 = {}
    for p in PATHS:
        st, body = https_get(ip, c, p, tok)
        d4[p] = st
        if p == "/api/status" and st == 200:
            js = json.loads(body.decode("utf-8"))
            res["D4_sha_ok"] = js.get("sha_ok")
            res["D4_fault"] = js.get("fault")
    res["D4_status"] = d4
    res["D4_ok"] = all(v == 200 for v in d4.values()) and res.get("D4_sha_ok") == 1

    text = board.debug(4, timeout=3.0)
    res["D5_serial_bytes"] = len(text)
    res["D5_ok"] = "STATUS " not in text
    board.close()

    res["all_ok"] = all(res[k] for k in ("D1_ok", "D2_ok", "D3_ok", "D4_ok", "D5_ok"))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
