#!/usr/bin/env python3
"""R1-4 HTTPS endpoint measurement (frozen firmware: esp_https_server, port 443, Core 0, self-signed ECDSA P-256 certificate, at most 2 TLS sessions).

H1 full handshake: --n runs, each on a new TCP connection; TCP connect, TLS handshake, GET /api/last and total time are timed separately; no session reuse.
H2 session resumption: --n runs, each on a new TCP connection passing the previous SSLSession; records session_reused and handshake time.
H3 requests on an established session: --n consecutive requests to /api/last over one TLS connection with HTTP/1.1 keep-alive.
H4 memory per TLS session: open and hold 0, 1, 2, 3 TLS sessions in turn (each completes one request); at each step read, over plaintext HTTP,
   internal free heap, largest internal free block and free PSRAM from /api/status; the 3rd session checks the firmware session limit and LRU eviction.
The client verifies the server certificate against the local self-signed certificate (hostname not checked) and ignores system proxy settings; before starting, it checks that the route to the board does not go through utun*.
The access token is read from firmware/wifi.yaml at run time; it is neither printed nor written to the results.
Usage: python run_https.py --ip <board IP> [--n 100]; or --port <serial port> to reset the board and read the IP from the boot messages.
Results: code/results/https/<tag>/summary.json, samples.csv
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import socket
import ssl
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf2  # noqa: E402
from run_wireless import FW, route_iface, token  # noqa: E402

OUT = HERE.parent / "results" / "https"


def ctx():
    c = ssl.create_default_context(cafile=str(FW / ".pio" / "cert" / "cert.pem"))
    c.check_hostname = False
    return c


def get(sock, ip, path, tok):
    """Send one HTTP/1.1 GET on an open (TLS) socket and read the full response; returns (status, bytes)."""
    hdr = f"GET {path} HTTP/1.1\r\nHost: {ip}\r\nConnection: keep-alive\r\n"
    if tok:
        hdr += f"Authorization: Bearer {tok}\r\n"
    sock.sendall((hdr + "\r\n").encode())
    resp = http.client.HTTPResponse(sock)
    resp.begin()
    body = resp.read()
    return resp.status, len(body)


def one(ip, c, tok, session=None, path="/api/last", timeout=5.0):
    t0 = time.perf_counter()
    raw = socket.create_connection((ip, 443), timeout=timeout)
    t1 = time.perf_counter()
    s = c.wrap_socket(raw, server_hostname=None, session=session, do_handshake_on_connect=False)
    s.do_handshake()
    t2 = time.perf_counter()
    st, nb = get(s, ip, path, tok)
    t3 = time.perf_counter()
    rec = dict(tcp_ms=(t1 - t0) * 1e3, tls_ms=(t2 - t1) * 1e3, req_ms=(t3 - t2) * 1e3, total_ms=(t3 - t0) * 1e3,
               status=st, bytes=nb, reused=bool(s.session_reused), version=s.version(), cipher=s.cipher()[0])
    return s, rec


def status(ip, tok):
    conn = http.client.HTTPConnection(ip, 80, timeout=5)
    try:
        conn.request("GET", "/api/status", headers={"Authorization": f"Bearer {tok}"} if tok else {})
        return json.loads(conn.getresponse().read().decode("utf-8"))
    finally:
        conn.close()


def stats(a):
    a = np.asarray(a, float)
    return {"n": int(len(a)), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "min": float(a.min()), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="")
    ap.add_argument("--port", default="")
    ap.add_argument("--tag", default="wifiduino")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--gap-s", type=float, default=0.2, help="interval between successive connections in H1 and H2")
    args = ap.parse_args()
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    ip = args.ip
    if args.port:
        b = pf2.attach(args.port, reset=True, timeout=60)
        ip = ip or b.ip
        b.close()
        time.sleep(3)
    if not ip:
        raise SystemExit("board IP unknown: give --ip or --port")
    iface = route_iface(ip)
    if iface.startswith("utun"):
        raise SystemExit(f"route to {ip} goes through {iface} (proxy TUN); exclude the LAN from TUN first")
    tok = token()
    c = ctx()
    res = {"tag": args.tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "ip": ip, "route_interface": iface,
           "n": args.n, "status_before": status(ip, tok)}
    rows = []

    for phase in ("H1", "H2"):
        sess = None
        for i in range(args.n):
            s, r = one(ip, c, tok, session=sess if phase == "H2" else None)
            if phase == "H2":
                sess = s.session
            s.close()
            rows.append(dict(phase=phase, i=i, **r))
            time.sleep(args.gap_s)
        print(phase, "done", flush=True)

    s, r = one(ip, c, tok)
    rows.append(dict(phase="H3", i=-1, **r))
    for i in range(args.n):
        t0 = time.perf_counter()
        st, nb = get(s, ip, "/api/last", tok)
        rows.append(dict(phase="H3", i=i, req_ms=(time.perf_counter() - t0) * 1e3, status=st, bytes=nb))
    s.close()
    print("H3 done", flush=True)
    time.sleep(2)

    mem, held = [], []
    js = status(ip, tok)
    mem.append(dict(sessions=0, int_free=js["int_free"], int_largest=js["int_largest"], spiram_free=js["spiram_free"]))
    for k in (1, 2, 3):
        try:
            s, r = one(ip, c, tok)
            held.append(s)
            ok = r["status"] == 200
        except OSError as exc:
            ok, r = False, {"error": type(exc).__name__}
        time.sleep(1)
        js = status(ip, tok)
        alive = []
        for h in held:
            try:
                alive.append(get(h, ip, "/api/last", tok)[0] == 200)
            except (OSError, http.client.HTTPException):
                alive.append(False)
        mem.append(dict(sessions=k, open_ok=ok, held_alive=alive, int_free=js["int_free"],
                        int_largest=js["int_largest"], spiram_free=js["spiram_free"]))
    for h in held:
        h.close()
    time.sleep(2)
    js = status(ip, tok)
    mem.append(dict(sessions="closed", int_free=js["int_free"], int_largest=js["int_largest"],
                    spiram_free=js["spiram_free"]))
    print("H4 done", flush=True)

    for phase in ("H1", "H2"):
        rr = [x for x in rows if x["phase"] == phase]
        res[phase] = {k: stats([x[k] for x in rr]) for k in ("tcp_ms", "tls_ms", "req_ms", "total_ms")}
        res[phase]["reused"] = int(sum(x["reused"] for x in rr))
        res[phase]["status"] = sorted({x["status"] for x in rr})
        res[phase]["version"] = sorted({x["version"] for x in rr})
        res[phase]["cipher"] = sorted({x["cipher"] for x in rr})
    h3 = [x for x in rows if x["phase"] == "H3" and x["i"] >= 0]
    res["H3"] = {"req_ms": stats([x["req_ms"] for x in h3]), "status": sorted({x["status"] for x in h3}),
                 "bytes": sorted({x["bytes"] for x in h3})}
    res["H4"] = mem
    res["status_after"] = status(ip, tok)
    (out / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    keys = sorted({k for x in rows for k in x})
    with (out / "samples.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({p: {k: res[p][k]["median"] for k in ("tls_ms", "total_ms")} for p in ("H1", "H2")}),
          json.dumps(res["H3"]["req_ms"]), json.dumps(mem), flush=True)


if __name__ == "__main__":
    main()
