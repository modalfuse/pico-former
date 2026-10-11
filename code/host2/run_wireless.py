#!/usr/bin/env python3
"""R2-10 wireless load tests T0-T5 (T6 additionally requires netem_mac.sh to inject packet loss and latency; this script is also used for recording).

Clients: --clients threads, each requesting the paths in --paths in turn (round-robin) every --interval-ms; each request opens a new TCP connection
(with --keepalive each client keeps one persistent connection and reconnects after an error),
using http.client directly against the board's LAN address (system proxy settings ignored). Each request records start time, latency, response bytes and outcome;
failures are classified as connect timeout, connection refused, connection reset, read timeout, HTTP error code, or other. A separate thread reads /api/status every --status-s seconds
(board-side HTTP counters, RSSI, reconnect count, internal memory and stack high-water marks; disable with --no-status for the original firmware, which lacks this endpoint).
Inference: with --push azt1d, AZT1D held-out windows are sent over serial in order (--windows of them, default all 19,152), recording the result frame for each window;
a window with no result frame within 5 s or a checksum failure on first send counts as a lost cycle and is resent (at most 3 times).
The no-load full held-out-set replay on the same frozen firmware (results/audit/azt1d_full/audit.npz) serves as the reference for comparing event bits and logits.
With --push none only HTTP is measured (T0, T2), for --minutes. For T1 on the original firmware v1.0.1, windows are sent by the push script of the original repository and this script acts only as the client.

Before starting, records the egress interface from `route get <board IP>`; stops if it is utun* (proxy TUN takeover).
The access token is read from firmware/wifi.yaml at run time; it is neither printed nor written to the results.

Usage:
  T2: python run_wireless.py --port <port> --tag t2_c4 --clients 4 --minutes 10
  T3: python run_wireless.py --port <port> --tag t3 --clients 4 --push azt1d
  T0: python run_wireless.py --ip <board IP> --tag t0 --clients 1 --paths / --minutes 10 --no-status
Results: code/results/wireless/<tag>/requests.csv, status.jsonl, windows.npz, summary.json
"""

from __future__ import annotations

import os
import argparse
import csv
import errno
import http.client
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf2  # noqa: E402

CODE = HERE.parent
FW = CODE / "pico-former" / "firmware"
OUT = CODE / "results" / "wireless"
REF = CODE / "results" / "audit" / "azt1d_full" / "audit.npz"
CACHE = Path(os.environ.get("PF_WORKSPACE", ".")) / "runs/20260830_s20260828/cache/azt1d_s20260828_str15.pkl"


def token() -> str:
    yml = FW / "wifi.yaml"
    if not yml.is_file():
        return ""
    for raw in yml.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.lower().startswith("api_token:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return ""


def route_iface(ip: str) -> str:
    p = subprocess.run(["route", "-n", "get", ip], capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.strip().startswith("interface:"):
            return line.split(":", 1)[1].strip()
    return ""


TLS_CTX = None  # set by --scheme https: verifies the board's self-signed certificate, no hostname check


def tls_context():
    import ssl
    ctx = ssl.create_default_context(cafile=str(FW / ".pio" / "cert" / "cert.pem"))
    ctx.check_hostname = False
    return ctx


def new_conn(ip, timeout):
    return (http.client.HTTPSConnection(ip, 443, timeout=timeout, context=TLS_CTX) if TLS_CTX
            else http.client.HTTPConnection(ip, 80, timeout=timeout))


def request_keep(conn, path, tok):
    """One request on a persistent connection (connected on first use); returns (outcome, http_code, bytes, conn_ok)."""
    try:
        if conn.sock is None:
            conn.connect()
    except socket.timeout:
        return "connect_timeout", 0, 0, False
    except ConnectionRefusedError:
        return "refused", 0, 0, False
    except OSError as exc:
        return ("unreachable" if exc.errno in (errno.EHOSTUNREACH, errno.ENETUNREACH) else "connect_error"), 0, 0, False
    try:
        conn.request("GET", path, headers={"Authorization": f"Bearer {tok}"} if tok else {})
        r = conn.getresponse()
        body = r.read()
    except socket.timeout:
        return "read_timeout", 0, 0, False
    except (ConnectionResetError, BrokenPipeError):
        return "reset", 0, 0, False
    except (http.client.HTTPException, OSError):
        return "read_error", 0, 0, False
    return ("ok" if r.status == 200 else f"http_{r.status}"), r.status, len(body), not r.will_close


def request(ip, path, tok, timeout):
    """One request on a fresh connection; returns (outcome, http_code, bytes)."""
    conn = new_conn(ip, timeout)
    try:
        try:
            conn.connect()
        except socket.timeout:
            return "connect_timeout", 0, 0
        except ConnectionRefusedError:
            return "refused", 0, 0
        except OSError as exc:
            return ("unreachable" if exc.errno in (errno.EHOSTUNREACH, errno.ENETUNREACH) else "connect_error"), 0, 0
        try:
            conn.request("GET", path, headers={"Authorization": f"Bearer {tok}"} if tok else {})
            r = conn.getresponse()
            body = r.read()
        except socket.timeout:
            return "read_timeout", 0, 0
        except (ConnectionResetError, BrokenPipeError):
            return "reset", 0, 0
        except (http.client.HTTPException, OSError):
            return "read_error", 0, 0
        return ("ok" if r.status == 200 else f"http_{r.status}"), r.status, len(body)
    finally:
        conn.close()


class Clients:
    def __init__(self, ip, tok, args, t0):
        self.ip, self.tok, self.args, self.t0 = ip, tok, args, t0
        self.rows, self.status = [], []
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def client(self, cid):
        paths = self.args.paths.split(",")
        k = 0
        conn = new_conn(self.ip, self.args.timeout_s) if self.args.keepalive else None
        while not self.stop.is_set():
            path = paths[k % len(paths)]
            ts = time.time()
            if conn is None:
                fresh = 1
                outcome, code, nbytes = request(self.ip, path, self.tok, self.args.timeout_s)
            else:
                fresh = int(conn.sock is None)
                outcome, code, nbytes, keep = request_keep(conn, path, self.tok)
                if not keep:
                    conn.close()
            dt = (time.time() - ts) * 1000
            with self.lock:
                self.rows.append((cid, round(ts - self.t0, 3), path, round(dt, 2), nbytes, outcome, code, fresh))
            k += 1
            rest = self.args.interval_ms / 1000 - (time.time() - ts)
            if rest > 0:
                self.stop.wait(rest)

    def poll_status(self):
        while not self.stop.wait(self.args.status_s):
            ts = time.time()
            conn = http.client.HTTPConnection(self.ip, 80, timeout=self.args.timeout_s)
            try:
                conn.request("GET", "/api/status", headers={"Authorization": f"Bearer {self.tok}"} if self.tok else {})
                js = json.loads(conn.getresponse().read().decode("utf-8"))
                js["t_s"] = round(ts - self.t0, 3)
                self.status.append(js)
            except (OSError, http.client.HTTPException, ValueError) as exc:
                self.status.append({"t_s": round(ts - self.t0, 3), "error": type(exc).__name__})
            finally:
                conn.close()

    def start(self):
        th = [threading.Thread(target=self.client, args=(c,), daemon=True) for c in range(self.args.clients)]
        if not self.args.no_status:
            th.append(threading.Thread(target=self.poll_status, daemon=True))
        for t in th:
            t.start()
        self.threads = th

    def join(self):
        self.stop.set()
        for t in self.threads:
            t.join(self.args.timeout_s + 2)


def push_windows(board, n_max, t0, out):
    batch = pf2.load_cache(CACHE)["test"]
    fr = pf2.frames(batch)
    n = min(n_max, len(fr["y"])) if n_max else len(fr["y"])
    keys = ("status", "us_total", "int_free", "int_min", "int_largest", "stack_infer", "stack_serial", "rssi", "link",
            "reconnects", "http_req", "http_fail", "reset_count")
    rec = {k: np.full(n, -1, np.int64) for k in keys}
    rec["t_s"] = np.full(n, np.nan)
    rec["evt"] = np.full((n, 16), -1, np.int32)
    rec["logits"] = np.full((n, 16), np.nan, np.float32)
    lost = np.zeros(n, np.int32)
    for i in range(n):
        r = None
        for attempt in range(3):
            try:
                r = board.infer(i + 1, fr["n_marks"][i], fr["q"][i], fr["flags"][i], fr["t0_s"][i], fr["t_last_s"][i],
                                fr["y"][i], fr["feat"][i], fr["mask"][i])
                if r["status"] not in (pf2.BUSY, pf2.BAD_FRAME):
                    break
            except TimeoutError:
                board.ser.reset_input_buffer()
            lost[i] += 1
        if r is None:
            raise RuntimeError(f"window {i} failed three times")
        rec["t_s"][i] = time.time() - t0
        for k in keys:
            rec[k][i] = r[k]
        rec["evt"][i], rec["logits"][i] = r["evt"], r["logits"]
        if (i + 1) % 256 == 0 or i + 1 == n:
            print(f"  windows {i + 1}/{n}  {r['us_total'] / 1000:.1f} ms  int_min={r['int_min']} "
                  f"rssi={r['rssi']} reconnects={r['reconnects']} lost={int((lost > 0).sum())}", flush=True)
            np.savez_compressed(out / "windows.npz", done=i + 1, lost=lost, **rec)
    np.savez_compressed(out / "windows.npz", done=n, lost=lost, **rec)
    return rec, lost


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def summarize(rows, status, rec, lost, args, meta):
    res = dict(meta)
    res["requests"] = {}
    for path in sorted({r[2] for r in rows}):
        rs = [r for r in rows if r[2] == path]
        ok = np.array([r[3] for r in rs if r[5] == "ok"])
        fails = {}
        for r in rs:
            if r[5] != "ok":
                fails[r[5]] = fails.get(r[5], 0) + 1
        res["requests"][path] = {
            "n": len(rs), "ok": int(len(ok)), "fail": int(len(rs) - len(ok)), "fail_by_type": fails,
            "latency_ms": {"p50": pct(ok, 50), "p95": pct(ok, 95), "p99": pct(ok, 99),
                           "max": float(ok.max()) if len(ok) else None},
            "bytes": sorted({r[4] for r in rs if r[5] == "ok"}),
            "per_client_n": [sum(1 for r in rs if r[0] == c) for c in range(args.clients)]}
        if args.keepalive:
            reuse = np.array([r[3] for r in rs if r[5] == "ok" and not r[7]])
            res["requests"][path]["new_connections"] = int(sum(r[7] for r in rs))
            res["requests"][path]["reused_latency_ms"] = {"n": int(len(reuse)), "p50": pct(reuse, 50),
                                                          "p95": pct(reuse, 95), "p99": pct(reuse, 99),
                                                          "max": float(reuse.max()) if len(reuse) else None}
    good = [s for s in status if "error" not in s]
    if good:
        res["status"] = {
            "n": len(status), "errors": len(status) - len(good),
            "int_min_min": min(s["int_min"] for s in good), "int_largest_min": min(s["int_largest"] for s in good),
            "stack_infer_min": min(s["stack_infer"] for s in good), "stack_sup_min": min(s["stack_sup"] for s in good),
            "http_stack_hw_min": min(s["http_stack_hw"] for s in good),
            "rssi_min": min(s["rssi"] for s in good), "rssi_max": max(s["rssi"] for s in good),
            "reconnects_max": max(s["reconnects"] for s in good), "down_total_ms_max": max(s["down_total_ms"] for s in good),
            "board_http_req": good[-1]["http_req"], "board_http_fail": good[-1]["http_fail"],
            "handler_us_max": max(s["handler_us_max"] for s in good), "fault": max(s["fault"] for s in good)}
    if rec is not None:
        done = rec["status"] >= 0
        ok = rec["status"] == pf2.OK
        us = rec["us_total"][ok] / 1000
        w = {"n": int(done.sum()), "lost_cycles": int((lost > 0).sum()), "resends": int(lost.sum()),
             "status": {int(a): int(b) for a, b in zip(*np.unique(rec["status"][done], return_counts=True))},
             "latency_ms": {"median": pct(us, 50), "p95": pct(us, 95), "p99": pct(us, 99),
                            "max": float(us.max()) if len(us) else None},
             "int_min_min": int(rec["int_min"][done].min()), "stack_infer_min": int(rec["stack_infer"][done].min()),
             "stack_serial_min": int(rec["stack_serial"][done].min()),
             "rssi_min": int(rec["rssi"][done].min()), "rssi_max": int(rec["rssi"][done].max()),
             "reconnects_max": int(rec["reconnects"][done].max()),
             "link_down_windows": int((rec["link"][done] == 0).sum())}
        if REF.exists():
            z = np.load(REF)
            m = int(done.sum())
            both = ok[:m] & (z["status"][:m] == pf2.OK)
            w["reference"] = str(REF.relative_to(CODE))
            w["evt_cells_disagree_vs_reference"] = int((rec["evt"][:m][both] != z["evt"][:m][both]).sum())
            w["logit_maxabs_vs_reference"] = float(np.abs(rec["logits"][:m][both] - z["logits"][:m][both]).max()) \
                if both.any() else None
            w["logits_bitwise_equal_vs_reference"] = bool(np.array_equal(rec["logits"][:m][both], z["logits"][:m][both]))
            zus = z["us_total"][:m][both] / 1000
            w["reference_latency_ms"] = {"median": pct(zus, 50), "p99": pct(zus, 99), "max": float(zus.max())}
        res["windows"] = w
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="")
    ap.add_argument("--ip", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--clients", type=int, default=1)
    ap.add_argument("--paths", default="/api/last")
    ap.add_argument("--interval-ms", type=float, default=200)
    ap.add_argument("--timeout-s", type=float, default=1.5)
    ap.add_argument("--status-s", type=float, default=10)
    ap.add_argument("--no-status", action="store_true")
    ap.add_argument("--push", choices=("none", "azt1d"), default="none")
    ap.add_argument("--windows", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--note", default="", help="network conditions, e.g. board-to-router distance, T6 injection settings")
    ap.add_argument("--scheme", choices=("http", "https"), default="http",
                    help="protocol for client requests; with https, connect to port 443 and verify against the local self-signed certificate (/api/status still uses HTTP)")
    ap.add_argument("--keepalive", action="store_true",
                    help="each client keeps one persistent connection and reconnects on the next request after a disconnect or error (default: new connection per request)")
    args = ap.parse_args()
    global TLS_CTX
    TLS_CTX = tls_context() if args.scheme == "https" else None
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)

    board, ip = None, args.ip
    if args.port:
        board = pf2.attach(args.port, reset=True, timeout=60)
        (out / "boot.log").write_text(board.banner, encoding="utf-8")
        ip = ip or board.ip
    if not ip:
        raise SystemExit("board IP unknown: give --ip or --port")
    iface = route_iface(ip)
    if iface.startswith("utun"):
        raise SystemExit(f"route to {ip} goes through {iface} (proxy TUN); exclude the LAN from TUN first")
    tok = token()
    meta = {"tag": args.tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "ip": ip, "route_interface": iface,
            "clients": args.clients, "paths": args.paths.split(","), "interval_ms": args.interval_ms,
            "timeout_s": args.timeout_s, "push": args.push, "note": args.note, "scheme": args.scheme,
            "keepalive": args.keepalive}
    print(json.dumps(meta), flush=True)

    t0 = time.time()
    cl = Clients(ip, tok, args, t0)
    cl.start()
    rec = lost = None
    try:
        if args.push == "azt1d":
            rec, lost = push_windows(board, args.windows, t0, out)
        else:
            end = t0 + args.minutes * 60
            while time.time() < end:
                time.sleep(5)
                with cl.lock:
                    n, nok = len(cl.rows), sum(1 for r in cl.rows if r[5] == "ok")
                print(f"  {time.time() - t0:.0f} s requests {n} ok {nok}", flush=True)
    finally:
        cl.join()
        meta["wall_s"] = round(time.time() - t0, 1)
        if board is not None:
            board.close()
        with (out / "requests.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["client", "t_s", "path", "latency_ms", "bytes", "outcome", "http_code", "new_connection"])
            w.writerows(cl.rows)
        with (out / "status.jsonl").open("w", encoding="utf-8") as f:
            for s in cl.status:
                f.write(json.dumps(s) + "\n")
    res = summarize(cl.rows, cl.status, rec, lost, args, meta)
    (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res)[:1500], flush=True)


if __name__ == "__main__":
    main()
