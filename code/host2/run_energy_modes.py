#!/usr/bin/env python3
"""R2-9 energy measurement mode driver (test build s3_energy, debug command 10; the frozen firmware does not include this command).

Switches through E1-E7 in turn following Section 4 of the energy measurement plan, recording the node's ENERGY lines (inference latency, Wi-Fi state, RSSI, CPU frequency).
In E5 and steps with the _poll suffix, the host requests /api/last over HTTP once per second and records the latency; in E7 each cycle runs:
cycle marker GPIO 6 high -> send one AZT1D held-out window over serial -> read /api/last over HTTP to confirm the result has been updated -> cycle marker low -> wait for the next cycle.
Without the PPK2, this script reports per-mode inference latency, HTTP latency and light-sleep wake-up interval; with the PPK2, run with --plan full and the current is recorded by the PPK2.
The access token is read from firmware/wifi.yaml at run time; it is neither printed nor written to the results.
Usage: python run_energy_modes.py --port <serial port> [--plan check|full|short|cycles|e7a|e7b] [--tag energy_check]
Results: code/results/energy/<tag>/summary.json, http.csv, board_text.log
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf2  # noqa: E402
from run_wireless import CACHE, CODE, route_iface, token  # noqa: E402

OUT = CODE / "results" / "energy"

# (step name, mode, parameter, poll HTTP)
PLANS = {
    "check": [
        ("E3", 3, 20, False), ("E1", 1, 20, False), ("E1_160MHz", -160, 20, False), ("E1_80MHz", -80, 20, False),
        ("restore", 0, 0, False), ("E2", 2, 15, False), ("E5", 5, 20, True), ("E4", 4, 15, False),
        ("E4_poll", 4, 20, True), ("E6", 6, 3 | (5 << 8), False), ("restore", 0, 0, False),
        ("E7a", 12, 3 | (20 << 8), False), ("restore", 0, 0, False), ("E7b", 7, 3, False), ("restore", 0, 0, False),
    ],
    # Two data sources for E7 (user decision, 2026-10-10): E7a replays windows pre-stored in Flash at a 5 min period (deployment setting in which
    # the sensing front end delivers readings over serial and results are published over Wi-Fi); E7b has the host send a window each cycle (via POST /api/window when driven over Wi-Fi). The difference between the two is the energy for delivering data over the air.
    "full": [
        ("E1", 1, 85, False), ("restore", 0, 0, False), ("E2", 2, 60, False), ("E3", 3, 85, False),
        ("E4", 4, 60, False), ("E5", 5, 60, True), ("E6", 6, 10 | (10 << 8), False), ("restore", 0, 0, False),
        ("E7a", 12, 12 | (300 << 8), False), ("restore", 0, 0, False), ("E7b", 7, 12, False), ("restore", 0, 0, False),
    ],
    "replay_check": [("E7a", 12, 3 | (20 << 8), False), ("restore", 0, 0, False)],
    # Reported runs (Table 9, Table S9b): three "short" runs on separate power cycles; E7a and E7b
    # three complete passes each ("cycles" twice, then "e7a" and "e7b" once each).
    "short": [
        ("E1", 1, 85, False), ("restore", 0, 0, False), ("E2", 2, 60, False), ("E3", 3, 85, False),
        ("E4", 4, 60, False), ("E5", 5, 60, True), ("E6", 6, 10 | (10 << 8), False), ("restore", 0, 0, False),
    ],
    "cycles": [
        ("E7a", 12, 12 | (300 << 8), False), ("restore", 0, 0, False), ("E7b", 7, 12, False), ("restore", 0, 0, False),
    ],
    # E7a alone: 12 Flash windows, period 300 s.
    "e7a": [
        ("E7a", 12, 12 | (300 << 8), False), ("restore", 0, 0, False),
    ],
    # E7b alone: 12 windows posted every 300 s.
    "e7b": [
        ("E7b", 7, 12, False), ("restore", 0, 0, False),
    ],
}
E7_PERIOD_S = {"check": 20.0, "full": 300.0, "cycles": 300.0, "e7b": 300.0}
ENERGY_RE = re.compile(r"ENERGY (\S+)(.*)")


class Poller(threading.Thread):
    def __init__(self, ip: str, tok: str, step: str, rows: list):
        super().__init__(daemon=True)
        self.ip, self.tok, self.step, self.rows = ip, tok, step, rows
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            t = time.time()
            self.rows.append(get_last(self.ip, self.tok, self.step))
            self.stop.wait(max(0.0, 1.0 - (time.time() - t)))


def get_last(ip: str, tok: str, step: str) -> dict:
    t = time.time()
    row = {"step": step, "t_s": t, "ok": 0, "ms": float("nan"), "seq": -1}
    try:
        conn = http.client.HTTPConnection(ip, 80, timeout=3.0)
        conn.request("GET", "/api/last", headers={"Authorization": f"Bearer {tok}"})
        resp = conn.getresponse()
        body = resp.read()
        row["ms"] = (time.time() - t) * 1000
        row["ok"] = int(resp.status == 200)
        if row["ok"]:
            row["seq"] = int(json.loads(body.decode("utf-8")).get("seq", -1))
        conn.close()
    except (OSError, http.client.HTTPException, ValueError):
        pass
    return row


def parse_energy(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        m = ENERGY_RE.search(line)
        if not m:
            continue
        rec = {"kind": m.group(1)}
        for kv in m.group(2).split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    rec[k] = int(v)
                except ValueError:
                    rec[k] = v
        out.append(rec)
    return out


def step_timeout(mode: int, param: int) -> float:
    if mode in (1, 3) or mode < 0:
        return 0.6 * (param or 80) + 40
    if mode in (2, 4, 5):
        return (param or 60) + 40
    if mode == 6:
        return (param & 0xFF or 10) * ((param >> 8 or 10) + 1.5) + 40
    if mode == 12:
        return (param & 0xFF or 12) * (param >> 8 or 300) + 40
    return 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--plan", choices=sorted(PLANS), default="check")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = OUT / (args.tag or f"energy_{args.plan}")
    out.mkdir(parents=True, exist_ok=True)

    board = pf2.attach(args.port, reset=True, timeout=60)
    (out / "boot.log").write_text(board.banner, encoding="utf-8")
    ip, tok = board.ip, token()
    if not ip or route_iface(ip).startswith("utun"):
        raise SystemExit(f"board ip {ip!r} missing or routed through utun")
    fr = None
    rows: list[dict] = []
    text: list[str] = []
    steps = []
    for name, mode, param, poll in PLANS[args.plan]:
        t0 = time.time()
        rec = {"step": name, "mode": mode, "param": param, "t_start_s": t0}
        if mode < 0:
            text.append(board.debug(10, 10 | (-mode << 8)))
            mode = 1
        poller = Poller(ip, tok, name, rows) if poll else None
        if poller:
            time.sleep(1.0)
            poller.start()
        if mode == 7:
            text.append(board.debug(10, 7))
            if fr is None:
                fr = pf2.frames(pf2.load_cache(CACHE)["test"])
            cycles = []
            for c in range(param):
                tc = time.time()
                text.append(board.debug(10, 8))
                i = c
                r = board.infer(100000 + c, fr["n_marks"][i], fr["q"][i], fr["flags"][i], fr["t0_s"][i],
                                fr["t_last_s"][i], fr["y"][i], fr["feat"][i], fr["mask"][i], timeout=10)
                h = get_last(ip, tok, f"E7_{c}")
                rows.append(h)
                text.append(board.debug(10, 9))
                cycles.append({"cycle": c, "status": r["status"], "us_total": r["us_total"], "http_ms": h["ms"],
                               "http_seq": h["seq"], "seq_match": int(h["seq"] == 100000 + c),
                               "busy_s": time.time() - tc})
                time.sleep(max(0.0, E7_PERIOD_S[args.plan] - (time.time() - tc)))
            rec["cycles"] = cycles
        else:
            text.append(board.debug(10, mode | (param << 8), timeout=step_timeout(mode, param)))
        if poller:
            poller.stop.set()
            poller.join()
        rec["energy"] = parse_energy(text[-1] if mode != 7 else "\n".join(text[-3:]))
        rec["wall_s"] = time.time() - t0
        steps.append(rec)
        print(name, json.dumps(rec["energy"][-2:] if rec["energy"] else rec.get("cycles", []))[:300], flush=True)
    final = get_last(ip, tok, "final")
    rows.append(final)
    board.close()

    with open(out / "http.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "t_s", "ok", "ms", "seq"])
        w.writeheader()
        w.writerows(rows)
    (out / "board_text.log").write_text("\n".join(text), encoding="utf-8")
    by_step = {}
    for r in rows:
        s = by_step.setdefault(r["step"], {"n": 0, "ok": 0, "ms": []})
        s["n"] += 1
        s["ok"] += r["ok"]
        if r["ok"]:
            s["ms"].append(r["ms"])
    http_sum = {k: {"n": v["n"], "ok": v["ok"],
                    "median_ms": sorted(v["ms"])[len(v["ms"]) // 2] if v["ms"] else None,
                    "max_ms": max(v["ms"]) if v["ms"] else None} for k, v in by_step.items()}
    summary = {"plan": args.plan, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "port": args.port,
               "steps": steps, "http": http_sum, "final_http_ok": final["ok"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(http_sum, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
