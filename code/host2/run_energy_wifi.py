#!/usr/bin/env python3
"""R2-9 Wi-Fi driver script for energy measurement with the PPK2 (test build s3_energy).

During measurement the WiFiduino is powered only by the PPK2 via the 3V3 rail, with USB-C, the DC jack and the serial module disconnected; all commands are sent over Wi-Fi:
POST /api/energy?mode=M&param=P starts a mode; GET /api/energy is polled until done equals this id, then the ENERGY record is retrieved;
during E1 and E6 the node turns Wi-Fi off and reconnects by itself afterwards, so polling failures in this period are expected.
In E5 and steps with the _poll suffix, /api/last is requested once per second. E7 is measured once for each of two data sources:
E7a (mode 12): the node takes pre-stored AZT1D held-out windows from Flash in order, runs inference and publishes at a 5 min period, and the host sends no requests meanwhile;
afterwards /api/history is read back and status and event bits are compared with results/audit/azt1d_full;
E7b (mode 7): one window is sent per cycle via POST /api/window; the node holds cycle marker GPIO 6 high from window receipt to result publication, and the host then reads /api/last to check the sequence number.
Mode 11 is sent first to release the UART0 pins and avoid leakage through the unpowered on-board CH340.
The node IP is given with --ip (obtained from the boot messages over USB; unchanged under a router DHCP reservation). The access token is read from firmware/wifi.yaml at run time.
Usage: python run_energy_wifi.py --ip <node IP> [--plan full|check] [--tag energy_wifi_full]
Results: code/results/energy/<tag>/summary.json, http.csv, steps.jsonl
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import sys
import time

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf2  # noqa: E402
from run_energy_modes import E7_PERIOD_S, OUT, PLANS, Poller, get_last, parse_energy, step_timeout  # noqa: E402
from run_wireless import CACHE, FW, route_iface, token  # noqa: E402


def request(ip: str, tok: str, method: str, path: str, body: bytes | None = None, timeout: float = 3.0):
    conn = http.client.HTTPConnection(ip, 80, timeout=timeout)
    try:
        hdr = {"Authorization": f"Bearer {tok}"}
        if body is not None:
            hdr["Content-Type"] = "application/octet-stream"
        conn.request(method, path, body=body, headers=hdr)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    finally:
        conn.close()


SEQ0_WIFI = 400000


def windows_meta() -> dict:
    return json.loads((FW / "include" / "energy_windows.meta.json").read_text(encoding="utf-8"))


def check_local(ip: str, tok: str, cycles: int, seq0: int | None = None) -> dict:
    """After E7a and E7b, read back /api/history and compare status and event bits with the azt1d_full reference (both use the same pre-stored windows)."""
    meta = windows_meta()
    seq0 = meta["seq0"] if seq0 is None else seq0
    st, hist = request(ip, tok, "GET", f"/api/history?since={seq0 - 1}", timeout=5.0)
    by_seq = {h["seq"]: h for h in hist} if st == 200 else {}
    rows = []
    for i in range(cycles):
        k = i % meta["n"]
        seq = seq0 + meta["test_idx"][k]
        h = by_seq.get(seq)
        bits = sum(int(v) << j for j, v in enumerate(meta["ref_evt"][k]))
        rows.append({"i": i, "seq": seq, "found": int(h is not None),
                     "status_match": int(h is not None and h["status"] == meta["ref_status"][k]),
                     "evt_match": int(h is not None and h["evt_bits"] == bits)})
    return {"n": cycles, "found": sum(r["found"] for r in rows), "status_match": sum(r["status_match"] for r in rows),
            "evt_match": sum(r["evt_match"] for r in rows), "rows": rows}


def run_cmd(ip: str, tok: str, path: str, body: bytes | None, timeout: float, quiet_s: float = 0.0) -> dict:
    """Submit a command and wait for the node to finish; do not poll for quiet_s seconds (so that host requests do not wake the radio); return {id, wall_s, log, energy}."""
    t0 = time.time()
    for _ in range(20):
        try:
            st, js = request(ip, tok, "POST", path, body)
        except (OSError, http.client.HTTPException, ValueError):
            time.sleep(1.0)
            continue
        if st == 202:
            break
        if st == 409:
            time.sleep(1.0)
            continue
        raise RuntimeError(f"POST {path} -> {st} {js}")
    else:
        raise RuntimeError(f"POST {path} not accepted")
    cid = js["id"]
    time.sleep(quiet_s)
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1.0)
        try:
            st, js = request(ip, tok, "GET", "/api/energy")
        except (OSError, http.client.HTTPException, ValueError):
            continue
        if st == 200 and not js.get("busy") and js.get("done") == cid:
            log = js.get("log", "")
            text = "\n".join("ENERGY " + s for s in log.split(";") if s)
            return {"id": cid, "wall_s": time.time() - t0, "log": log, "energy": parse_energy(text)}
    raise TimeoutError(f"{path} id={cid} not done in {timeout:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--plan", choices=sorted(PLANS), default="full")
    ap.add_argument("--tag", default="")
    ap.add_argument("--keep-uart", action="store_true", help="do not release the UART0 pins (use when the board is still connected via USB for functional verification)")
    args = ap.parse_args()
    out = OUT / (args.tag or f"energy_wifi_{args.plan}")
    out.mkdir(parents=True, exist_ok=True)
    ip, tok = args.ip, token()
    if route_iface(ip).startswith("utun"):
        raise SystemExit("route to board goes through utun")

    rows: list[dict] = []
    steps = []
    jl = open(out / "steps.jsonl", "a", encoding="utf-8")
    if not args.keep_uart:
        run_cmd(ip, tok, "/api/energy?mode=11", None, 30)
    fr = None
    for name, mode, param, poll in PLANS[args.plan]:
        t0 = time.time()
        rec = {"step": name, "mode": mode, "param": param, "t_start_s": t0}
        if mode < 0:
            run_cmd(ip, tok, f"/api/energy?mode=10&param={-mode}", None, 30)
            mode = 1
        if mode == 7:
            run_cmd(ip, tok, "/api/energy?mode=7", None, 40)
            if fr is None:
                fr = pf2.frames(pf2.load_cache(CACHE)["test"])
            meta = windows_meta()
            cycles = []
            for c in range(param):
                tc = time.time()
                i = meta["test_idx"][c % meta["n"]]
                seq = SEQ0_WIFI + i
                frame = pf2.pack_window(seq, fr["n_marks"][i], fr["q"][i], fr["flags"][i], fr["t0_s"][i],
                                        fr["t_last_s"][i], fr["y"][i], fr["feat"][i], fr["mask"][i])
                r = run_cmd(ip, tok, "/api/window", frame, 30)
                h = get_last(ip, tok, f"E7_{c}")
                rows.append(h)
                w = next((e for e in r["energy"] if e["kind"] == "window"), {})
                cycles.append({"cycle": c, "t_s": tc, "status": w.get("status"), "us_total": w.get("us"),
                               "http_ms": h["ms"], "seq_match": int(h["seq"] == seq), "busy_s": time.time() - tc})
                print(name, cycles[-1], flush=True)
                time.sleep(max(0.0, E7_PERIOD_S[args.plan] - (time.time() - tc)))
            rec["cycles"] = cycles
            rec["check"] = check_local(ip, tok, param, SEQ0_WIFI)
            print(name, {k: v for k, v in rec["check"].items() if k != "rows"}, flush=True)
        elif mode == 12:
            n, period = param & 0xFF or 12, param >> 8 or 300
            rec.update(run_cmd(ip, tok, f"/api/energy?mode=12&param={param}", None, 120,
                               quiet_s=max(0.0, (n - 1) * period + 5)))
            rec["check"] = check_local(ip, tok, n)
            print(name, {k: v for k, v in rec["check"].items() if k != "rows"}, flush=True)
        else:
            poller = Poller(ip, tok, name, rows) if poll else None
            if poller:
                poller.start()
            r = run_cmd(ip, tok, f"/api/energy?mode={mode}&param={param}", None, step_timeout(mode, param) + 30)
            if poller:
                poller.stop.set()
                poller.join()
            rec.update(r)
        rec["wall_s"] = time.time() - t0
        steps.append(rec)
        jl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jl.flush()
        print(name, json.dumps(rec.get("energy", [])[-2:])[:300], flush=True)
    jl.close()

    with open(out / "http.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "t_s", "ok", "ms", "seq"])
        w.writeheader()
        w.writerows(rows)
    by_step: dict = {}
    for r in rows:
        s = by_step.setdefault(r["step"], {"n": 0, "ok": 0, "ms": []})
        s["n"] += 1
        s["ok"] += r["ok"]
        if r["ok"]:
            s["ms"].append(r["ms"])
    http_sum = {k: {"n": v["n"], "ok": v["ok"], "median_ms": sorted(v["ms"])[len(v["ms"]) // 2] if v["ms"] else None,
                    "max_ms": max(v["ms"]) if v["ms"] else None} for k, v in by_step.items()}
    summary = {"plan": args.plan, "channel": "wifi", "uart_released": not args.keep_uart,
               "date": time.strftime("%Y-%m-%d %H:%M:%S"), "ip": ip, "steps": steps, "http": http_sum}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(http_sum, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
