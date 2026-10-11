#!/usr/bin/env python3
"""R2-10, R2-18 Wi-Fi outage test (test build s3_linkdrop; the frozen firmware does not include debug command 9).

The host sends AZT1D held-out windows over serial at a fixed interval while --clients clients request /api/last over HTTP and read
/api/status every 2 s. For each event: after --pre-s seconds of normal link, debug command 9 is sent; the node drops Wi-Fi and does not reconnect for --hold-s seconds; afterwards
the firmware supervisor task reconnects through the normal path; after another --post-s seconds, the results computed during the outage are read via /api/history?since=<last window sequence number before the outage>
(read 20 s after the outage ends, retried every 5 s on failure; by then the number of windows since the outage began is below the 64 entries of the ring buffer)
and compared with the serial result frames one by one for sequence number, status and event bits.
Records: the first window with link=0 after the drop, the node LINK line (outage duration), the first window with link=1 after recovery and the time of the first successful request,
and inference latency and outputs during the outage (compared with the no-load replay results/audit/azt1d_full).
The access token is read from firmware/wifi.yaml at run time; it is neither printed nor written to the results.
Usage: python run_linkdrop.py --port <serial port> [--events 3 --hold-s 60]
Results: code/results/wireless/<tag>/summary.json, windows.npz, requests.csv, status.jsonl, board_text.log
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf2  # noqa: E402
from run_wireless import CACHE, CODE, OUT, REF, Clients, route_iface, token  # noqa: E402

KEYS = ("status", "us_total", "int_min", "rssi", "link", "reconnects", "http_req", "http_fail")


def history(ip, tok, since, timeout=5.0):
    conn = http.client.HTTPConnection(ip, 80, timeout=timeout)
    try:
        conn.request("GET", f"/api/history?since={since}", headers={"Authorization": f"Bearer {tok}"})
        return json.loads(conn.getresponse().read().decode("utf-8"))
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--tag", default="linkdrop")
    ap.add_argument("--events", type=int, default=3)
    ap.add_argument("--pre-s", type=float, default=45)
    ap.add_argument("--hold-s", type=float, default=60)
    ap.add_argument("--post-s", type=float, default=60)
    ap.add_argument("--gap-s", type=float, default=1.5, help="interval between successive windows; the number of windows during an outage must stay below the ring-buffer size of 64")
    ap.add_argument("--clients", type=int, default=2)
    args = ap.parse_args()
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)

    board = pf2.attach(args.port, reset=True, timeout=60)
    (out / "boot.log").write_text(board.banner, encoding="utf-8")
    ip = board.ip
    if route_iface(ip).startswith("utun"):
        raise SystemExit("route to board goes through utun")
    tok = token()
    cargs = SimpleNamespace(paths="/api/last", keepalive=False, timeout_s=1.5, interval_ms=200, status_s=2,
                            no_status=False, clients=args.clients)
    fr = pf2.frames(pf2.load_cache(CACHE)["test"])
    period = args.pre_s + args.hold_s + args.post_s
    n = int(args.events * period / args.gap_s) + 20
    rec = {k: np.full(n, -1, np.int64) for k in KEYS}
    rec["t_s"] = np.full(n, np.nan)
    rec["evt"] = np.full((n, 16), -1, np.int32)
    rec["logits"] = np.full((n, 16), np.nan, np.float32)
    resend = np.zeros(n, np.int32)
    text = []
    events = []

    t0 = time.time()
    cl = Clients(ip, tok, cargs, t0)
    cl.start()
    i = 0
    try:
        for e in range(args.events):
            ev = {"event": e}
            base = t0 + e * period
            phase_cmd = base + args.pre_s
            phase_end = base + period
            while time.time() < phase_end:
                now = time.time()
                if "cmd_t_s" not in ev and now >= phase_cmd:
                    ev["seq_before"] = i
                    ev["cmd_t_s"] = round(now - t0, 3)
                    board.text = ""
                    text.append(f"{now - t0:9.3f} " + board.debug(9, int(args.hold_s * 1000)).strip())
                    ev["hold_end_t_s"] = round(ev["cmd_t_s"] + args.hold_s, 3)
                    next_hist = phase_cmd + args.hold_s + 20
                if "cmd_t_s" in ev and "history" not in ev and now >= next_hist:
                    try:
                        ev["history"] = history(ip, tok, ev["seq_before"])
                        ev["history_t_s"] = round(time.time() - t0, 3)
                        ev.pop("history_error", None)
                    except (OSError, http.client.HTTPException, ValueError) as exc:
                        ev["history_error"] = type(exc).__name__
                        next_hist = now + 5
                if i >= n:
                    break
                r = None
                for _ in range(3):
                    try:
                        r = board.infer(i + 1, fr["n_marks"][i], fr["q"][i], fr["flags"][i], fr["t0_s"][i],
                                        fr["t_last_s"][i], fr["y"][i], fr["feat"][i], fr["mask"][i])
                        if r["status"] not in (pf2.BUSY, pf2.BAD_FRAME):
                            break
                    except TimeoutError:
                        board.ser.reset_input_buffer()
                    resend[i] += 1
                if r is None:
                    raise RuntimeError(f"window {i} failed three times")
                ts = time.time() - t0
                rec["t_s"][i] = ts
                for k in KEYS:
                    rec[k][i] = r[k]
                rec["evt"][i], rec["logits"][i] = r["evt"], r["logits"]
                if board.text:
                    for ln in board.text.splitlines():
                        if ln.strip():
                            text.append(f"{ts:9.3f} {ln.strip()}")
                    board.text = ""
                i += 1
                rest = args.gap_s - (time.time() - t0 - ts)
                if rest > 0:
                    time.sleep(rest)
            events.append(ev)
            print(json.dumps({k: ev[k] for k in ev if k != "history"}), flush=True)
    finally:
        cl.join()
        board.close()
    n = i
    for k in list(rec):
        rec[k] = rec[k][:n]
    resend = resend[:n]
    np.savez_compressed(out / "windows.npz", done=n, resend=resend, **rec)
    with (out / "requests.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["client", "t_s", "path", "latency_ms", "bytes", "outcome", "http_code", "new_connection"])
        w.writerows(cl.rows)
    with (out / "status.jsonl").open("w", encoding="utf-8") as f:
        for s in cl.status:
            f.write(json.dumps(s) + "\n")
    (out / "board_text.log").write_text("\n".join(text) + "\n", encoding="utf-8")

    z = np.load(REF)
    ref_evt, ref_log = z["evt"][:n], z["logits"][:n]
    rows = sorted(cl.rows, key=lambda r: r[1])
    link_down = [(float(m.group(1)), ln) for ln in text for m in [re.match(r"\s*([\d.]+) LINK down", ln)] if m]
    link_up = [(float(m.group(1)), int(m.group(2))) for ln in text
               for m in [re.match(r"\s*([\d.]+) LINK up t=\d+ down_ms=(\d+)", ln)] if m]
    res = {"tag": args.tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "ip": ip, "args": vars(args),
           "windows": n, "resends": int(resend.sum()),
           "status_counts": {int(a): int(b) for a, b in zip(*np.unique(rec["status"], return_counts=True))},
           "evt_cells_disagree_vs_reference": int((rec["evt"] != ref_evt).sum()),
           "logits_bitwise_equal_vs_reference": bool(np.array_equal(rec["logits"], ref_log)),
           "reference": str(REF.relative_to(CODE)), "events": []}
    for ev in events:
        c, h = ev["cmd_t_s"], ev["hold_end_t_s"]
        win = (rec["t_s"] >= c) & (rec["t_s"] < h)
        after = np.nonzero((rec["t_s"] >= c) & (rec["link"] == 0))[0]
        back = np.nonzero((rec["t_s"] >= h) & (rec["link"] == 1))[0]
        first_ok = next((r[1] for r in rows if r[1] >= h and r[5] == "ok"), None)
        last_ok = next((r[1] for r in reversed(rows) if r[1] < c and r[5] == "ok"), None)
        req = [r for r in rows if c <= r[1] < (first_ok if first_ok is not None else h)]
        outage = np.nonzero(win)[0]
        hist = {x["seq"]: x for x in ev.get("history", [])}
        bits = [int(sum(int(b) << k for k, b in enumerate(rec["evt"][j]))) if rec["status"][j] == 0 else 0
                for j in outage]
        match = [j + 1 in hist and hist[j + 1]["status"] == rec["status"][j] and hist[j + 1]["evt_bits"] == b
                 for j, b in zip(outage, bits)]
        lu = next((x for x in link_up if x[0] >= c), None)
        res["events"].append({
            "event": ev["event"], "cmd_t_s": c, "hold_end_t_s": h,
            "last_ok_request_before_s": last_ok,
            "first_link0_window_after_cmd_s": round(float(rec["t_s"][after[0]] - c), 3) if len(after) else None,
            "board_link_down_line": next((x[1] for x in link_down if x[0] >= c), None),
            "board_down_ms": lu[1] if lu else None,
            "first_link1_window_after_hold_s": round(float(rec["t_s"][back[0]] - h), 3) if len(back) else None,
            "first_ok_request_after_hold_s": round(first_ok - h, 3) if first_ok is not None else None,
            "requests_during_outage": len(req),
            "failed_during_outage_by_type": {k: sum(1 for r in req if r[5] == k) for k in {r[5] for r in req}},
            "windows_during_hold": int(win.sum()),
            "status_during_hold": {int(a): int(b) for a, b in zip(*np.unique(rec["status"][win], return_counts=True))},
            "infer_ms_during_hold": {"median": float(np.median(rec["us_total"][win]) / 1000) if win.any() else None,
                                     "max": float(rec["us_total"][win].max() / 1000) if win.any() else None},
            "history_read_after_hold_s": round(ev["history_t_s"] - h, 3) if "history_t_s" in ev else None,
            "history_items": len(hist), "hold_windows_in_history_and_equal": int(sum(match)),
            "history_error": ev.get("history_error"),
        })
    ok_all = rec["status"] == 0
    res["infer_ms"] = {"median": float(np.median(rec["us_total"][ok_all]) / 1000),
                       "max": float(rec["us_total"][ok_all].max() / 1000)}
    res["reconnects_final"] = int(rec["reconnects"][-1]) if n else None
    (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
