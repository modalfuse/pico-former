#!/usr/bin/env python3
"""R2-18 fault injection F1-F5, F7 (F6, Wi-Fi outage, is covered by R2-10 T5).

The board must run the main firmware s3_n16r8. Pass criteria for each item follow the point-by-point response plan for R2-18:
- F1 invalid output: debug command 1 makes one logit of the next inference NaN; expect FAULT_OUTPUT(20) with no event bits, then OK on the following inference;
- F2 corrupted weights, F3 corrupted thresholds: flash s3_corrupt_w, s3_corrupt_t (W_EVT2[0] or W_THR[0] modified, original SHA-256 kept);
  expect FAULT_WEIGHTS(21) at boot, and 21 returned for submitted windows;
- F4 inference hang: debug command 2 puts the inference task into an infinite loop; the next window should return FAULT_INFER(22) within 2 s, then the task watchdog (5 s) restarts the board;
  after restart reset_reason is TASK_WDT(6) and reset_count increases by 1;
- F5 memory exhaustion: flash s3_eatsram (occupies internal SRAM before allocation until only 150,000 B remain); expect FAULT_MEMORY(23) at boot and no inference task started;
- F7 stale prediction: debug command 3 temporarily sets the staleness timeout to --stale-ms (default 8 s, instead of 10 min); after one window is sent, poll /api/last
  and record when stale changes from 0 to 1 and whether the event fields are removed accordingly; restore 600,000 ms at the end.
In addition, debug command 5 recomputes the SHA-256 of weights and thresholds once on the normal firmware (the same function as the M2 periodic check).
F2, F3, F5 flash test builds via PlatformIO; s3_n16r8 is flashed back at the end (deterministically rebuilt from the current source: 16b32e5f... since 2026-10-10, 10cba768... before).
The access token is read from firmware/wifi.yaml at run time; it is neither printed nor written to the results.

Usage: python run_faults.py --port <port> [--only F1,F4,F7] [--stale-ms 8000]
Results: code/results/fault/faults.json, <test>_*.log
"""

from __future__ import annotations

import os
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf2  # noqa: E402

CODE = HERE.parent
FW = CODE / "pico-former" / "firmware"
OUT = CODE / "results" / "fault"
CACHE = Path(os.environ.get("PF_WORKSPACE", ".")) / "runs/20260830_s20260828/cache/azt1d_s20260828_str15.pkl"
ESP_RST_TASK_WDT = 6
RESET = re.compile(r"RESET reason=(\d+) count=(\d+)")


def token() -> str:
    yml = FW / "wifi.yaml"
    if not yml.is_file():
        return ""
    for raw in yml.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.lower().startswith("api_token:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return ""


def http_json(ip: str, path: str, tok: str, timeout: float = 3.0) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(f"http://{ip}{path}", headers={"Authorization": f"Bearer {tok}"} if tok else {})
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def pick_window():
    batch = pf2.load_cache(CACHE)["test"]
    fr = pf2.frames(batch)
    for i in range(len(fr["y"])):
        st, y, f, m, q, fl, nd = pf2.guard(fr["y"][i], fr["feat"][i], fr["mask"][i], fr["q"][i], fr["flags"][i],
                                           fr["t0_s"][i], fr["t_last_s"][i])
        if st == pf2.OK and not fl & pf2.FLAG_CLAMPED and nd == 0:
            return i, dict(n_marks=fr["n_marks"][i], quality=fr["q"][i], flags=fr["flags"][i], t0_s=fr["t0_s"][i],
                           t_last_s=fr["t_last_s"][i], y_hist=fr["y"][i], feat=fr["feat"][i], mask=fr["mask"][i])
    raise RuntimeError("no OK window")


def summary(r: dict) -> dict:
    return {"status": int(r["status"]), "flags": int(r["flags"]), "evt_none": bool((r["evt"] == 0xFF).all()),
            "evt": r["evt"].tolist(), "logits_finite": bool(np.isfinite(r["logits"]).all()),
            "us_total": int(r["us_total"]), "reset_reason": int(r["reset_reason"]), "reset_count": int(r["reset_count"])}


def flash(env: str, log: Path) -> bool:
    p = subprocess.run(["pio", "run", "-e", env, "-t", "upload", "--upload-port", ARGS.port], cwd=FW,
                       capture_output=True, text=True)
    log.write_text(p.stdout[-6000:] + p.stderr[-3000:], encoding="utf-8")
    return p.returncode == 0


def f1(win):
    b = pf2.attach(ARGS.port, reset=True, timeout=60)
    (OUT / "F1_boot.log").write_text(b.banner, encoding="utf-8")
    base = summary(b.infer(1, **win))
    b.debug(1)
    t = time.time()
    hit = summary(b.infer(2, **win))
    dt = time.time() - t
    after = summary(b.infer(3, **win))
    b.close()
    return {"baseline": base, "poisoned": hit, "poisoned_round_trip_s": dt, "next": after,
            "pass": base["status"] == pf2.OK and hit["status"] == pf2.FAULT_OUTPUT and hit["evt_none"]
            and after["status"] == pf2.OK}


def f4(win):
    b = pf2.attach(ARGS.port, reset=True, timeout=60)
    b.debug(8)
    st0 = b.status()
    base = summary(b.infer(1, **win))
    b.debug(2)
    t = time.time()
    hit = summary(b.infer(2, **win, timeout=10.0))
    t_fault = time.time() - t
    buf, t_reset, m = "", None, None
    while time.time() - t < 30:
        buf += b._read().decode("utf-8", errors="replace")
        m = RESET.search(buf)
        if m:
            t_reset = time.time() - t
            break
    b.wait_ready(60)
    buf += b.banner
    (OUT / "F4_reboot.log").write_text(buf, encoding="utf-8")
    st1 = b.status()
    after = summary(b.infer(3, **win))
    b.debug(8)
    b.close()
    reason = int(m.group(1)) if m else -1
    return {"status_before": {k: st0[k] for k in ("reset_reason", "reset_count")}, "baseline": base,
            "after_hang_cmd": hit, "fault_frame_s": t_fault, "reset_line_s": t_reset, "reset_reason": reason,
            "status_after": {k: st1[k] for k in ("reset_reason", "reset_count", "fault")}, "next": after,
            "pass": hit["status"] == pf2.FAULT_INFER and hit["evt_none"] and t_fault < 3.0
            and reason == ESP_RST_TASK_WDT and st1["reset_count"] == st0["reset_count"] + 1 and after["status"] == pf2.OK}


def f7(win, tok):
    b = pf2.attach(ARGS.port, reset=True, timeout=60)
    ip = b.ip
    st = b.status()
    default_ms = st["stale_ms"]
    b.debug(3, ARGS.stale_ms)
    r = summary(b.infer(1, **win))
    t = time.time()
    trace = []
    while time.time() - t < ARGS.stale_ms / 1000 + 6:
        try:
            j = http_json(ip, "/api/last", tok)
            trace.append({"t_s": round(time.time() - t, 2), "age_s": j.get("age_s"), "stale": j.get("stale"),
                          "events_present": any(k.startswith("evt") and j[k] is not None for k in j)})
        except Exception as exc:  # noqa: BLE001
            trace.append({"t_s": round(time.time() - t, 2), "error": type(exc).__name__})
        time.sleep(0.5)
    b.debug(3, default_ms)
    st2 = b.status()
    b.close()
    fresh = [x for x in trace if x.get("stale") == 0]
    stale = [x for x in trace if x.get("stale") == 1]
    flip = stale[0] if stale else None
    return {"default_stale_ms": default_ms, "test_stale_ms": ARGS.stale_ms, "window": r, "trace": trace,
            "first_stale": flip, "restored_stale_ms": st2["stale_ms"],
            "pass": bool(fresh) and flip is not None and all(fr["events_present"] for fr in fresh)
            and all(not s["events_present"] for s in stale) and flip["age_s"] * 1000 >= ARGS.stale_ms
            and st2["stale_ms"] == default_ms}


def sha_check():
    b = pf2.attach(ARGS.port, reset=True, timeout=60)
    text = b.debug(5)
    b.close()
    js = json.loads(text.split("STATUS ", 1)[1].splitlines()[0])
    return {"sha_ok": js["sha_ok"], "sha_checks": js["sha_checks"], "fault": js["fault"], "pass": js["sha_ok"] == 1}


def fault_build(name, env, code, win):
    if not flash(env, OUT / f"{name}_flash.log"):
        return {"env": env, "flash": False, "pass": False}
    b = pf2.attach(ARGS.port, reset=True, timeout=60)
    banner = b.banner
    (OUT / f"{name}_boot.log").write_text(banner, encoding="utf-8")
    st = b.status()
    r = summary(b.infer(1, **win))
    b.close()
    return {"env": env, "flash": True, "banner_fault_line": f"FAULT state={code}" in banner,
            "status_fault": st["fault"], "sha_ok": st["sha_ok"], "int_free": st["int_free"], "window": r,
            "pass": st["fault"] == code and r["status"] == code and r["evt_none"]}


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--only", default="SHA,F1,F4,F7,F2,F3,F5")
    ap.add_argument("--stale-ms", type=int, default=8000)
    ARGS = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    want = ARGS.only.split(",")
    idx, win = pick_window()
    tok = token()
    res = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "window_idx": int(idx), "tests": {}}
    path = OUT / "faults.json"
    if path.exists():
        res["tests"] = json.loads(path.read_text(encoding="utf-8")).get("tests", {})
    steps = [("SHA", sha_check), ("F1", lambda: f1(win)), ("F4", lambda: f4(win)), ("F7", lambda: f7(win, tok)),
             ("F2", lambda: fault_build("F2", "s3_corrupt_w", pf2.FAULT_WEIGHTS, win)),
             ("F3", lambda: fault_build("F3", "s3_corrupt_t", pf2.FAULT_WEIGHTS, win)),
             ("F5", lambda: fault_build("F5", "s3_eatsram", pf2.FAULT_MEMORY, win))]
    try:
        for name, fn in steps:
            if name not in want:
                continue
            t = time.time()
            try:
                r = fn()
            except Exception as exc:  # noqa: BLE001
                r = {"error": f"{type(exc).__name__}: {exc}", "pass": False}
            r["seconds"] = round(time.time() - t, 1)
            res["tests"][name] = r
            print(name, "pass" if r.get("pass") else "FAIL", json.dumps(r)[:400], flush=True)
            path.write_text(json.dumps(res, indent=1, default=int), encoding="utf-8")
    finally:
        if any(n in want for n in ("F2", "F3", "F5")):
            ok = flash("s3_n16r8", OUT / "restore_flash.log")
            res["restored_s3_n16r8"] = ok
            path.write_text(json.dumps(res, indent=1, default=int), encoding="utf-8")
            print("restore s3_n16r8", ok, flush=True)


if __name__ == "__main__":
    main()
