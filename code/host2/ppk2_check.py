"""Power-on check of the PPK2 wiring before an energy run.

Turns on the PPK2 source output, aborts if the mean current over any 1 s
exceeds --max-ma or stays below --min-ma, waits for the node on Wi-Fi, then
runs three passes (mode 3), one /api/last request, and 5 s idle (mode 2),
and reports the high intervals seen on D0, D1, and D2.

Usage:
  python ppk2_check.py --ip <node IP> [--mv 3300] [--max-ma 500]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import numpy as np
from ppk2_api.ppk2_api import PPK2_API

from ppk2_process import FS_NOMINAL, Decoder
from run_energy_wifi import request, run_cmd
from run_wireless import token


class Reader(threading.Thread):
    def __init__(self, ppk2, dec):
        super().__init__(daemon=True)
        self.ppk2, self.dec = ppk2, dec
        self.stop = threading.Event()
        self.ua, self.logic = [], []
        self.rem = b""

    def run(self):
        while not self.stop.is_set():
            data = self.ppk2.get_data()
            if not data:
                time.sleep(0.005)
                continue
            buf = self.rem + data
            n = len(buf) // 4 * 4
            self.rem = buf[n:]
            if n:
                ua, lg = self.dec.decode(np.frombuffer(buf[:n], dtype="<u4"))
                self.ua.append(ua)
                self.logic.append(lg)

    def samples(self):
        if not self.ua:
            return np.empty(0), np.empty(0, dtype=np.uint8)
        return np.concatenate(self.ua), np.concatenate(self.logic)


def intervals(logic, bit):
    b = ((logic >> bit) & 1).astype(np.int8)
    d = np.diff(np.concatenate([[0], b, [0]]))
    up, dn = np.flatnonzero(d > 0), np.flatnonzero(d < 0)
    return [(int(u), int(v)) for u, v in zip(up, dn)]


def main():
    ap = argparse.ArgumentParser(description="PPK2 power-on and marker check.")
    ap.add_argument("--ip", required=True)
    ap.add_argument("--mv", type=int, default=3300)
    ap.add_argument("--max-ma", type=float, default=500.0)
    ap.add_argument("--min-ma", type=float, default=1.0)
    args = ap.parse_args()

    port = next(iter(PPK2_API.list_devices()), "")
    if not port:
        raise SystemExit("PPK2 not found")
    ppk2 = PPK2_API(port, timeout=1, write_timeout=1, exclusive=True)
    ppk2.get_modifiers()
    ppk2.use_source_meter()
    ppk2.set_source_voltage(args.mv)
    dec = Decoder(ppk2.modifiers, args.mv)
    rd = Reader(ppk2, dec)
    ppk2.start_measuring()
    rd.start()
    marks = {}
    t0 = time.time()
    ppk2.toggle_DUT_power("ON")
    ok = True
    try:
        checked = 0
        while time.time() - t0 < 3.0:
            time.sleep(0.25)
            ua, _ = rd.samples()
            while len(ua) >= (checked + 1) * FS_NOMINAL // 4:
                seg = ua[checked * FS_NOMINAL // 4:(checked + 1) * FS_NOMINAL // 4]
                checked += 1
                if seg.mean() / 1e3 > args.max_ma:
                    raise RuntimeError(f"mean current {seg.mean() / 1e3:.1f} mA over 0.25 s exceeds limit")
        ua, _ = rd.samples()
        if ua[FS_NOMINAL:].mean() / 1e3 < args.min_ma:
            raise RuntimeError(f"mean current {ua[FS_NOMINAL:].mean() / 1e3:.3f} mA: no load on VOUT")
        print(f"boot: mean {ua.mean() / 1e3:.1f} mA, max {ua.max() / 1e3:.1f} mA over {len(ua) / FS_NOMINAL:.1f} s", flush=True)

        tok = token()
        for _ in range(60):
            try:
                st, _ = request(args.ip, tok, "GET", "/api/energy", timeout=2.0)
                if st == 200:
                    break
            except OSError:
                pass
            time.sleep(1.0)
        else:
            raise RuntimeError("node not reachable over Wi-Fi within 60 s")
        print(f"node reachable after {time.time() - t0:.1f} s", flush=True)
        run_cmd(args.ip, tok, "/api/energy?mode=11", None, 30)

        marks["infer"] = len(rd.samples()[0])
        run_cmd(args.ip, tok, "/api/energy?mode=3&param=3", None, 30)
        time.sleep(0.5)
        marks["http"] = len(rd.samples()[0])
        request(args.ip, tok, "GET", "/api/last", timeout=3.0)
        time.sleep(0.5)
        marks["idle"] = len(rd.samples()[0])
        run_cmd(args.ip, tok, "/api/energy?mode=2&param=5", None, 40)
        time.sleep(0.5)
        marks["end"] = len(rd.samples()[0])
    except Exception as e:
        ok = False
        print(f"ABORT: {e}", flush=True)
    finally:
        ppk2.toggle_DUT_power("OFF")
        time.sleep(0.2)
        rd.stop.set()
        rd.join()
        ppk2.stop_measuring()
    print("output off", flush=True)

    ua, lg = rd.samples()
    report = {"ok": ok, "seconds": len(ua) / FS_NOMINAL, "mean_ma": float(ua.mean() / 1e3) if len(ua) else None,
              "marks_s": {k: v / FS_NOMINAL for k, v in marks.items()}, "channels": {}}
    for ch, bit in (("D0", 0), ("D1", 1), ("D2", 2)):
        iv = intervals(lg, bit)
        report["channels"][ch] = [{"start_s": round(a / FS_NOMINAL, 4), "ms": round((b - a) / FS_NOMINAL * 1e3, 2),
                                   "mean_ma": round(float(ua[a:b].mean() / 1e3), 2)} for a, b in iv[:12]]
        report["channels"][ch + "_count"] = len(iv)
    print(json.dumps(report, indent=1), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
