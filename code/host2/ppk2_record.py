#!/usr/bin/env python3
"""R2-9: PPK2 raw data recording (source-meter mode, 100 kS/s, current and digital channels D0-D7).

The PPK2 powers the 3V3 rail of the WiFiduino in source-meter mode and samples continuously. This script only writes the raw byte stream from the PPK2 to raw.bin,
records the host time and byte offset of each chunk read (index.csv), and saves the device calibration parameters (meta.json); decoding and interval extraction are done offline by ppk2_process.py,
so that samples are not dropped when per-sample decoding in Python cannot keep up with 100 kS/s. Run concurrently with run_energy_wifi.py; the host times of both are used to align measurement steps.
Usage: python ppk2_record.py --tag <same as run_energy_wifi.py> [--mv 3300] [--seconds 0 (until Ctrl-C)] [--port <device>]
      [--power-on] (turn on the PPK2 output; without it the output is assumed to be on already)
Results: code/results/energy/<tag>/ppk2/{raw.bin, index.csv, meta.json}
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from ppk2_api.ppk2_api import PPK2_API

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "energy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mv", type=int, default=3300)
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--port", default="")
    ap.add_argument("--power-on", action="store_true")
    ap.add_argument("--power-off-at-end", action="store_true")
    args = ap.parse_args()
    out = OUT / args.tag / "ppk2"
    out.mkdir(parents=True, exist_ok=True)

    port = args.port or next(iter(PPK2_API.list_devices()), "")
    if not port:
        raise SystemExit("PPK2 not found")
    ppk2 = PPK2_API(port, timeout=1, write_timeout=1, exclusive=True)
    ppk2.get_modifiers()
    ppk2.use_source_meter()
    ppk2.set_source_voltage(args.mv)
    if args.power_on:
        ppk2.toggle_DUT_power("ON")
        time.sleep(0.2)
    meta = {"port": port, "mv": args.mv, "mode": "source", "sample_rate_hz": 100000,
            "modifiers": ppk2.modifiers, "t_start_host_s": time.time(), "power_on_by_script": args.power_on}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    t_end = time.time() + args.seconds if args.seconds > 0 else float("inf")
    nbytes = 0
    with open(out / "raw.bin", "wb") as raw, open(out / "index.csv", "w") as idx:
        idx.write("t_host_s,byte_offset,nbytes\n")
        ppk2.start_measuring()
        last_print = time.time()
        while not stop["flag"] and time.time() < t_end:
            data = ppk2.get_data()
            if data:
                t = time.time()
                raw.write(data)
                idx.write(f"{t:.6f},{nbytes},{len(data)}\n")
                nbytes += len(data)
            else:
                time.sleep(0.005)
            if time.time() - last_print > 30:
                last_print = time.time()
                raw.flush()
                idx.flush()
                print(f"{time.strftime('%H:%M:%S')} {nbytes / 4 / 1e5:.1f} s recorded", flush=True)
        ppk2.stop_measuring()
    if args.power_off_at_end:
        ppk2.toggle_DUT_power("OFF")
    meta["t_end_host_s"] = time.time()
    meta["bytes"] = nbytes
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"done: {nbytes / 4 / 1e5:.1f} s, {nbytes / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
