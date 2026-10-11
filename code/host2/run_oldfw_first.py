"""R2-11: re-measure the first inference after power-up and back-to-back inference on the firmware used in the original submission.

Applies to the debug firmware (rice-s3/firmware, source of the 420.9 ms in Table 6) and validation firmware v1.0.1 (rice-s3/pico-former/firmware,
source of the ~398 ms on the held-out set): both run one inference on the golden window in setup() and print "done total=...", and only then connect to Wi-Fi;
afterwards they accept the PFB1 command (n back-to-back inferences on the golden window) and print "BURN done n= wall_us= last_us=".
Each round power-cycles the board via RTS reset, records the first inference, then sends PFB1 n=20. Output: results/latency/<tag>/first_vs_burn.json.
"""
import os
import argparse
import json
import re
import struct
import time
from pathlib import Path

import serial

HERE = Path(__file__).resolve().parent
RE_DONE = re.compile(r"done total=([\d.]+) ms")
RE_BURN = re.compile(r"BURN done n=(\d+) wall_us=(\d+) last_us=(\d+)")


def read_until(ser, pat, timeout):
    t0 = time.time()
    buf = b""
    while time.time() - t0 < timeout:
        buf += ser.read(4096)
        txt = buf.decode("latin-1")
        m = pat.search(txt)
        if m:
            return m, txt
    return None, buf.decode("latin-1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--boots", type=int, default=5)
    ap.add_argument("--burn", type=int, default=20)
    ap.add_argument("--tag", default="oldfw", help="output directory results/latency/<tag>")
    ap.add_argument("--fw", default="rice-s3/firmware .pio/build/s3_n16r8/firmware.bin", help="firmware description, written to the results")
    args = ap.parse_args()
    out = HERE.parent / "results/latency" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    ser = serial.Serial(args.port, 115200, timeout=0.2, dsrdtr=False, rtscts=False)
    ser.dtr = False
    ser.rts = False
    time.sleep(0.05)
    runs = []
    for b in range(args.boots):
        ser.dtr = False
        ser.rts = True
        time.sleep(0.2)
        ser.reset_input_buffer()
        ser.rts = False
        m, txt = read_until(ser, RE_DONE, 30)
        first = float(m.group(1)) if m else None
        _, txt2 = read_until(ser, re.compile(r"Pico-Former ready"), 40)
        wifi = "wifi skip/fail" not in txt2
        ser.reset_input_buffer()
        ser.write(b"PFB1" + struct.pack("<H", args.burn))
        mb, _ = read_until(ser, RE_BURN, 60)
        rec = {"boot": b + 1, "first_ms": first, "wifi_connected": wifi}
        if mb:
            n, wall, last = (int(x) for x in mb.groups())
            rec.update(burn_n=n, burn_mean_ms=wall / n / 1000.0, burn_last_ms=last / 1000.0)
        runs.append(rec)
        print(rec, flush=True)
    ser.close()
    firsts = [r["first_ms"] for r in runs if r["first_ms"] is not None]
    means = [r["burn_mean_ms"] for r in runs if "burn_mean_ms" in r]
    summary = {
        "firmware": args.fw,
        "runs": runs,
        "first_ms_median": sorted(firsts)[len(firsts) // 2] if firsts else None,
        "burn_mean_ms_median": sorted(means)[len(means) // 2] if means else None,
    }
    (out / "first_vs_burn.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("first_ms_median", "burn_mean_ms_median")}))


if __name__ == "__main__":
    main()
