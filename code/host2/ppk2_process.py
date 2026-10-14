"""Offline decoding of a PPK2 recording written by ppk2_record.py.

Reads results/energy/<tag>/ppk2/{raw.bin, meta.json, index.csv}, converts the
raw 4-byte samples to current with the calibration stored in meta.json (same
formula and range-switch spike filter as ppk2_api), and cuts intervals on the
digital inputs D0 (inference), D1 (HTTP handler), and D2 (cycle marker).

Outputs in the same directory:
  intervals.csv   one row per high interval on D0/D1/D2
  summary.json    record length, mean current, per-channel statistics
  trace_1khz.npz  current averaged over 1 ms and logic bits OR-ed over 1 ms

Usage:
  python ppk2_process.py --tag <tag> [--chunk-s 10] [--min-ms 0.05]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "energy"

FS_NOMINAL = 100_000
ADC_MULT = 1.8 / 163840
ALPHA = 0.18
ALPHA4 = 0.06
SPIKE_SAMPLES = 3
CHANNELS = {"D0": 0, "D1": 1, "D2": 2}


def calib_arrays(mod, vdd_mv):
    keys = [str(r) for r in range(5)]
    arr = {k: np.array([float(mod[k][r]) for r in keys]) for k in ("R", "GS", "GI", "O", "S", "I", "UG")}
    arr["offset"] = arr["S"] * (vdd_mv / 1000.0) + arr["I"]
    return arr


@njit(cache=True)
def spike_filter(amp, rng, out, state):
    """Range-switch filter of ppk2_api.get_adc_result, sample by sample.

    state = [rolling_avg, rolling_avg4, prev_range, consecutive, after_spike];
    NaN averages and prev_range -1 mean no sample seen yet.
    """
    ra, ra4, prev_r, consec, after = state[0], state[1], int(state[2]), int(state[3]), int(state[4])
    for k in range(amp.shape[0]):
        a = amp[k]
        r = rng[k]
        p_ra, p_ra4 = ra, ra4
        ra = a if np.isnan(ra) else ALPHA * a + (1 - ALPHA) * ra
        ra4 = a if np.isnan(ra4) else ALPHA4 * a + (1 - ALPHA4) * ra4
        if prev_r < 0:
            prev_r = r
        v = a
        if prev_r != r or after > 0:
            if prev_r != r:
                consec = 0
                after = SPIKE_SAMPLES
            else:
                consec += 1
            if r == 4:
                if consec < 2:
                    ra, ra4 = p_ra, p_ra4
                v = ra4
            else:
                v = ra
            after -= 1
        prev_r = r
        out[k] = v
    return np.array([ra, ra4, float(prev_r), float(consec), float(after)])


class Decoder:
    def __init__(self, mod, vdd_mv):
        self.c = calib_arrays(mod, vdd_mv)
        self.state = np.array([np.nan, np.nan, -1.0, 0.0, 0.0])
        self.prev_counter = None
        self.counter_gaps = 0

    def decode(self, words):
        adc_code = (words & 0x3FFF).astype(np.float64) * 4
        rng = np.minimum((words >> 14) & 0x7, 4).astype(np.int64)
        counter = ((words >> 18) & 0x3F).astype(np.int64)
        logic = ((words >> 24) & 0xFF).astype(np.uint8)

        c = self.c
        x = (adc_code - c["O"][rng]) * (ADC_MULT / c["R"][rng])
        amp = c["UG"][rng] * (x * (c["GS"][rng] * x + c["GI"][rng]) + c["offset"][rng])

        out = np.empty_like(amp)
        self.state = spike_filter(amp, rng, out, self.state)

        cprev = np.concatenate([[counter[0] if self.prev_counter is None else (self.prev_counter + 1) & 0x3F], (counter[:-1] + 1) & 0x3F])
        self.counter_gaps += int(np.count_nonzero(counter != cprev))
        self.prev_counter = int(counter[-1])
        return out * 1e6, logic


def host_time_map(index_csv):
    rows = list(csv.DictReader(open(index_csv)))
    if not rows:
        return None
    t = np.array([float(r["t_host_s"]) for r in rows])
    s = np.array([(int(r["byte_offset"]) + int(r["nbytes"])) // 4 for r in rows], dtype=np.float64)
    slope, icpt = np.polyfit(s, t, 1)
    return slope, icpt


def main():
    ap = argparse.ArgumentParser(description="Decode a PPK2 raw recording and cut intervals on D0/D1/D2.")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--chunk-s", type=float, default=10.0)
    ap.add_argument("--min-ms", type=float, default=0.05, help="drop high intervals shorter than this (glitches)")
    args = ap.parse_args()

    d = OUT / args.tag / "ppk2"
    meta = json.loads((d / "meta.json").read_text())
    vdd_mv = float(meta["mv"])
    volt = vdd_mv / 1000.0
    dec = Decoder(meta["modifiers"], vdd_mv)
    tmap = host_time_map(d / "index.csv")

    raw = np.memmap(d / "raw.bin", dtype="<u4", mode="r")
    n = len(raw)
    chunk = int(args.chunk_s * FS_NOMINAL)
    dec_n = 100

    q_cum = 0.0
    sum_ua = 0.0
    open_iv = {ch: None for ch in CHANNELS}
    prev_bit = {ch: 0 for ch in CHANNELS}
    intervals = []
    trace_i, trace_l = [], []
    carry_i = np.empty(0)
    carry_l = np.empty(0, dtype=np.uint8)

    for start in range(0, n, chunk):
        words = np.asarray(raw[start:start + chunk])
        ua, logic = dec.decode(words)
        cs = q_cum + np.cumsum(ua)
        q_before = np.concatenate([[q_cum], cs[:-1]])
        for ch, bit in CHANNELS.items():
            b = ((logic >> bit) & 1).astype(np.int8)
            db = np.diff(np.concatenate([[prev_bit[ch]], b]))
            for k in np.flatnonzero(db != 0):
                gi = start + int(k)
                if db[k] > 0:
                    open_iv[ch] = (gi, q_before[k])
                elif open_iv[ch] is not None:
                    s0, q0 = open_iv[ch]
                    intervals.append((ch, s0, gi, q_before[k] - q0))
                    open_iv[ch] = None
            prev_bit[ch] = int(b[-1])
        q_cum = float(cs[-1])
        sum_ua += float(ua.sum())

        ci = np.concatenate([carry_i, ua])
        cl = np.concatenate([carry_l, logic])
        m = len(ci) // dec_n * dec_n
        trace_i.append(ci[:m].reshape(-1, dec_n).mean(axis=1))
        trace_l.append(np.bitwise_or.reduce(cl[:m].reshape(-1, dec_n), axis=1))
        carry_i, carry_l = ci[m:], cl[m:]

    rows = []
    for ch, s0, s1, q in intervals:
        dur_s = (s1 - s0) / FS_NOMINAL
        if dur_s * 1e3 < args.min_ms:
            continue
        charge_c = q / FS_NOMINAL * 1e-6
        rows.append({
            "channel": ch,
            "start_s": round(s0 / FS_NOMINAL, 6),
            "end_s": round(s1 / FS_NOMINAL, 6),
            "duration_ms": round(dur_s * 1e3, 4),
            "mean_ma": round(charge_c / dur_s * 1e3, 5),
            "charge_mc": round(charge_c * 1e3, 6),
            "energy_mj": round(charge_c * volt * 1e3, 6),
            "t_host_start_s": round(tmap[0] * s0 + tmap[1], 4) if tmap else "",
        })
    with open(d / "intervals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["channel"])
        w.writeheader()
        w.writerows(rows)

    per_ch = {}
    for ch in CHANNELS:
        r = [x for x in rows if x["channel"] == ch]
        if not r:
            per_ch[ch] = {"n": 0}
            continue
        e = np.array([x["energy_mj"] for x in r])
        t = np.array([x["duration_ms"] for x in r])
        i = np.array([x["mean_ma"] for x in r])
        per_ch[ch] = {
            "n": len(r),
            "energy_mj_median": float(np.median(e)),
            "energy_mj_iqr": [float(np.percentile(e, 25)), float(np.percentile(e, 75))],
            "duration_ms_median": float(np.median(t)),
            "mean_ma_median": float(np.median(i)),
            "open_at_end": open_iv[ch] is not None,
        }

    dur_total = n / FS_NOMINAL
    summary = {
        "tag": args.tag,
        "samples": int(n),
        "duration_s": dur_total,
        "supply_v": volt,
        "mean_ma": sum_ua / max(n, 1) / 1e3,
        "energy_j": sum_ua / FS_NOMINAL * 1e-6 * volt,
        "counter_gaps": dec.counter_gaps,
        "host_s_per_sample": tmap[0] if tmap else None,
        "channels": per_ch,
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    np.savez_compressed(d / "trace_1khz.npz", current_ua=np.concatenate(trace_i) if trace_i else np.empty(0),
                        logic=np.concatenate(trace_l) if trace_l else np.empty(0, dtype=np.uint8), fs_hz=1000)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
