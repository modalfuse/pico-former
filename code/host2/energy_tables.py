"""Energy terms of Table 9 and Supplementary Table S9b from the PPK2 recordings.

Two stages:

  extract  Decodes results/energy/<tag>/ppk2/raw.bin (ppk2_process.Decoder) and
           assigns D0/D1 pulses and 1 s blocks to the host steps of
           results/energy/<tag>/steps.jsonl. Writes
           results/energy/<tag>/ppk2/energy_steps.json. Needs raw.bin.
  tables   Reads energy_steps.json of every run and summary.json of the reset
           recordings, and writes results/energy/energy_tables.json.

Rules:
  - Sample times are nominal (10 us). The stream occasionally drops whole USB
    packets during host stalls, which the 6-bit sample counter does not always
    flag; such a drop shows as a lasting step in host arrival time minus
    nominal time. A sample is placed on the host clock with the lower envelope
    of that difference taken over the later index.csv rows, which aligns
    pulses with the host steps. The total step is reported as unrecorded_s.
  - D0 pulses of 200-800 ms are forward passes. A pass split by a low glitch is
    merged first: gap below 5 ms, one piece shorter than 200 ms, merged length
    at most 800 ms. E1 and E3 drop their first 5 passes.
  - D1 pulses of 0.3-30 ms are HTTP handler intervals.
  - Idle is the median of the 1 s blocks of a step in which D0 stays low.
  - Energy is V * integral of I over the interval at the PPK2 output voltage.
  - Each quantity is the median over the passes of one run, then the median
    over the runs.

Usage:
  python energy_tables.py extract --tag energy_short_r1 [--tag ...]
  python energy_tables.py tables
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ppk2_process import FS_NOMINAL, Decoder

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "energy"

RESET_TAGS = ["base_reset", "base_reset_r2", "base_reset_r3"]
SHORT_TAGS = ["energy_short_r1", "energy_short_r2", "energy_short_r3"]
E7A_TAGS = ["energy_cycles_r1", "energy_cycles_r2", "energy_e7a_r3"]
E7B_TAGS = ["energy_cycles_r1", "energy_cycles_r2", "energy_e7b_r3"]
MERGE_MS = 5.0
D0_MS = (200.0, 800.0)
D1_MS = (0.3, 30.0)
DROP_FIRST = {"E1": 5, "E3": 5}


def read_steps(tag):
    steps = []
    for line in (OUT / tag / "steps.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("step") != "restore":
            steps.append({"step": rec["step"], "t0": rec["t_start_s"], "t1": rec["t_start_s"] + rec["wall_s"]})
    return steps


def extract(tag, chunk_s=10.0):
    d = OUT / tag / "ppk2"
    meta = json.loads((d / "meta.json").read_text())
    volt = float(meta["mv"]) / 1000.0
    dec = Decoder(meta["modifiers"], float(meta["mv"]))
    t_start = float(meta["t_start_host_s"])
    idx = np.loadtxt(d / "index.csv", delimiter=",", skiprows=1, ndmin=2)
    row_end = (idx[:, 1] + idx[:, 2]) / 4
    lag = np.minimum.accumulate((idx[:, 0] - t_start - row_end / FS_NOMINAL)[::-1])[::-1]

    def host_t(i):
        k = min(int(np.searchsorted(row_end, i)), len(lag) - 1)
        return t_start + i / FS_NOMINAL + lag[k]

    steps = read_steps(tag)
    acc = [{"step": s["step"], "t0": s["t0"], "t1": s["t1"], "sum_ua": 0.0, "n": 0,
            "d0": [], "d1": [], "idle_1s": []} for s in steps]

    def find(t):
        for a in acc:
            if a["t0"] <= t < a["t1"]:
                return a
        return None

    raw = np.memmap(d / "raw.bin", dtype="<u4", mode="r")
    n = len(raw)
    chunk = int(chunk_s * FS_NOMINAL)
    q = 0.0
    open_iv = {0: None, 1: None}
    prev_bit = {0: 0, 1: 0}
    ivs = {0: [], 1: []}
    for start in range(0, n, chunk):
        ua, logic = dec.decode(np.asarray(raw[start:start + chunk]))
        cs = q + np.cumsum(ua)
        q_before = np.concatenate([[q], cs[:-1]])
        for bit in (0, 1):
            b = ((logic >> bit) & 1).astype(np.int8)
            db = np.diff(np.concatenate([[prev_bit[bit]], b]))
            for k in np.flatnonzero(db != 0):
                gi = start + int(k)
                if db[k] > 0:
                    open_iv[bit] = (gi, float(q_before[k]))
                elif open_iv[bit] is not None:
                    s0, q0 = open_iv[bit]
                    ivs[bit].append([s0, gi, float(q_before[k]) - q0])
                    open_iv[bit] = None
            prev_bit[bit] = int(b[-1])
        for sec in range(len(ua) // FS_NOMINAL):
            a0, a1 = sec * FS_NOMINAL, (sec + 1) * FS_NOMINAL
            a = find(host_t(start + a0))
            if a is None:
                continue
            a["sum_ua"] += float(ua[a0:a1].sum())
            a["n"] += a1 - a0
            if int((logic[a0:a1] & 1).max()) == 0:
                a["idle_1s"].append(round(float(ua[a0:a1].mean()) / 1e3, 4))
        q = float(cs[-1])

    merged = []
    ms_of = lambda a, b: (b - a) / FS_NOMINAL * 1e3
    for s0, s1, qq in ivs[0]:
        if (merged and ms_of(merged[-1][1], s0) < MERGE_MS
                and min(ms_of(*merged[-1][:2]), ms_of(s0, s1)) < D0_MS[0]
                and ms_of(merged[-1][0], s1) <= D0_MS[1]):
            merged[-1][1] = s1
            merged[-1][2] += qq
        else:
            merged.append([s0, s1, qq])
    for bit, rows, lim in ((0, merged, D0_MS), (1, ivs[1], D1_MS)):
        for s0, s1, qq in rows:
            ms = (s1 - s0) / FS_NOMINAL * 1e3
            if not lim[0] <= ms <= lim[1]:
                continue
            a = find(host_t(s0))
            if a is None:
                continue
            ma = qq / (s1 - s0) / 1e3
            a["d0" if bit == 0 else "d1"].append([round(s0 / FS_NOMINAL, 5), round(ms, 4), round(ma, 4)])

    out = {"tag": tag, "supply_v": volt, "samples": int(n), "duration_s": n / FS_NOMINAL,
           "counter_gaps": dec.counter_gaps, "unrecorded_s": float(lag[-1] - lag[0]), "steps": []}
    for a in acc:
        out["steps"].append({
            "step": a["step"],
            "mean_ma": a["sum_ua"] / a["n"] / 1e3 if a["n"] else None,
            "seconds": a["n"] / FS_NOMINAL,
            "d0_start_s_ms_ma": a["d0"],
            "d1_start_s_ms_ma": a["d1"],
            "idle_1s_ma": a["idle_1s"],
        })
    (d / "energy_steps.json").write_text(json.dumps(out), encoding="utf-8")
    print(tag, {s["step"]: (len(s["d0_start_s_ms_ma"]), len(s["d1_start_s_ms_ma"]), round(s["mean_ma"] or 0, 2))
                for s in out["steps"]}, "gaps", dec.counter_gaps, "unrecorded_s", round(out["unrecorded_s"], 2), flush=True)


def load(tag):
    return json.loads((OUT / tag / "ppk2" / "energy_steps.json").read_text())


def step_of(rec, name):
    for s in rec["steps"]:
        if s["step"] == name:
            return s
    raise KeyError(f"{rec['tag']}: no step {name}")


def passes(rec, name):
    s = step_of(rec, name)
    p = np.array(s["d0_start_s_ms_ma"], float).reshape(-1, 3)[DROP_FIRST.get(name, 0):]
    ms, ma = p[:, 1], p[:, 2]
    mj = rec["supply_v"] * ma * ms / 1e3
    return ms, ma, mj


def med(x):
    return float(np.median(np.asarray(x, float)))


def tables():
    floor_runs = [json.loads((OUT / t / "ppk2" / "summary.json").read_text())["mean_ma"] for t in RESET_TAGS]
    floor = float(np.mean(floor_runs))
    short = [load(t) for t in SHORT_TAGS]
    volt = short[0]["supply_v"]
    res = {"supply_v": volt, "reset_floor_ma": {"runs": floor_runs, "mean": floor}}

    def idle(rec, name):
        return med(step_of(rec, name)["idle_1s_ma"])

    for name in ("E2", "E4", "E6"):
        runs = [idle(r, name) for r in short]
        res[name] = {"idle_ma_runs": runs, "idle_ma": med(runs), "idle_mw": med(runs) * volt}

    for name in ("E1", "E3"):
        per = []
        for r in short:
            ms, ma, mj = passes(r, name)
            per.append({"n": len(ma), "ms": med(ms), "ma": med(ma), "mj": med(mj)})
        res[name] = {"runs": per, "ms": med([p["ms"] for p in per]), "ma": med([p["ma"] for p in per]),
                     "mj": med([p["mj"] for p in per])}
    e2 = res["E2"]["idle_ma"]
    ms, ma = res["E3"]["ms"], res["E3"]["ma"]
    res["E3"]["above_E2"] = {"ma": ma - e2, "mj": (ma - e2) * volt * ms / 1e3}

    d1 = []
    for r in short:
        p = np.array(step_of(r, "E5")["d1_start_s_ms_ma"], float).reshape(-1, 3)
        d1.append({"n": len(p), "ms": med(p[:, 1]), "ma": med(p[:, 2]), "mj": med(volt * p[:, 1] * p[:, 2] / 1e3),
                   "mj_above_E2": med(volt * p[:, 1] * (p[:, 2] - e2) / 1e3)})
    res["E5"] = {"runs": d1, "ms": med([x["ms"] for x in d1]), "ma": med([x["ma"] for x in d1]),
                 "mj": med([x["mj"] for x in d1]), "mj_above_E2": med([x["mj_above_E2"] for x in d1])}

    cyc_idle = []
    for name, tags in (("E7a", E7A_TAGS), ("E7b", E7B_TAGS)):
        per = []
        for t in tags:
            r = load(t)
            ms, ma, mj = passes(r, name)
            s = step_of(r, name)
            per.append({"tag": t, "n": len(ma), "ms": med(ms), "ma": med(ma), "mj": med(mj),
                        "idle_ma": med(s["idle_1s_ma"]), "mean_ma": s["mean_ma"]})
            cyc_idle.append(per[-1]["idle_ma"])
        res[name] = {"runs": per, "ms": med([p["ms"] for p in per]), "ma": med([p["ma"] for p in per]),
                     "mj": med([p["mj"] for p in per]), "mean_ma": med([p["mean_ma"] for p in per])}
    idle7 = med(cyc_idle)
    res["E7_idle_ma"] = {"runs": cyc_idle, "median": idle7, "mw": idle7 * volt}
    for name in ("E7a", "E7b"):
        e = res[name]
        inc = e["ma"] - idle7
        e["above_idle"] = {"ma": inc, "mj": inc * volt * e["ms"] / 1e3, "share": inc / e["ma"]}
        e["cycle_mw"] = e["mean_ma"] * volt
        e["cycle_j"] = e["mean_ma"] * volt * 300.0 / 1e3
        e["module"] = {"ma": e["ma"] - floor, "mj": (e["ma"] - floor) * volt * e["ms"] / 1e3}
    for name in ("E1", "E3"):
        e = res[name]
        e["module"] = {"ma": e["ma"] - floor, "mj": (e["ma"] - floor) * volt * e["ms"] / 1e3}
    res["E7_idle_ma"]["module_mw"] = (idle7 - floor) * volt
    cyc = med([res["E7a"]["mean_ma"], res["E7b"]["mean_ma"]])
    res["cycle"] = {"mean_ma": cyc, "mw": cyc * volt, "j": cyc * volt * 0.3,
                    "module_ma": cyc - floor, "module_mw": (cyc - floor) * volt, "module_j": (cyc - floor) * volt * 0.3}

    (OUT / "energy_tables.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    f = lambda x, k=1: f"{x:.{k}f}"
    print("reset floor", f(floor, 2), "mA", [f(x, 2) for x in floor_runs])
    for name in ("E1", "E3", "E7a", "E7b"):
        e = res[name]
        print(name, f(e["ma"]), "mA", f(e["mj"]), "mJ", f(e["ms"]), "ms; module", f(e["module"]["ma"]), "mA",
              f(e["module"]["mj"]), "mJ; runs", [(p["n"], f(p["ma"], 2), f(p["mj"], 2)) for p in e["runs"]])
    for name in ("E7a", "E7b"):
        e = res[name]
        print(name, "above idle", f(e["above_idle"]["ma"]), "mA", f(e["above_idle"]["mj"]), "mJ",
              f(100 * e["above_idle"]["share"], 0), "%; step means", [f(p["mean_ma"], 2) for p in e["runs"]],
              "idle", [f(p["idle_ma"], 2) for p in e["runs"]])
    print("E3 above E2", f(res["E3"]["above_E2"]["ma"]), "mA", f(res["E3"]["above_E2"]["mj"]), "mJ")
    for name in ("E2", "E4", "E6"):
        print(name, f(res[name]["idle_ma"], 2), "mA", f(res[name]["idle_mw"], 0), "mW", [f(x, 2) for x in res[name]["idle_ma_runs"]])
    print("E5", f(res["E5"]["ms"], 2), "ms", f(res["E5"]["ma"]), "mA", f(res["E5"]["mj"], 2), "mJ, above E2",
          f(res["E5"]["mj_above_E2"], 2), "mJ", [(x["n"], f(x["ms"], 2), f(x["ma"])) for x in res["E5"]["runs"]])
    print("E7 idle", f(idle7, 2), "mA", f(idle7 * volt, 0), "mW", [f(x, 2) for x in cyc_idle])
    c = res["cycle"]
    print("cycle", f(c["mean_ma"], 2), "mA", f(c["mw"], 0), "mW", f(c["j"]), "J; module", f(c["module_ma"]), "mA",
          f(c["module_mw"], 0), "mW", f(c["module_j"]), "J")


def main():
    ap = argparse.ArgumentParser(description="Energy terms of Table 9 and Table S9b from PPK2 recordings.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="decode raw.bin and assign pulses and 1 s blocks to host steps")
    ex.add_argument("--tag", action="append", required=True)
    sub.add_parser("tables", help="write results/energy/energy_tables.json")
    args = ap.parse_args()
    if args.cmd == "extract":
        for t in args.tag:
            extract(t)
    else:
        tables()
    return 0


if __name__ == "__main__":
    sys.exit(main())
