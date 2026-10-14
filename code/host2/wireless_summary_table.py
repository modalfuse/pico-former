"""R2-8, R2-10: one row per wireless test with the same quantities (Supplementary Table S7e).

Per test: requests and failed requests; latency of successful requests (p50, p95, p99, max); response body sizes;
longest handler time; pass time (median, p99, max) and serial retransmissions of the replayed windows; and the
lowest values of the minimum free internal SRAM, the largest free internal block, and the unused stack of the
inference, HTTP, and HTTPS tasks. Memory and stack minima are taken over both the node status polls
(status.jsonl, every 10 s; every 2 s in the link-drop test) and the per-window result frames (windows.npz), whichever is lower.
The lowest free PSRAM, which holds the Wi-Fi, lwIP, and TLS buffers, comes from the status polls only; it is
also given as the drop below the free PSRAM of the unloaded replay with all services running.

Input: code/results/wireless/<tag>/{summary.json, status.jsonl, windows.npz, requests.csv} (run_wireless.py,
run_linkdrop.py); request statistics come from summary.json and, where it has none, from requests.csv.
The unloaded reference row comes from code/results/audit/azt1d_full/audit.json.
Output: code/results/wireless/s7e.json
"""

import csv
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"
TESTS = [
    ("t2_c1", "HTTP, 1 client, no inference"),
    ("t2_c4", "HTTP, 4 clients, no inference"),
    ("t2_c8", "HTTP, 8 clients, no inference"),
    ("t3_c4", "HTTP, 4 clients, AZT1D holdout"),
    ("https_keep_c2", "HTTPS, 2 persistent clients"),
    ("https_c1_5s", "HTTPS, full handshake every 5 s"),
    ("https_load_c2", "HTTPS, about 2 handshakes per second"),
    ("linkdrop", "HTTP, 2 clients, three 60 s link outages"),
]


def f1(x):
    """One decimal, half up, as in Tables 8 and S7a-S7d."""
    return f"{Decimal(str(x)).quantize(Decimal('0.1'), ROUND_HALF_UP):,}"


def pct(a, q):
    return float(np.percentile(a, q))


def request_stats(d, summary):
    req = summary.get("requests", {}).get("/api/last")
    if req:
        lat = req["latency_ms"]
        return {"n": req["n"], "failed": req["fail"], "p50": lat["p50"], "p95": lat["p95"], "p99": lat["p99"],
                "max": lat["max"], "bytes": [min(req["bytes"]), max(req["bytes"])]}
    rows = list(csv.DictReader((d / "requests.csv").open()))
    ok = np.array([float(r["latency_ms"]) for r in rows if r["outcome"] == "ok"])
    sizes = [int(r["bytes"]) for r in rows if r["outcome"] == "ok"]
    return {"n": len(rows), "failed": len(rows) - len(ok), "p50": pct(ok, 50), "p95": pct(ok, 95),
            "p99": pct(ok, 99), "max": float(ok.max()), "bytes": [min(sizes), max(sizes)]}


def positive_min(values):
    v = [x for x in values if x > 0]
    return min(v) if v else None


def row(tag, label, psram_ref):
    d = RES / "wireless" / tag
    summary = json.loads((d / "summary.json").read_text())
    polls = [json.loads(line) for line in (d / "status.jsonl").open()]
    polls = [p for p in polls if "int_min" in p]
    out = {"tag": tag, "test": label, "requests": request_stats(d, summary), "status_polls": len(polls),
           "handler_ms_max": max(p["handler_us_max"] for p in polls) / 1000,
           "reconnects": max(p["reconnects"] for p in polls)}
    int_min = [p["int_min"] for p in polls]
    largest = [p["int_largest"] for p in polls]
    stack_infer = [p["stack_infer"] for p in polls]
    wpath = d / "windows.npz"
    if wpath.exists():
        z = np.load(wpath)
        n = int(z["done"])
        us = z["us_total"][:n] / 1000.0
        resend = z["lost"] if "lost" in z else z["resend"]
        out["windows"] = {"n": n, "median": float(np.median(us)), "p99": pct(us, 99), "max": float(us.max()),
                          "retransmissions": int(resend[:n].sum()),
                          "windows_retransmitted": int((resend[:n] > 0).sum())}
        int_min += z["int_min"][:n].tolist()
        if "int_largest" in z:
            largest += z["int_largest"][:n].tolist()
        if "stack_infer" in z:
            stack_infer += z["stack_infer"][:n].tolist()
    psram_min = min(p["spiram_free"] for p in polls)
    out["memory"] = {"int_min": int(min(int_min)), "int_largest_min": int(min(largest)),
                     "psram_free_min": int(psram_min), "psram_below_unloaded": int(psram_ref - psram_min),
                     "stack_infer_min": int(min(stack_infer)),
                     "http_stack_min": positive_min(p["http_stack_hw"] for p in polls),
                     "https_stack_min": positive_min(p["https_stack_hw"] for p in polls)}
    return out


def reference():
    a = json.loads((RES / "audit" / "azt1d_full" / "audit.json").read_text())
    return {"tag": "azt1d_full", "test": "Unloaded replay, AZT1D holdout", "audit": a}


def main():
    ref = reference()
    psram_ref = ref["audit"]["mem"]["spiram_free_min"]
    rows = [row(t, label, psram_ref) for t, label in TESTS]
    (RES / "wireless" / "s7e.json").write_text(json.dumps({"rows": rows, "reference": ref}, indent=1))
    print(f"unloaded replay: min. free PSRAM {psram_ref:,} B")
    for r in rows:
        q, m, w = r["requests"], r["memory"], r.get("windows")
        pt = f"{f1(w['median'])} / {f1(w['p99'])} / {f1(w['max'])} | {w['retransmissions']}" if w else "— | —"
        stacks = " / ".join(f"{v:,}" if v else "—" for v in (m["stack_infer_min"], m["http_stack_min"], m["https_stack_min"]))
        print(f"| {r['test']} | {q['n']:,} ({q['failed']:,}) | {f1(q['p50'])} / {f1(q['p95'])} / {f1(q['p99'])} / "
              f"{f1(q['max'])} | {q['bytes'][0]}–{q['bytes'][1]} | {f1(r['handler_ms_max'])} | {pt} | "
              f"{m['int_min']:,} | {m['int_largest_min']:,} | {m['psram_free_min']:,} ({m['psram_below_unloaded']:,}) | "
              f"{stacks} |")


if __name__ == "__main__":
    main()
