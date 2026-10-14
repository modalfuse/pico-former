"""R2-10, R2-18 link-drop test: request counts and recovery time from the request log (Supplementary Table S7d).

Per outage: requests started between the drop command and the end of the outage and how many failed, and the
earliest completion (start + latency) of a successful request started after the outage ended, relative to that end.
Over the run: total, successful and failed requests, and failed requests outside the outages.

Input: code/results/wireless/linkdrop/{requests.csv, summary.json} (run_linkdrop.py).
Output: code/results/wireless/linkdrop/requests_summary.json
"""

import csv
import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results" / "wireless" / "linkdrop"


def main():
    rows = [(float(r["t_s"]), float(r["latency_ms"]) / 1000, r["outcome"] == "ok")
            for r in csv.DictReader((RES / "requests.csv").open())]
    events = json.loads((RES / "summary.json").read_text())["events"]
    out = {"requests": len(rows), "ok": sum(ok for _, _, ok in rows), "failed": sum(not ok for _, _, ok in rows),
           "events": []}
    for e in events:
        c, h = e["cmd_t_s"], e["hold_end_t_s"]
        during = [ok for t, _, ok in rows if c <= t < h]
        recovered = min(t + lat for t, lat, ok in rows if ok and t >= h) - h
        out["events"].append({"event": e["event"], "requests_during_outage": len(during),
                              "failed_during_outage": sum(not ok for ok in during),
                              "first_success_completed_after_outage_s": round(recovered, 3)})
    out["failed_outside_outages"] = out["failed"] - sum(ev["failed_during_outage"] for ev in out["events"])
    (RES / "requests_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
