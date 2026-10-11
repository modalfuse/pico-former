"""R2-13: distribution of host decision margin |logit - tau| for the 16 event outputs on the full held-out set.

Input: results/gelu/gelu_host_logits.npz (host FP64, exact-GELU logits and frozen thresholds).
Outputs: results/margin/r2_13_margin.json, r2_13_margin_table.md, archive/images/margin-cdf.svg.
Superseded by r2_13_margin_five_sets.py; the figure is kept only as a record of the 10-09 discussion
(see the archive note dated 20261011 in archive/).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "margin"
OUT.mkdir(parents=True, exist_ok=True)

EVENTS = ["hypo70", "hypo54", "hyper180", "hyper250"]
HORIZONS = [1, 2, 3, 4]
NAMES = [f"{e}@{h}h" for e in EVENTS for h in HORIZONS]
CUTS = [1e-5, 1e-4, 6.2e-4, 1.84e-3, 1e-2, 1e-1]
RESIDUAL = 1e-5      # order of the residual board-host difference once activations match
TANH_GAP = 1.84e-3   # max host logit difference between tanh and exact GELU

d = np.load(HERE / "results/gelu/gelu_host_logits.npz")
logit, tau = d["ref_cpu64"], d["thresholds"]
m = np.abs(logit - tau[None, :])
n_win = m.shape[0]

rows = []
for k, name in enumerate(NAMES):
    mk = m[:, k]
    rows.append({
        "output": name,
        "positives": int((logit[:, k] >= tau[k]).sum()),
        "min": float(mk.min()),
        "p1": float(np.quantile(mk, 0.01)),
        "median": float(np.median(mk)),
        **{f"lt_{c:g}": int((mk < c).sum()) for c in CUTS},
    })
allm = m.ravel()
summary = {
    "n_windows": n_win, "n_cells": int(allm.size),
    "min": float(allm.min()), "median": float(np.median(allm)),
    **{f"lt_{c:g}": int((allm < c).sum()) for c in CUTS},
    "windows_with_any_lt_1.84e-3": int((m < TANH_GAP).any(1).sum()),
    "windows_with_any_lt_1e-5": int((m < RESIDUAL).any(1).sum()),
}
json.dump({"per_output": rows, "all": summary}, open(OUT / "r2_13_margin.json", "w"), indent=1)

hdr = "| Output | Positive windows | Min margin | 1st percentile | Median | " + " | ".join(f"<{c:g}" for c in CUTS) + " |"
lines = [hdr, "|" + " --- |" * (5 + len(CUTS))]
for r in rows:
    lines.append(f"| {r['output']} | {r['positives']} | {r['min']:.2e} | {r['p1']:.2e} | {r['median']:.2f} | "
                 + " | ".join(str(r[f'lt_{c:g}']) for c in CUTS) + " |")
s = summary
lines.append(f"| Total | — | {s['min']:.2e} | — | {s['median']:.2f} | "
             + " | ".join(str(s[f'lt_{c:g}']) for c in CUTS) + " |")
(OUT / "r2_13_margin_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(json.dumps(summary, indent=1))

plt.rcParams["svg.fonttype"] = "path"
fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), sharey=True)
grid = np.logspace(-6, 1, 300)
for i, e in enumerate(EVENTS):
    ax = axes[i]
    for j, h in enumerate(HORIZONS):
        mk = np.sort(m[:, 4 * i + j])
        ax.loglog(grid, np.searchsorted(mk, grid) / n_win, label=f"{h} h")
    ax.axvline(RESIDUAL, color="k", ls=":", lw=0.8)
    ax.axvline(TANH_GAP, color="k", ls="--", lw=0.8)
    ax.set_title(e, fontsize=9)
    ax.set_xlabel(r"$|\ell-\tau|$", fontsize=9)
    ax.set_ylim(1e-5, 1.2)
axes[0].set_ylabel("Fraction of windows", fontsize=9)
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
img = HERE.parent / "archive" / "images"
img.mkdir(exist_ok=True)
fig.savefig(img / "margin-cdf.svg")
fig.savefig(img / "margin-cdf.pdf")
