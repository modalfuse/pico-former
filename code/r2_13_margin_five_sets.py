#!/usr/bin/env python3
"""R2-13: host decision margin distributions of the 16 event outputs on five full sets
(audit float64 reference, frozen thresholds).

Margin m = |host float64 logit - tau|, from ref_logits in code/results/audit/<set>/audit.npz, the same reference
as Table 4 and Table S5a; the max board-host logit difference per output is also reported, to compare margins
with measured differences.
Outputs: code/results/margin/r2_13_margin_five_sets.json; supplementary figure
figures/figs-margin-cdf.{svg,png} (16 outputs in 4 panels by event, five full sets pooled).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE.parent / "figures" / "figs-margin-cdf"
SETS = [("AZT1D, deployed model", "azt1d_full"), ("AZT1D, second model", "s20260829_full"),
        ("AZT1D, third model", "s20260830_full"), ("OhioT1DM", "ohio_full"), ("DiaTrend", "devkitc_diatrend17_full")]
EVT = ("hypo70", "hypo54", "hyper180", "hyper250")
NAMES = [f"{e} {h} h" for e in EVT for h in (1, 2, 3, 4)]
CUTS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)

plt.rcParams.update({"svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 8})


def load(tag: str):
    d = np.load(RES / "audit" / tag / "audit.npz")
    ok = d["status"] == 0
    ref = d["ref_logits"][ok]
    thr = d["thr"].astype(np.float64)
    m = np.abs(ref - thr)
    e = np.abs(d["logits"][ok].astype(np.float64) - ref)
    return m, e, (ref >= thr)


def summarize(m: np.ndarray, e: np.ndarray, pos: np.ndarray) -> dict:
    return {"cells": int(m.size), "positive": int(pos.sum()), "min": float(m.min()), "median": float(np.median(m)),
            **{f"lt_{c:g}": int((m < c).sum()) for c in CUTS}, "max_diff": float(e.max()),
            "lt_max_diff": int((m < e.max()).sum())}


def main() -> None:
    per_set, ms, es, ps = {}, [], [], []
    for name, tag in SETS:
        m, e, p = load(tag)
        per_set[name] = {"all": summarize(m, e, p),
                         "outputs": {NAMES[j]: summarize(m[:, j], e[:, j], p[:, j]) for j in range(16)}}
        ms.append(m), es.append(e), ps.append(p)
    M, E, P = np.concatenate(ms), np.concatenate(es), np.concatenate(ps)
    pooled = {"all": summarize(M, E, P), "outputs": {NAMES[j]: summarize(M[:, j], E[:, j], P[:, j]) for j in range(16)}}
    res = {"sets": per_set, "pooled": pooled}
    (RES / "margin").mkdir(exist_ok=True)
    (RES / "margin/r2_13_margin_five_sets.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    hdr = "output         pos      min      median  " + " ".join(f"<{c:g}" for c in CUTS) + "  maxdiff  <maxdiff"
    print(hdr)
    for k, v in list(pooled["outputs"].items()) + [("ALL", pooled["all"])]:
        print(f"{k:13s} {v['positive']:7d} {v['min']:.2e} {v['median']:6.2f}  " +
              " ".join(f"{v[f'lt_{c:g}']:5d}" for c in CUTS) + f"  {v['max_diff']:.2e} {v['lt_max_diff']:4d}")
    for name, v in per_set.items():
        a = v["all"]
        print(f"{name:24s} cells={a['cells']} min={a['min']:.2e} median={a['median']:.2f} "
              f"<1e-5={a['lt_1e-05']} <1e-4={a['lt_0.0001']} <1e-3={a['lt_0.001']} maxdiff={a['max_diff']:.2e} "
              f"<maxdiff={a['lt_max_diff']}")

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.1), sharey=True)
    emax = float(E.max())
    colors = ("#1F77B4", "#E67E22", "#27AE60", "#8E44AD")
    for k, ax in enumerate(axes):
        for h in range(4):
            x = np.sort(M[:, 4 * k + h])
            ax.plot(np.maximum(x, 1e-8), np.arange(1, len(x) + 1) / len(x), color=colors[h], lw=1.0,
                    label=f"{h + 1} h")
        ax.axvline(emax, color="#C0392B", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-8, 10)
        ax.set_ylim(1e-7, 1)
        ax.set_title(EVT[k], fontsize=8)
        ax.set_xlabel("Host margin |ℓ − τ|")
        ax.grid(alpha=0.25, which="major")
    axes[0].set_ylabel("Fraction of cells ≤ margin")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(FIG.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"largest board–host difference over five sets: {emax:.2e}")


if __name__ == "__main__":
    main()
