#!/usr/bin/env python3
"""R2-15 design-space figure: activation memory vs held-out error and event miss rate; a latency panel is added when board latency results exist.

Input: code/results/design_space/grid_summary.json; optional latency_board.json (run_latency_bench.py).
Output: design-space.svg and design-space.pdf in IMG (code/results/figures/);
supplementary Fig. S2 figures/figs-design-space.{svg,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "design_space"
IMG = HERE.parent / "results" / "figures"
FIG = HERE.parents[1] / "figures"
SRAM_KB = 225860 / 1024  # revised firmware with Wi-Fi, HTTP and HTTPS runtime: seven buffers 196,720 B + minimum free on-chip memory during replay 29,140 B, measured for R2-8

plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["font.size"] = 8

COLOR = {32: "#4C72B0", 48: "#DD8452", 64: "#55A868", 96: "#8C8C8C"}
MARK = {1: "o", 2: "s", 3: "^"}
GROUP_TOK = {"L48", "L60", "L84"}
GROUP_HEAD = {"h2", "h8"}
LABEL = {"base": "deployed", "L48": "L=48", "L60": "L=60", "L84": "L=84", "h2": "2h", "h8": "8h"}
# label positions in the first panel (activation memory KB, 1 h MAE), placed in empty space clear of points and error bars
LABEL_AT = {"L48": (100, 13.2), "L60": (128, 13.05), "L84": (150, 15.15), "base": (205, 13.05),
            "h2": (170, 13.2), "h8": (192, 15.0)}
# horizontal offsets (KB) for configurations with the same activation memory, to avoid overlap
JITTER = {1: -7.0, 2: 0.0, 3: 7.0, "h2": -14.0, "h8": 14.0}


def mem_x(r):
    j = JITTER.get(r["config"], JITTER[r["layers"]] if r["config"] not in GROUP_TOK else 0.0)
    return r["act_bytes"] / 1024 + j


def style(r):
    if r["config"] in GROUP_TOK:
        return dict(color="#C44E52", marker="D", mfc="white")
    if r["config"] in GROUP_HEAD:
        return dict(color="#8172B2", marker="P", mfc="white")
    return dict(color=COLOR[r["d"]], marker=MARK[r["layers"]], mfc=COLOR[r["d"]])


def scatter(ax, rows, x_of, key, ylabel, labels=True):
    for r in rows:
        x = x_of(r)
        if x is None:
            continue
        st = style(r)
        ax.errorbar(x, r[key + "_mean"], yerr=r[key + "_sd"], fmt=st["marker"], color=st["color"], mfc=st["mfc"],
                    ms=5, capsize=2, lw=0.8, mew=1.0)
        if labels and r["config"] in LABEL:
            ax.annotate(LABEL[r["config"]], (x, r[key + "_mean"]), xytext=LABEL_AT[r["config"]], textcoords="data",
                        fontsize=6.5, ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color="0.45", lw=0.5, shrinkA=1.5, shrinkB=3))
    ax.set_ylabel(ylabel)


def main():
    rows = json.loads((RES / "grid_summary.json").read_text())["rows"]
    lat_path = RES / "latency_board.json"
    lat = json.loads(lat_path.read_text()) if lat_path.exists() else {}
    lat = {k: v["total"] / 1000 for k, v in lat.items() if v.get("done") == "ok" and "total" in v}
    panels = [("traj_mae_1h", "1 h MAE (mg/dL)"), ("traj_mae_4h", "4 h MAE (mg/dL)"),
              ("hypo70_4h_fnr", "4 h hypo70 FNR")]
    ncol = len(panels) + (1 if lat else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(2.35 * ncol, 2.4), constrained_layout=True)
    for i, (ax, (key, yl)) in enumerate(zip(axes, panels)):
        scatter(ax, rows, mem_x, key, yl, labels=i == 0)
        ax.axvline(SRAM_KB, color="k", ls="--", lw=0.8)
        ax.set_xlabel("Activation memory (KB)")
    y0, y1 = axes[0].get_ylim()
    axes[0].text(SRAM_KB + 4, y0 + 0.97 * (y1 - y0), "internal\nSRAM\navailable", ha="left", va="top", fontsize=6.5)
    if lat:
        scatter(axes[-1], rows, lambda r: lat.get(r["config"]), "traj_mae_1h", "1 h MAE (mg/dL)", labels=False)
        axes[-1].set_xlabel("Board latency per window (ms)")
    handles = [plt.Line2D([], [], color=c, marker="s", ls="", ms=5, label=f"d={d}") for d, c in COLOR.items()]
    handles += [plt.Line2D([], [], color="k", marker=m, ls="", ms=5, mfc="white", label=f"{n} layer" + "s" * (n > 1))
                for n, m in MARK.items()]
    handles += [plt.Line2D([], [], color="#C44E52", marker="D", ls="", ms=5, mfc="white", label="token count"),
                plt.Line2D([], [], color="#8172B2", marker="P", ls="", ms=5, mfc="white", label="head count")]
    fig.legend(handles=handles, loc="outside lower center", ncol=9, fontsize=6.5, frameon=False)
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / "design-space.svg")
    fig.savefig(IMG / "design-space.pdf")
    fig.savefig(FIG / "figs-design-space.svg")
    fig.savefig(FIG / "figs-design-space.png", dpi=300)
    print(IMG / "design-space.svg", "latency panel" if lat else "no latency panel")


if __name__ == "__main__":
    main()
