#!/usr/bin/env python3
"""Fig. 4: latency breakdown (A) and internal SRAM budget (B) of the revised firmware.

A: per-stage latency of inferences 2-20 after power-up on the golden window (L1, Wi-Fi off, median over
   5 power-ups, WiFiduino); the dashed line is the median latency over the 19,152 AZT1D held-out windows.
   Also prints the values of both boards needed for Table 7.
B: boot log and per-window records of the full held-out set replay of AZT1D seed 20260828 on WiFiduino,
   and section sizes of the frozen image.
Data: code/results/latency/{lmodes.json, devkitc/lmodes.json}, code/results/audit/azt1d_full/audit.json,
      code/results/r2_8_mem/r2_8_mem.json. Output: figures/fig4-latency-memory-breakdown.{svg,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent
RES = CODE / "results"
OUT = ROOT / "figures" / "fig4-latency-memory-breakdown"

plt.rcParams.update({"svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 8})


def warm_split(path: Path, mode: str = "1") -> dict:
    d = json.loads(path.read_text())["modes"][mode]
    rows = [r for run in d["runs"] for r in run["lrun"] if r["i"] >= 1]
    first = [run["lrun"][0] for run in d["runs"]]
    med = lambda rs, k: float(np.median([r[k] for r in rs])) / 1000  # noqa: E731
    out = {k: med(rows, f"us_{k}") for k in ("total", "enc", "cross", "heads")}
    out["glue"] = out["total"] - out["enc"] - out["cross"] - out["heads"]
    out["first"] = med(first, "us_total")
    out["n"] = len(rows)
    return out


def main() -> None:
    a = warm_split(RES / "latency/lmodes.json")
    b = warm_split(RES / "latency/devkitc/lmodes.json")
    wifi_first = warm_split(RES / "latency/lmodes.json", "2")["first"]
    hold = json.loads((RES / "audit/azt1d_full/audit.json").read_text())["latency_ms"]["median"]
    mem = json.loads((RES / "r2_8_mem/r2_8_mem.json").read_text())
    run = mem["runs"][0]
    assert run["dir"] == "azt1d_full"
    h = run["heap_stages"]
    sec = mem["sections"]["10cba768 (replays, seed 20260828)"]

    print("Table 7 (WiFiduino / DevKitC-1), ms:")
    for k in ("enc", "cross", "heads", "glue", "total", "first"):
        print(f"  {k:6s} {a[k]:.2f} / {b[k]:.2f}")
    print(f"  first pass with Wi-Fi association (WiFiduino): {wifi_first:.2f}; passes per board: {a['n']}, {b['n']}")
    print(f"  holdout median: {hold:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.32})

    ax = axes[0]
    parts = [("Encoder (2 layers)", a["enc"], "#1F77B4"), ("Cross-attention (curve + event)", a["cross"], "#E67E22"),
             ("Heads", a["heads"], "#27AE60"), ("Other", a["glue"], "#7F8C8D")]
    left = 0.0
    for name, w, c in parts:
        ax.barh(0, w, left=left, height=0.5, color=c, edgecolor="white", label=f"{name}: {w:.1f}")
        if w > 30:
            ax.text(left + w / 2, 0, f"{w:.1f}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        left += w
    ax.vlines(hold, -0.38, 0.62, color="#333333", ls="--", lw=0.9)
    ax.text(hold - 4, 0.33, f"holdout median {hold:.1f}", fontsize=7, ha="right", va="bottom")
    ax.set_xlim(0, 420)
    ax.set_ylim(-0.45, 1.25)
    ax.set_yticks([])
    ax.set_xlabel("Golden-window pass (ms)")
    ax.set_title(f"A  Latency split (total {a['total']:.1f} ms)", loc="left", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=6.5, ncol=2, handlelength=1.0, columnspacing=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    kb = 1024.0
    used_run = h["after_web"]["int_free"] - run["mem"]["int_min_min"]
    heap_parts = [
        ("Working buffers", run["init_drop_int"], "#1F77B4"),
        ("Inference task", h["after_init"]["int_free"] - h["before_wifi"]["int_free"], "#5DADE2"),
        ("Wi-Fi, HTTP, HTTPS", run["net_drop_int"], "#E67E22"),
        ("Used during replay", used_run, "#F5CBA7"),
        ("Minimum free", run["mem"]["int_min_min"], "#D5F5E3"),
    ]
    link_parts = [("IRAM code", sec["iram_code"], "#8E44AD"), ("Static RAM", sec["static_ram"], "#7F8C8D")]
    for y, parts_, label in ((1, link_parts, "Linker"), (0, heap_parts, "Heap at boot")):
        left = 0.0
        for name, w, c in parts_:
            ax.barh(y, w / kb, left=left, height=0.55, color=c, edgecolor="white")
            txt = f"{w / kb:.1f}"
            if w / kb > 22:
                ax.text(left + w / kb / 2, y, txt, ha="center", va="center", fontsize=7,
                        color="white" if c in ("#1F77B4", "#8E44AD", "#7F8C8D", "#E67E22") else "#333333")
            left += w / kb
    ax.set_yticks([0, 1], [f"Heap at boot\n({h['before_init']['int_free'] / kb:.1f})", "Linker"])
    ax.set_xlim(0, 340)
    ax.set_xlabel("Internal SRAM (KB)")
    ax.set_title("B  Internal SRAM account", loc="left", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in link_parts + heap_parts]
    names = [n for n, _, _ in link_parts + heap_parts]
    ax.legend(handles, names, loc="upper right", frameon=False, fontsize=6.3, ncol=2, handlelength=1.0,
              columnspacing=0.8, bbox_to_anchor=(1.0, 1.02))
    ax.set_ylim(-0.5, 2.1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print("Fig. 4B (B):", {n: w for n, w, _ in link_parts + heap_parts})


if __name__ == "__main__":
    main()
