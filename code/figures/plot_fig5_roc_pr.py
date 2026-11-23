#!/usr/bin/env python3
"""Plot origin-causal ROC and precision-recall curves for the four 4 h events.

The ROC panels distinguish the validation-frozen event-head operating point,
the retrospective curve readout at the same held-out FPR, and the two curve
cuts selected on the symmetric-resampling validation cache. The retrospective
ROC interpolation does not define an executable threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
RESULTS = CODE / "results"
CURVES = RESULTS / "c5" / "20261113_165152"
METRICS = RESULTS / "c4" / "20261113_165152" / "route_a_metrics.json"
COMPARISON = CURVES / "event_vs_curve.json"
OUT = RESULTS / "figures" / "fig5-roc-pr-event-vs-trajectory.svg"
EVENTS = [
    ("hypo70", "Glucose < 70 mg/dL"),
    ("hypo54", "Glucose < 54 mg/dL"),
    ("hyper180", "Glucose > 180 mg/dL"),
    ("hyper250", "Glucose > 250 mg/dL"),
]

plt.rcParams.update(
    {"svg.fonttype": "path", "pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8}
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.compact:
        plt.rcParams["font.size"] = 7

    comparison = json.loads(COMPARISON.read_text())
    primary = next(item for item in comparison["sets"] if item["set"] == "azt1d_s20260828_test")
    rows = {(row["event"], row["ph_min"]): row for row in primary["outputs"]}
    metrics = json.loads(METRICS.read_text())
    primary_metrics = next(
        item for item in metrics["sets"] if item["set"] == "azt1d_s20260828_test"
    )
    cuts = {
        (row["event"], row["ph_min"]): row
        for row in primary_metrics["s4b_cuts_from_submitted_validation"]
    }

    fig, axes = plt.subplots(
        2, 4, figsize=(7.16, 3.25 if args.compact else 3.9), constrained_layout=True
    )
    for j, (event, title) in enumerate(EVENTS):
        row = rows[(event, 240)]
        selected = cuts[(event, 240)]
        curve = np.load(CURVES / f"roc_pr_{event}_240.npz")

        ax = axes[0, j]
        ax.plot(curve["fpr_head"], curve["tpr_head"], color="#1f4e79", lw=1.1, label="Event head")
        ax.plot(
            curve["fpr_curve"], curve["tpr_curve"], color="#c55a11", lw=1.1, label="Curve readout"
        )
        ax.plot([0, 1], [0, 1], color="0.75", lw=0.6, ls="--")
        head = row["head"]
        ax.plot(
            head["fpr"],
            1 - head["fnr"],
            "o",
            ms=4,
            color="#1f4e79",
            label="Frozen head threshold",
        )
        ax.plot(
            head["fpr"],
            1 - row["curve_fnr_at_head_fpr"],
            "o",
            ms=5,
            mfc="none",
            mec="#c55a11",
            mew=1.0,
            label="Retrospective ROC readout",
        )
        for key, fill, label in (
            ("OP1_curve_fnr_fpr", "none", "Validation FPR-matched cut"),
            ("OP2_curve_fnr_fpr", "#c55a11", "Validation FPR-budget cut"),
        ):
            point = selected[key]
            ax.plot(
                point["fpr"],
                1 - point["fnr"],
                "s",
                ms=4.5,
                mfc=fill,
                mec="#c55a11",
                mew=1.0,
                label=label,
            )
        ax.set_title(
            f"{title}, 4 h\nAUROC {row['auroc_head']:.3f} / {row['auroc_curve']:.3f}",
            fontsize=7 if args.compact else 7.5,
        )
        ax.set_xlabel("False positive rate")
        if j == 0:
            ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax = axes[1, j]
        ax.plot(curve["recall_head"], curve["precision_head"], color="#1f4e79", lw=1.1)
        ax.plot(curve["recall_curve"], curve["precision_curve"], color="#c55a11", lw=1.1)
        auprc_head = -np.trapezoid(curve["precision_head"], curve["recall_head"])
        auprc_curve = -np.trapezoid(curve["precision_curve"], curve["recall_curve"])
        prevalence = (head["tp"] + head["fn"]) / (
            head["tp"] + head["fn"] + head["fp"] + head["tn"]
        )
        ax.axhline(prevalence, color="0.75", lw=0.6, ls="--")
        ax.set_title(
            f"AUPRC {auprc_head:.3f} / {auprc_curve:.3f}",
            fontsize=7 if args.compact else 7.5,
        )
        ax.set_xlabel("Recall")
        if j == 0:
            ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        fontsize=6.2 if args.compact else 7,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(args.out)


if __name__ == "__main__":
    main()
