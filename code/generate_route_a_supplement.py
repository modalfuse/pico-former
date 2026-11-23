#!/usr/bin/env python3
"""Generate Markdown tables for the origin-causal fixed-model results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SET_LABELS = {
    "azt1d_s20260828_test": "AZT1D, deployed checkpoint",
    "azt1d_s20260829_test": "AZT1D, second checkpoint",
    "azt1d_s20260830_test": "AZT1D, third checkpoint",
    "ohiot1dm": "OhioT1DM",
    "diatrend": "DiaTrend",
}


def fmt(value: float) -> str:
    return f"{value:.3f}"


def interval(values: list[float] | None) -> str:
    if values is None:
        return "—"
    return f"{values[0]:.3f}–{values[2]:.3f}"


def primary_metrics_table(result: dict) -> str:
    primary = result["sets"][0]
    lines = [
        (
            f"**Table S1. Origin-causal event outputs at the "
            f"validation-frozen thresholds on {primary['windows']:,} windows "
            f"from {primary['participants']} participants.**"
        ),
        "",
        "| Event | Horizon (h) | Positive / total | Sens. | Spec. | PPV | NPV | FNR | FPR | AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary["outputs"]:
        head = row["head"]
        lines.append(
            f"| {row['event']} | {row['ph_min'] // 60} | "
            f"{row['n_pos']:,} / {row['n']:,} | {fmt(head['sensitivity'])} | "
            f"{fmt(head['specificity'])} | {fmt(head['ppv'])} | "
            f"{fmt(head['npv'])} | {fmt(head['fnr'])} | {fmt(head['fpr'])} | "
            f"{fmt(row['auroc_head'])} |"
        )
    return "\n".join(lines)


def comparison_table(title: str, dataset: dict, table_name: str) -> str:
    lines = [
        (
            f"**{table_name}. {title}: event head versus curve readout on "
            f"{dataset['windows']:,} origin-causal windows "
            f"({dataset['participants']} participants).** AUROC difference is "
            "event head minus curve readout; FNR difference is curve readout "
            "minus event head. Curve FNR is read retrospectively from the "
            "held-out ROC at the event head's realized FPR. Intervals use paired "
            "participant-clustered bootstrap resampling."
        ),
        "",
        "| Event | H (h) | Pos. | Head / curve AUROC | AUROC diff. 95% interval | "
        "Head FNR / FPR | Retrospective curve FNR | FNR diff. 95% interval | Valid resamples |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dataset["outputs"]:
        head = row["head"]
        boot = row["boot"]
        lines.append(
            f"| {row['event']} | {row['ph_min'] // 60} | {row['n_pos']:,} | "
            f"{fmt(row['auroc_head'])} / {fmt(row['auroc_curve'])} | "
            f"{interval(boot['auroc_head_minus_curve'])} | "
            f"{fmt(head['fnr'])} / {fmt(head['fpr'])} | "
            f"{fmt(row['curve_fnr_at_head_fpr'])} | "
            f"{interval(boot['fnr_curve_minus_head'])} | {boot['n_valid']:,} |"
        )
    return "\n".join(lines)


def validation_cut_table(result: dict) -> str:
    primary = result["sets"][0]
    rows = primary["s4b_cuts_from_submitted_validation"]
    lines = [
        (
            "**Table S4b. Clinical-level reference and validation-selected curve "
            "cuts evaluated on origin-causal test windows.** OP0 applies the "
            "clinical glucose level directly. OP1 and OP2 were selected on the "
            "symmetric-resampling validation cache; no threshold was re-selected on "
            "causal validation windows."
        ),
        "",
        "| Event | H (h) | Validation head FPR | OP0 FNR / FPR | "
        "OP1 cut (mg/dL) | OP1 FNR / FPR | OP2 cut (mg/dL) | OP2 FNR / FPR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        op0 = row["OP0_curve_fnr_fpr"]
        op1 = row["OP1_curve_fnr_fpr"]
        op2 = row["OP2_curve_fnr_fpr"]
        lines.append(
            f"| {row['event']} | {row['ph_min'] // 60} | "
            f"{fmt(row['val_head_fpr'])} | {fmt(op0['fnr'])} / {fmt(op0['fpr'])} | "
            f"{row['OP1_cut_mgdl']:.2f} | {fmt(op1['fnr'])} / {fmt(op1['fpr'])} | "
            f"{row['OP2_cut_mgdl']:.2f} | {fmt(op2['fnr'])} / {fmt(op2['fpr'])} |"
        )
    return "\n".join(lines)


def generate(result: dict) -> str:
    sections = [primary_metrics_table(result)]
    sections.append(
        comparison_table(
            "Primary AZT1D holdout",
            result["sets"][0],
            "Table S4a",
        )
    )
    sections.append(validation_cut_table(result))
    for index, dataset in enumerate(result["sets"], start=1):
        suffix = chr(ord("a") + index - 1)
        sections.append(
            comparison_table(
                SET_LABELS.get(dataset["set"], dataset["set"]),
                dataset,
                f"Table S8{suffix}",
            )
        )
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_md", type=Path)
    args = parser.parse_args()
    result = json.loads(args.input_json.read_text(encoding="utf-8"))
    args.output_md.write_text(generate(result), encoding="utf-8")


if __name__ == "__main__":
    main()
