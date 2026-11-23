#!/usr/bin/env python3
"""Insert Route-A result tables into the revision supplementary Markdown."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SET_LABELS = {
    "azt1d_s20260828_test": "AZT1D, deployed checkpoint",
    "azt1d_s20260829_test": "AZT1D, second checkpoint",
    "azt1d_s20260830_test": "AZT1D, third checkpoint",
    "ohiot1dm": "OhioT1DM",
    "diatrend": "DiaTrend",
}


def extract_table(text: str, name: str, next_name: str | None) -> str:
    start = text.index(f"**{name}.")
    end = text.index(f"**{next_name}.", start) if next_name else len(text)
    return text[start:end].strip()


def replace_section(text: str, heading: str, next_heading: str, body: str) -> str:
    pattern = rf"(?ms)^{re.escape(heading)}\n.*?(?=^{re.escape(next_heading)}\n)"
    replacement = f"{heading}\n\n{body}\n\n"
    updated, count = re.subn(pattern, lambda _: replacement, text)
    if count != 1:
        raise RuntimeError(f"Expected one section {heading!r}, found {count}")
    return updated


def per_participant_section(primary: dict) -> str:
    lines = [
        (
            "**Table S2. Per-participant Route-A results, deployed checkpoint, "
            "AZT1D origin-causal holdout.** MAE is in mg/dL; FNR/FPR use the "
            "R1-submitted frozen 4 h hypo70 threshold."
        ),
        "",
        "| Participant | Windows | MAE 0–1 / 0–4 h | 4 h hypo70 positive | FNR / FPR |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary["per_participant"]:
        metrics = row["hypo70_4h"]
        lines.append(
            f"| {row['subject_id']} | {row['windows']:,} | "
            f"{row['mae_0_to_h']['60']:.2f} / {row['mae_0_to_h']['240']:.2f} | "
            f"{metrics['n_pos']:,} | {metrics['fnr']:.3f} / {metrics['fpr']:.3f} |"
        )
    return "\n".join(lines)


def leave_one_out_section(primary: dict) -> str:
    lines = [
        (
            "**Table S3. Route-A aggregate 4 h hypo70 results after removing one "
            "primary-holdout participant.** The released checkpoint and "
            "R1-submitted frozen threshold are unchanged."
        ),
        "",
        "| Participant removed | Remaining positive windows | FNR | FPR |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in primary["leave_one_out_hypo70_4h"]:
        lines.append(
            f"| {row['left_out']} | {row['n_pos']:,} | "
            f"{row['fnr']:.3f} | {row['fpr']:.3f} |"
        )
    return "\n".join(lines)


def route_a_partition_table(result: dict) -> str:
    lines = [
        (
            "**Table S11c. Origin-causal Route-A test-window counts and 4 h event "
            "prevalence.** These test windows were reconstructed for fixed-model "
            "host evaluation. The R1 training and validation counts in Table S11a "
            "remain the data actually used for model fitting and threshold selection."
        ),
        "",
        "| Set | Participants | Windows | hypo70 (%) | hypo54 (%) | hyper180 (%) | hyper250 (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in result["sets"]:
        by_event = {
            row["event"]: row for row in dataset["outputs"] if row["ph_min"] == 240
        }
        values = [
            100.0 * by_event[event]["n_pos"] / dataset["windows"]
            for event in ("hypo70", "hypo54", "hyper180", "hyper250")
        ]
        lines.append(
            f"| {SET_LABELS[dataset['set']]} | {dataset['participants']} | "
            f"{dataset['windows']:,} | "
            + " | ".join(f"{value:.1f}" for value in values)
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("supplement_md", type=Path)
    parser.add_argument("generated_tables_md", type=Path)
    parser.add_argument("result_json", type=Path)
    args = parser.parse_args()

    supplement = args.supplement_md.read_text(encoding="utf-8")
    generated = args.generated_tables_md.read_text(encoding="utf-8")
    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    primary = result["sets"][0]

    new_s1 = extract_table(generated, "Table S1", "Table S4a")
    new_s4a = extract_table(generated, "Table S4a", "Table S4b")
    new_s4b = extract_table(generated, "Table S4b", "Table S8a")
    new_s8 = extract_table(generated, "Table S8a", None)

    old_s1_body = re.search(
        r"(?ms)^## S1\..*?\n\n(.*?)(?=^## S2\.)", supplement
    ).group(1).strip()
    old_s1_body = old_s1_body.replace("**Table S1.", "**Table S1b.", 1)
    old_s1_body = old_s1_body.replace(
        "Event outputs at frozen",
        "R1-submitted-window event outputs at frozen",
        1,
    )
    s1_body = (
        f"{new_s1}\n\n"
        "The Route-A evaluation fixes the released checkpoint, R1 training-set "
        "normalization statistics, and R1-submitted thresholds. Training and "
        "threshold selection used the original symmetric-resampling workflow.\n\n"
        f"{old_s1_body}"
    )
    supplement = replace_section(
        supplement,
        "## S1. All sixteen event outputs, deployed model, AZT1D holdout",
        "## S2. Per-participant results",
        s1_body,
    )

    supplement = replace_section(
        supplement,
        "## S2. Per-participant results",
        "## S3. Leave-one-participant-out",
        per_participant_section(primary),
    )
    supplement = replace_section(
        supplement,
        "## S3. Leave-one-participant-out",
        "## S4. Event head versus thresholding the curve",
        leave_one_out_section(primary),
    )

    old_s4_body = re.search(
        r"(?ms)^## S4\..*?\n\n(.*?)(?=^## S5\.)", supplement
    ).group(1).strip()
    old_s4_body = old_s4_body.replace("**Table S4a.", "**Table S4c.", 1)
    old_s4_body = old_s4_body.replace("**Table S4b.", "**Table S4d.", 1)
    old_s4_body = old_s4_body.replace(
        "Held-out AUROC",
        "R1-submitted-window AUROC",
        1,
    )
    old_s4_body = old_s4_body.replace(
        "Validation-selected operating points",
        "R1-submitted-window validation-selected operating points",
        1,
    )
    supplement = replace_section(
        supplement,
        "## S4. Event head versus thresholding the curve",
        "## S5. Decision margins in the consistency audit",
        f"{new_s4a}\n\n{new_s4b}\n\n{old_s4_body}",
    )

    old_s8_body = re.search(
        r"(?ms)^## S8\..*?\n\n(.*?)(?=^## S9\.)", supplement
    ).group(1).strip()
    old_s8_body = old_s8_body.replace("**Table S8.", "**Table S8f.", 1)
    old_s8_body = old_s8_body.replace(
        "FNR / FPR / AUROC at frozen",
        "R1-submitted-window FNR / FPR / AUROC at frozen",
        1,
    )
    supplement = replace_section(
        supplement,
        "## S8. Sixteen outputs for the further models and external cohorts",
        "## S9. Energy measurement modes and further terms",
        f"{new_s8}\n\n{old_s8_body}",
    )

    marker = "\n## S12. Memory placement, heap, and stack"
    if marker not in supplement:
        raise RuntimeError("S12 marker not found")
    supplement = supplement.replace(
        marker,
        "\n\n" + route_a_partition_table(result) + marker,
        1,
    )

    args.supplement_md.write_text(supplement, encoding="utf-8")


if __name__ == "__main__":
    main()
