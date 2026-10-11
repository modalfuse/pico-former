"""Recompute RICE-Former on the full AZT1D holdout set.

Uses the same ckpt as Pico-Former firmware and the official release metrics.
Baseline rows in the figures come from the archived paper CSVs; RICE rows are
computed in this run.
"""

from __future__ import annotations

import json
import pickle
import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RELEASE_SRC = Path(os.environ.get("RICE_FORMER_SRC", "")).expanduser()
if not str(RELEASE_SRC):
    raise SystemExit("Set RICE_FORMER_SRC to the rice-former src directory (the folder that contains ceqt/).")
CKPT = Path(os.environ.get("RICE_CKPT", "")).expanduser()
CACHE = Path(os.environ.get("RICE_CACHE", "")).expanduser()
ARCHIVED = Path(os.environ.get("RICE_TABLES", "")).expanduser()
if not CKPT.is_file() or not CACHE.is_file():
    raise SystemExit("Set RICE_CKPT and RICE_CACHE to the authorized checkpoint and AZT1D cache.")
OUT = Path(os.environ.get("PICO_OUT", ROOT / "results" / "tables")).expanduser()
IMG = Path(os.environ.get("PICO_IMG", ROOT / "results" / "figures")).expanduser()

sys.path.insert(0, str(RELEASE_SRC))

from ceqt.eval import (  # noqa: E402
    curve_rows,
    event_rows_from_logits,
    event_rows_from_traj,
    slice_masks,
    subset,
)
from ceqt.models.ceqt import CEQT, CEQTNet  # noqa: E402
from ceqt.ops.ink import InertiaKernel  # noqa: E402

NAMES = {
    "CEQT": "RICE-Former",
    "lstm_x": "LSTM+X",
    "dual_lstm": "Dual-head LSTM",
    "tft": "TFT",
    "nhits": "N-HiTS+X",
    "timexer": "TimeXer",
    "patchtst": "PatchTST",
    "xgb_evt": "XGB event",
}
COLORS = {
    "RICE-Former": "#0072B2",
    "LSTM+X": "#D55E00",
    "Dual-head LSTM": "#009E73",
    "TFT": "#CC79A7",
    "N-HiTS+X": "#E69F00",
    "TimeXer": "#56B4E9",
    "PatchTST": "#777777",
    "XGB event": "#000000",
    "Trajectory threshold": "#8C564B",
}


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fmt_pair(fnr: float, fpr: float) -> str:
    return f"{fnr:.3f}/{fpr:.3f}"


def pick_event(df: pd.DataFrame, event: str, ph: int, head: str | None = None) -> pd.Series:
    q = (df["event"] == event) & (df["ph"] == ph)
    if head is not None:
        q = q & (df["head"] == head)
    return df.loc[q].iloc[0]


def pick_curve(df: pd.DataFrame, ph: int) -> pd.Series:
    return df.loc[df["ph"] == ph].iloc[0]


def finish(fig, stem: str) -> None:
    fig.savefig(IMG / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(IMG / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_curve(curve: pd.DataFrame) -> None:
    archived = pd.read_csv(ARCHIVED / "main_azt1d_curve.csv")
    archived["label"] = archived["model"].map(NAMES)
    rice = curve.copy()
    rice["label"] = "RICE-Former"
    others = archived[archived["label"] != "RICE-Former"]
    df = pd.concat([rice, others], ignore_index=True)
    order = [
        "RICE-Former",
        "LSTM+X",
        "Dual-head LSTM",
        "N-HiTS+X",
        "TimeXer",
        "PatchTST",
        "TFT",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    specs = [
        ("mae", "MAE (mg/dL)", "A  Point error", "lower is better"),
        ("ssa", "Shape similarity", "B  Curve shape", "higher is better"),
        ("peak_mae_min", "Peak-time error (min)", "C  Peak timing", "lower is better"),
    ]
    for ax, (metric, ylabel, title, note) in zip(axes, specs):
        for label in order:
            part = df[df["label"] == label].sort_values("ph")
            ax.plot(
                part["ph"] / 60,
                part[metric],
                marker="o",
                color=COLORS[label],
                linewidth=2.7 if label == "RICE-Former" else 1.3,
                markersize=5 if label == "RICE-Former" else 3.5,
                alpha=1 if label == "RICE-Former" else 0.78,
                label=label,
            )
        ax.set(title=title, xlabel="Prediction horizon (h)", ylabel=ylabel)
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(alpha=0.25)
        ax.text(0.03, 0.95, note, transform=ax.transAxes, va="top", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=0.23, wspace=0.30)
    finish(fig, "fig-2-curve-performance-by-horizon")


def plot_events(query: pd.DataFrame, traj: pd.DataFrame) -> None:
    archived = pd.read_csv(ARCHIVED / "main_azt1d_event.csv")
    keep = ["dual_lstm", "lstm_x", "timexer", "patchtst", "xgb_evt"]
    archived = archived[
        archived["model"].isin(keep)
        & archived["event"].isin(["hypo70", "hyper180"])
        & archived["ph"].isin([120, 240])
    ].copy()
    archived["label"] = archived["model"].map(NAMES)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, event, ph in [
        (axes[0, 0], "hypo70", 120),
        (axes[0, 1], "hypo70", 240),
        (axes[1, 0], "hyper180", 120),
        (axes[1, 1], "hyper180", 240),
    ]:
        baseline = archived[(archived.event == event) & (archived.ph == ph)]
        qrow = pick_event(query, event, ph)
        trow = pick_event(traj, event, ph)
        for _, row in baseline.iterrows():
            ax.scatter(row.fpr, row.fnr, s=42, color=COLORS[row.label], alpha=0.68, zorder=2)
            ax.annotate(
                row.label,
                (row.fpr, row.fnr),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6.5,
                color="#555555",
            )
        ax.scatter(
            trow.fpr,
            trow.fnr,
            s=92,
            marker="s",
            color=COLORS["Trajectory threshold"],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        ax.scatter(
            qrow.fpr,
            qrow.fnr,
            s=105,
            marker="o",
            color=COLORS["RICE-Former"],
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            "Trajectory threshold",
            (trow.fpr, trow.fnr),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=7,
            color=COLORS["Trajectory threshold"],
        )
        ax.annotate(
            "Event queries",
            (qrow.fpr, qrow.fnr),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
            weight="bold",
            color=COLORS["RICE-Former"],
        )
        ax.add_patch(
            FancyArrowPatch(
                (trow.fpr, trow.fnr),
                (qrow.fpr, qrow.fnr),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=2,
                color="#333333",
                connectionstyle="arc3,rad=0.06",
                zorder=3,
            )
        )
        if event == "hypo70":
            ax.axvline(0.06, color="#888888", linestyle="--", linewidth=1)
        ax.set(title=f"{event}, {ph // 60} h", xlabel="False-positive rate", ylabel="False-negative rate")
        ax.grid(alpha=0.25)
    fig.suptitle("Event operating points on AZT1D (RICE recomputed)", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    finish(fig, "fig-3-event-operating-points")


def plot_slices(curve_by_slice: dict[str, pd.DataFrame], query_by_slice: dict[str, pd.DataFrame]) -> None:
    archived = pd.read_csv(ARCHIVED / "slice_azt1d.csv")
    names = {"all": "All windows", "meal_2h": "Meal in prior 2 h", "bolus_3h": "Bolus in prior 3 h"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    keys = ["all", "meal_2h", "bolus_3h"]
    x = range(3)
    width = 0.36

    rice_mae = [float(pick_curve(curve_by_slice[k], 60)["mae"]) for k in keys]
    lstm_mae = [
        float(archived[(archived.model == "lstm_x") & (archived.slice == k) & (archived.ph == 60) & archived.mae.notna()]["mae"].iloc[0])
        for k in keys
    ]
    rice_h70 = [float(pick_event(query_by_slice[k], "hypo70", 240)["fnr"]) for k in keys]
    lstm_h70 = [
        float(
            archived[
                (archived.model == "lstm_x")
                & (archived.slice == k)
                & (archived.ph == 240)
                & (archived.event == "hypo70")
            ]["fnr"].iloc[0]
        )
        for k in keys
    ]
    rice_h180 = [float(pick_event(query_by_slice[k], "hyper180", 240)["fnr"]) for k in keys]
    lstm_h180 = [
        float(
            archived[
                (archived.model == "lstm_x")
                & (archived.slice == k)
                & (archived.ph == 240)
                & (archived.event == "hyper180")
            ]["fnr"].iloc[0]
        )
        for k in keys
    ]
    for ax, rice, lstm, ylabel, title in [
        (axes[0], rice_mae, lstm_mae, "PH60 MAE (mg/dL)", "A  Curve error"),
        (axes[1], rice_h70, lstm_h70, "PH240 hypo70 FNR", "B  Hypoglycemia misses"),
        (axes[2], rice_h180, lstm_h180, "PH240 hyper180 FNR", "C  Hyperglycemia misses"),
    ]:
        ax.bar([i - width / 2 for i in x], rice, width, label="RICE-Former", color=COLORS["RICE-Former"])
        ax.bar([i + width / 2 for i in x], lstm, width, label="LSTM+X", color=COLORS["LSTM+X"])
        ax.set_xticks(list(x), [names[k] for k in keys], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    finish(fig, "fig-4-event-slice-performance")


def write_report(
    n_win: int,
    subjects: list[int],
    curve: pd.DataFrame,
    query: pd.DataFrame,
    traj: pd.DataFrame,
    g_off: pd.DataFrame,
    curve_by_slice: dict[str, pd.DataFrame],
    query_by_slice: dict[str, pd.DataFrame],
    device: str,
) -> None:
    c = {int(r.ph): r for _, r in curve.iterrows()}
    go = {int(r.ph): r for _, r in g_off.iterrows()}
    q70 = {int(r.ph): r for _, r in query[query.event == "hypo70"].iterrows()}
    q180 = {int(r.ph): r for _, r in query[query.event == "hyper180"].iterrows()}
    t70 = {int(r.ph): r for _, r in traj[traj.event == "hypo70"].iterrows()}
    t180 = {int(r.ph): r for _, r in traj[traj.event == "hyper180"].iterrows()}

    def qpair(d, ph):
        return fmt_pair(float(d[ph].fnr), float(d[ph].fpr))

    meal_c = pick_curve(curve_by_slice["meal_2h"], 60)
    bolus_c = pick_curve(curve_by_slice["bolus_3h"], 60)
    meal_e = pick_event(query_by_slice["meal_2h"], "hypo70", 240)
    bolus_e = pick_event(query_by_slice["bolus_3h"], "hypo70", 240)

    lines = [
        "# AZT1D held-out set recomputation: RICE-Former",
        "",
        "| Item | Value |",
        "| --- | --- |",
        "| Date | 2026-08-31 |",
        "| Metrics | same public-archive metrics helpers: `layer1_table` / `event_rows_from_logits` / `event_rows_from_traj` / `slice_masks` |",
        f"| Weights | `{CKPT}` (same set as on the board) |",
        f"| Windows | {n_win} (paper: 19,152) |",
        f"| Held-out participants | {', '.join(str(s) for s in subjects)} |",
        f"| Device | {device} |",
        "",
        "RICE-Former rows are recomputed in this run. Baseline rows are taken from `runs/20260830_s20260828/tables/` and were not retrained in this run. The submitted v1.0 is unchanged.",
        "",
        "## Table 5.2 Trajectory (Fig. 2)",
        "",
        "| Model | MAE 1 h | MAE 2 h | MAE 3 h | MAE 4 h | SSA 1 h | SSA 2 h | SSA 3 h | SSA 4 h |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| RICE-Former (this run) | **{c[60].mae:.2f}** | **{c[120].mae:.2f}** | **{c[180].mae:.2f}** | **{c[240].mae:.2f}** "
            f"| **{c[60].ssa:.3f}** | **{c[120].ssa:.3f}** | **{c[180].ssa:.3f}** | **{c[240].ssa:.3f}** |"
        ),
        "| RICE-Former (manuscript) | **14.54** | **21.04** | **24.69** | **26.79** | **0.644** | **0.675** | **0.683** | **0.679** |",
        "| LSTM+X (archived) | 15.84 | 22.14 | 25.83 | 27.95 | 0.620 | 0.646 | 0.654 | 0.653 |",
        "",
        f"RMSE 1–4 h: {c[60].rmse:.2f} / {c[120].rmse:.2f} / {c[180].rmse:.2f} / {c[240].rmse:.2f}.",
        f"Peak time error: {c[60].peak_mae_min:.1f} / {c[120].peak_mae_min:.1f} / {c[180].peak_mae_min:.1f} / {c[240].peak_mae_min:.1f} min.",
        "",
        "![Curve performance across prediction horizons](images/fig-2-curve-performance-by-horizon.png)",
        "",
        "*Fig. 2. Trajectory performance on the AZT1D held-out set. RICE-Former is recomputed in this run; other neural baselines are taken from archived tables.*",
        "",
        "## Table 5.3 Inertia–residual and trajectory thresholding",
        "",
        "| Configuration | 1 h MAE | 4 h MAE | 4 h SSA | 4 h hypo70 FNR/FPR | 4 h hyper180 FNR/FPR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Full RICE-Former | {c[60].mae:.2f} | {c[240].mae:.2f} | {c[240].ssa:.3f} "
            f"| {qpair(q70, 240)} | {qpair(q180, 240)} |"
        ),
        (
            f"| Without learned residual | {go[60].mae:.2f} | {go[240].mae:.2f} | undefined for constant path "
            f"| {qpair(q70, 240)} | {qpair(q180, 240)} |"
        ),
        (
            f"| Events from trajectory thresholds | {c[60].mae:.2f} | {c[240].mae:.2f} | {c[240].ssa:.3f} "
            f"| {qpair(t70, 240)} | {qpair(t180, 240)} |"
        ),
        "",
        f"Residual contribution to MAE at 1 h / 4 h: {go[60].mae - c[60].mae:.2f} / {go[240].mae - c[240].mae:.2f} mg/dL.",
        "",
        "## Table 5.4 Event operating points (Fig. 3)",
        "",
        "| Model/output head | hypo70 2 h | hypo70 4 h | hyper180 2 h | hyper180 4 h |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| RICE-Former event query (this run) | {qpair(q70, 120)} | {qpair(q70, 240)} "
            f"| {qpair(q180, 120)} | {qpair(q180, 240)} |"
        ),
        f"| Trajectory thresholding of the same curve (this run) | {qpair(t70, 120)} | {qpair(t70, 240)} | {qpair(t180, 120)} | {qpair(t180, 240)} |",
        "| RICE-Former event query (manuscript) | 0.458/0.060 | 0.669/0.043 | 0.467/0.045 | 0.635/0.048 |",
        "| LSTM+X trajectory (archived) | 0.954/0.001 | 0.970/0.001 | 0.570/0.020 | 0.706/0.021 |",
        "",
        f"1 h hypo70 event query: {qpair(q70, 60)} (manuscript 0.209/0.079). 2 / 3 / 4 h FPR: {float(q70[120].fpr):.3f} / {float(q70[180].fpr):.3f} / {float(q70[240].fpr):.3f}.",
        "",
        "![Event operating points](images/fig-3-event-operating-points.png)",
        "",
        "*Fig. 3. FPR–FNR on the AZT1D held-out set. Blue dots and brown squares are the RICE event query and trajectory thresholding from this run; other baselines are taken from archived tables. The dashed line is the 0.06 FPR budget on the validation set.*",
        "",
        "## Fig. 4 Event-conditioned slices",
        "",
        f"All windows 1 h MAE {c[60].mae:.2f}; within 2 h after a meal {float(meal_c.mae):.2f} ({int(meal_c.n) // 12} windows); within 3 h after a bolus {float(bolus_c.mae):.2f} ({int(bolus_c.n) // 12} windows).",
        f"4 h hypo70: all {qpair(q70, 240)}; after meal {fmt_pair(float(meal_e.fnr), float(meal_e.fpr))}; after bolus {fmt_pair(float(bolus_e.fnr), float(bolus_e.fpr))}.",
        "",
        "![Event-conditioned performance](images/fig-4-event-slice-performance.png)",
        "",
        "*Fig. 4. Slices: all / meal within the last 2 h / bolus within the last 3 h. RICE bars are from this run; LSTM+X bars are taken from the archived `slice_azt1d.csv`.*",
        "",
        "## Event query over 16 decision cells (all windows)",
        "",
        "| Event | 1 h | 2 h | 3 h | 4 h |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for ev in ("hypo70", "hypo54", "hyper180", "hyper250"):
        cells = [fmt_pair(float(pick_event(query, ev, ph).fnr), float(pick_event(query, ev, ph).fpr)) for ph in (60, 120, 180, 240)]
        lines.append(f"| {ev} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines += [
        "",
        r"A decision is logit ≥ the frozen validation-set threshold; no 70/180 threshold is applied to \(\hat y\). Scores are not probability-calibrated.",
        "",
        "## Relation to the 32-window spot check",
        "",
        "The on-board golden windows and `pc_metrics_32.json` are used only for firmware alignment and do not replace this table. The 32-window MAE is higher because those windows are the first segment of the test sequence, not the full 19,152 windows.",
        "",
        "## Files",
        "",
        "- `curve.csv` / `event_query.csv` / `event_traj.csv` / `curve_g_off.csv` / `slice_curve.csv` / `slice_event.csv`",
        "- `summary.json`",
    ]
    (OUT / "azt1d-test-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = CEQTNet()
    net.load_state_dict(blob["state"])
    net.to(device)
    net.eval()
    thresholds = np.asarray(blob["thresholds"], dtype=np.float64)
    model = CEQT(
        net=net,
        ink=InertiaKernel(),
        y_mean=float(blob["y_mean"]),
        y_std=float(blob["y_std"]),
        use_g=True,
        use_quality=True,
        use_type=True,
        thresholds=thresholds,
    )
    with CACHE.open("rb") as f:
        data = pickle.load(f)
    test = data["test"]
    n_win = len(test)
    subjects = sorted(int(s) for s in np.unique(test.subject_id))
    print(f"test windows={n_win} subjects={subjects} device={device}", flush=True)

    out = model.predict_batch(test, device=device, chunk=1024)
    curve = pd.DataFrame(curve_rows("CEQT", test, out["yhat"]))
    query = pd.DataFrame(event_rows_from_logits("CEQT", test, out["logits"], thresholds, yhat=out["yhat"]))
    traj = pd.DataFrame(event_rows_from_traj("CEQT", test, out["yhat"]))
    g_off = pd.DataFrame(curve_rows("ceqt_g_off", test, out["y_iner"]))

    curve_by_slice = {"all": curve}
    query_by_slice = {"all": query}
    slice_curve_rows = list(curve.to_dict("records"))
    slice_event_rows = list(query.to_dict("records"))
    for name, mask in slice_masks(test).items():
        if name == "all":
            continue
        sub = subset(test, mask)
        print(f"slice {name} n={len(sub)}", flush=True)
        sub_yhat = out["yhat"][mask]
        sub_logits = out["logits"][mask]
        c_df = pd.DataFrame(curve_rows("CEQT", sub, sub_yhat, slice_name=name))
        e_df = pd.DataFrame(
            event_rows_from_logits("CEQT", sub, sub_logits, thresholds, yhat=sub_yhat, slice_name=name)
        )
        curve_by_slice[name] = c_df
        query_by_slice[name] = e_df
        slice_curve_rows.extend(c_df.to_dict("records"))
        slice_event_rows.extend(e_df.to_dict("records"))

    curve.to_csv(OUT / "curve.csv", index=False)
    query.to_csv(OUT / "event_query.csv", index=False)
    traj.to_csv(OUT / "event_traj.csv", index=False)
    g_off.to_csv(OUT / "curve_g_off.csv", index=False)
    pd.DataFrame(slice_curve_rows).to_csv(OUT / "slice_curve.csv", index=False)
    pd.DataFrame(slice_event_rows).to_csv(OUT / "slice_event.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    plot_curve(curve)
    plot_events(query, traj)
    plot_slices(curve_by_slice, query_by_slice)

    summary = {
        "n_windows": n_win,
        "subjects": subjects,
        "device": str(device),
        "mae": {int(r.ph): float(r.mae) for _, r in curve.iterrows()},
        "ssa": {int(r.ph): float(r.ssa) for _, r in curve.iterrows()},
        "g_off_mae": {int(r.ph): float(r.mae) for _, r in g_off.iterrows()},
        "hypo70_query": {
            int(r.ph): {"fnr": float(r.fnr), "fpr": float(r.fpr), "n_event": int(r.n_event)}
            for _, r in query[query.event == "hypo70"].iterrows()
        },
        "hyper180_query": {
            int(r.ph): {"fnr": float(r.fnr), "fpr": float(r.fpr), "n_event": int(r.n_event)}
            for _, r in query[query.event == "hyper180"].iterrows()
        },
        "hypo70_traj": {
            int(r.ph): {"fnr": float(r.fnr), "fpr": float(r.fpr)}
            for _, r in traj[traj.event == "hypo70"].iterrows()
        },
        "hyper180_traj": {
            int(r.ph): {"fnr": float(r.fnr), "fpr": float(r.fpr)}
            for _, r in traj[traj.event == "hyper180"].iterrows()
        },
        "note": "Full AZT1D holdout; same public-archive protocol. Not the 32-window firmware check.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(n_win, subjects, curve, query, traj, g_off, curve_by_slice, query_by_slice, str(device))
    print(json.dumps({k: summary[k] for k in ("n_windows", "mae", "ssa")}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
