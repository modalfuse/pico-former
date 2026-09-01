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
        "# AZT1D 留出集重算：RICE-Former",
        "",
        "| 项 | 内容 |",
        "| --- | --- |",
        "| 日期 | 2026-08-31 |",
        "| 口径 | same public-archive metrics helpers: `layer1_table` / `event_rows_from_logits` / `event_rows_from_traj` / `slice_masks` |",
        f"| 权重 | `{CKPT}`（板上同一套） |",
        f"| 窗口 | {n_win}（论文 19,152） |",
        f"| 留出参与者 | {', '.join(str(s) for s in subjects)} |",
        f"| 设备 | {device} |",
        "",
        "RICE-Former 行现场重算。对照模型行取自 `实验/ceqt/runs/20260830_s20260828/tables/`，本轮未重训。已投稿 v1.0 不改。",
        "",
        "## 表 5.2 轨迹（对应图 2）",
        "",
        "| 模型 | MAE 1 h | MAE 2 h | MAE 3 h | MAE 4 h | SSA 1 h | SSA 2 h | SSA 3 h | SSA 4 h |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| RICE-Former（本轮） | **{c[60].mae:.2f}** | **{c[120].mae:.2f}** | **{c[180].mae:.2f}** | **{c[240].mae:.2f}** "
            f"| **{c[60].ssa:.3f}** | **{c[120].ssa:.3f}** | **{c[180].ssa:.3f}** | **{c[240].ssa:.3f}** |"
        ),
        "| RICE-Former（稿） | **14.54** | **21.04** | **24.69** | **26.79** | **0.644** | **0.675** | **0.683** | **0.679** |",
        "| LSTM+X（归档） | 15.84 | 22.14 | 25.83 | 27.95 | 0.620 | 0.646 | 0.654 | 0.653 |",
        "",
        f"RMSE 1–4 h：{c[60].rmse:.2f} / {c[120].rmse:.2f} / {c[180].rmse:.2f} / {c[240].rmse:.2f}。",
        f"峰值时间误差：{c[60].peak_mae_min:.1f} / {c[120].peak_mae_min:.1f} / {c[180].peak_mae_min:.1f} / {c[240].peak_mae_min:.1f} min。",
        "",
        "![不同预测时程下的曲线性能](images/fig-2-curve-performance-by-horizon.png)",
        "",
        "*图 2. AZT1D 留出集轨迹性能。RICE-Former 为本次重算；其余神经对照取自归档表。*",
        "",
        "## 表 5.3 惯性–残差与轨迹设阈",
        "",
        "| 配置 | 1 h MAE | 4 h MAE | 4 h SSA | 4 h hypo70 FNR/FPR | 4 h hyper180 FNR/FPR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| 完整 RICE-Former | {c[60].mae:.2f} | {c[240].mae:.2f} | {c[240].ssa:.3f} "
            f"| {qpair(q70, 240)} | {qpair(q180, 240)} |"
        ),
        (
            f"| 无学习所得残差 | {go[60].mae:.2f} | {go[240].mae:.2f} | 常数路径不定义 "
            f"| {qpair(q70, 240)} | {qpair(q180, 240)} |"
        ),
        (
            f"| 由轨迹阈值生成事件 | {c[60].mae:.2f} | {c[240].mae:.2f} | {c[240].ssa:.3f} "
            f"| {qpair(t70, 240)} | {qpair(t180, 240)} |"
        ),
        "",
        f"残差在 1 h / 4 h 的 MAE 贡献：{go[60].mae - c[60].mae:.2f} / {go[240].mae - c[240].mae:.2f} mg/dL。",
        "",
        "## 表 5.4 事件工作点（对应图 3）",
        "",
        "| 模型/输出头 | hypo70 2 h | hypo70 4 h | hyper180 2 h | hyper180 4 h |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| RICE-Former 事件查询（本轮） | {qpair(q70, 120)} | {qpair(q70, 240)} "
            f"| {qpair(q180, 120)} | {qpair(q180, 240)} |"
        ),
        f"| 同曲线轨迹设阈（本轮） | {qpair(t70, 120)} | {qpair(t70, 240)} | {qpair(t180, 120)} | {qpair(t180, 240)} |",
        "| RICE-Former 事件查询（稿） | 0.458/0.060 | 0.669/0.043 | 0.467/0.045 | 0.635/0.048 |",
        "| LSTM+X 轨迹（归档） | 0.954/0.001 | 0.970/0.001 | 0.570/0.020 | 0.706/0.021 |",
        "",
        f"1 h hypo70 事件查询：{qpair(q70, 60)}（稿 0.209/0.079）。2 / 3 / 4 h FPR：{float(q70[120].fpr):.3f} / {float(q70[180].fpr):.3f} / {float(q70[240].fpr):.3f}。",
        "",
        "![事件工作点](images/fig-3-event-operating-points.png)",
        "",
        "*图 3. AZT1D 留出集 FPR–FNR。蓝点与棕方为本轮 RICE 事件查询与轨迹设阈；其余对照取自归档表。虚线为验证集 0.06 FPR 预算。*",
        "",
        "## 图 4 事件条件化分层",
        "",
        f"全部窗口 1 h MAE {c[60].mae:.2f}；餐后 2 h {float(meal_c.mae):.2f}（{int(meal_c.n) // 12} 窗）；大剂量后 3 h {float(bolus_c.mae):.2f}（{int(bolus_c.n) // 12} 窗）。",
        f"4 h hypo70：全部 {qpair(q70, 240)}；餐后 {fmt_pair(float(meal_e.fnr), float(meal_e.fpr))}；大剂量后 {fmt_pair(float(bolus_e.fnr), float(bolus_e.fpr))}。",
        "",
        "![事件条件化性能](images/fig-4-event-slice-performance.png)",
        "",
        "*图 4. 全部 / 近 2 h 进餐 / 近 3 h 大剂量分层。RICE 柱为本轮；LSTM+X 柱取自归档 `slice_azt1d.csv`。*",
        "",
        "## 16 格事件查询（全部窗口）",
        "",
        "| 事件 | 1 h | 2 h | 3 h | 4 h |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for ev in ("hypo70", "hypo54", "hyper180", "hyper250"):
        cells = [fmt_pair(float(pick_event(query, ev, ph).fnr), float(pick_event(query, ev, ph).fpr)) for ph in (60, 120, 180, 240)]
        lines.append(f"| {ev} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    lines += [
        "",
        r"决策为 logit ≥ 验证集冻结阈值，不是对 \(\hat y\) 再设 70/180。分数未经概率校准。",
        "",
        "## 与 32 窗抽检的关系",
        "",
        "板上黄金窗与 `pc_metrics_32.json` 只用于固件对齐，不能代替本表。32 窗 MAE 偏高，是因为取了 test 序列最前面一段，不是全文 19,152 窗。",
        "",
        "## 文件",
        "",
        "- `curve.csv` / `event_query.csv` / `event_traj.csv` / `curve_g_off.csv` / `slice_curve.csv` / `slice_event.csv`",
        "- `summary.json`",
    ]
    (OUT / "AZT1D测试集重算 20260831.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
