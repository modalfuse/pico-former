"""R2-15 design space: spread across configurations and across seeds (Supplementary Table S15c).

For each metric: range and width of the 15 configuration means, mean seed-to-seed standard deviation (ddof=1), and
one-way ANOVA over the 15 groups of three seeds.

Input: code/results/design_space/grid_summary.json (eval_grid.py).
Output: code/results/design_space/spread_anova.json
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import f_oneway

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "code" / "results" / "design_space"
METRICS = [
    ("traj_mae_1h", "MAE 0-1 h (mg/dL)"),
    ("traj_mae_4h", "MAE 0-4 h (mg/dL)"),
    ("hypo70_4h_fnr", "4 h hypo70 FNR"),
    ("hyper180_4h_fnr", "4 h hyper180 FNR"),
]


def main():
    runs = json.loads((RES / "grid_summary.json").read_text())["runs"]
    configs = sorted({r["config"] for r in runs})
    out = {"n_configs": len(configs), "metrics": {}}
    for key, label in METRICS:
        groups = [np.array([r[key] for r in runs if r["config"] == c], dtype=float) for c in configs]
        means = np.array([g.mean() for g in groups])
        sds = np.array([g.std(ddof=1) for g in groups])
        f, p = f_oneway(*groups)
        out["metrics"][key] = {
            "label": label,
            "seeds_per_config": sorted({len(g) for g in groups}),
            "config_mean_min": float(means.min()),
            "config_mean_max": float(means.max()),
            "width": float(means.max() - means.min()),
            "mean_seed_sd": float(sds.mean()),
            "anova_f": float(f),
            "anova_p": float(p),
        }
        print(f"{label}: {means.min():.3f}-{means.max():.3f}, width {means.max() - means.min():.3f}, "
              f"mean seed SD {sds.mean():.3f}, F={f:.2f}, p={p:.2f}")
    (RES / "spread_anova.json").write_text(json.dumps(out, indent=2))
    print("wrote", RES / "spread_anova.json")


if __name__ == "__main__":
    main()
