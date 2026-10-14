"""R2-21: digests of the released checkpoints and their frozen thresholds (Supplementary Tables S18a and S18b).

For each seed: SHA-256 of checkpoints/s<seed>/ceqt.pt, SHA-256 of the 16 thresholds as float32 little-endian bytes,
and the thresholds in the shortest decimal form that rounds to the stored float32 value. When the weight header
exported by host/export_golden.py is present (firmware/include/ceqt_weights.h for the deployed seed,
ceqt_weights_s<seed>.h for the others), its SHA-256 is reported as well.

Usage: python code/checkpoint_digests.py
Output: code/results/checkpoints/digests.json
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260828, 20260829, 20260830)
EVT = ["hypo70", "hypo54", "hyper180", "hyper250"]
OUTPUTS = [f"{e} {h} h" for e in EVT for h in (1, 2, 3, 4)]
OUT = ROOT / "code" / "results" / "checkpoints" / "digests.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def header_path(seed: int) -> Path:
    inc = ROOT / "firmware" / "include"
    return inc / ("ceqt_weights.h" if seed == SEEDS[0] else f"ceqt_weights_s{seed}.h")


def main():
    res = {}
    for seed in SEEDS:
        ck = ROOT / "checkpoints" / f"s{seed}" / "ceqt.pt"
        blob = torch.load(ck, map_location="cpu", weights_only=True)
        thr = np.asarray(blob["thresholds"], dtype="<f4")
        hdr = header_path(seed)
        res[str(seed)] = {
            "checkpoint_sha256": sha256(ck),
            "thresholds_sha256": hashlib.sha256(thr.tobytes()).hexdigest(),
            "weight_header_sha256": sha256(hdr) if hdr.is_file() else None,
            "thresholds": {name: str(v) for name, v in zip(OUTPUTS, thr)},
        }
        print(seed, res[str(seed)]["checkpoint_sha256"], res[str(seed)]["thresholds_sha256"],
              res[str(seed)]["weight_header_sha256"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
