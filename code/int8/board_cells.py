"""R2-14: per-window and per-cell comparison of the board ESP-DL / FP32-B replay (run_int8_board.py) with the host.

For each tag (q8, q16, qmix, fp32):
- number of windows whose board logits are bit-identical to the host result in the same format (<tag>_test.npz), and the max difference;
- with the frozen checkpoint thresholds (FP32 thresholds, not re-selected), cells where the board and the same-format host result disagree;
- cells and windows where the board and the same-format host result change relative to host FP32 (logits_fp32 in side_test.npz);
- latency and memory (board.json).
Output: results/int8/board_cells.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results" / "int8"
THR = np.load(HERE / "results" / "audit" / "azt1d_full" / "audit.npz")["thr"]


def main():
    side = np.load(RES / "side_test.npz")
    out = {}
    for tag in ("q8", "q16", "qmix", "fp32"):
        d = RES / f"board_{tag}"
        if not (d / "board.npz").exists():
            continue
        b = np.load(d / "board.npz")
        j = json.loads((d / "board.json").read_text())
        idx = b["idx"]
        host = np.load(RES / f"{tag}_test.npz")["logits"][idx]
        f32 = side["logits_fp32"][idx]
        lb = b["logits"]
        eb, eh, ef = lb >= THR, host >= THR, f32 >= THR
        diff = np.abs(lb - host)
        cells = np.argwhere(eb != eh)
        out[tag] = dict(
            n=int(len(idx)),
            windows_bit_identical_vs_host=int((diff.max(1) == 0).sum()),
            logits_differing_vs_host=int((diff > 0).sum()),
            logit_maxabs_vs_host=float(diff.max()),
            cells_disagree_vs_host=int(len(cells)),
            disagree_cells=[dict(window=int(idx[w]), out=int(k), board=float(lb[w, k]), host=float(host[w, k]),
                                 thr=float(THR[k])) for w, k in cells],
            board_cells_changed_vs_fp32=int((eb != ef).sum()),
            board_windows_changed_vs_fp32=int((eb != ef).any(1).sum()),
            host_cells_changed_vs_fp32=int((eh != ef).sum()),
            logit_maxabs_board_vs_fp32=float(np.abs(lb - f32).max()),
            latency_ms=j["latency_ms"],
            int_min_free_min=j.get("int_min_free_min"),
            psram_free_min=j.get("psram_free_min"),
            serial_retries=j.get("serial_retries"),
        )
        o = out[tag]
        print(tag, {k: v for k, v in o.items() if k != "disagree_cells"}, o["disagree_cells"][:3], flush=True)
    (RES / "board_cells.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
