#!/usr/bin/env python3
"""R2-8: summary of internal memory and stack records of the revised firmware over six full-set replays
(data source of Table S12).

Data come from existing replays: BUF / HEAP lines in boot logs boot_*.log, the mem field of audit.json
(per-window minima returned by the board), and section sizes of the frozen firmware ELF
(xtensa-esp32s3-elf-size -A). No new board runs.
Results are written to code/results/r2_8_mem/r2_8_mem.json.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "results" / "audit"
OUT = HERE / "results" / "r2_8_mem"
SIZE = Path.home() / ".platformio/packages/toolchain-xtensa-esp32s3/bin/xtensa-esp32s3-elf-size"
ELF = {
    "10cba768 (replays, seed 20260828)": HERE / "results/firmware_freeze_20261009/s3_n16r8_firmware.elf",
    "16b32e5f (release, seed 20260828)": HERE / "results/firmware_freeze_20261010/firmware.elf",
}
RUNS = [
    ("AZT1D s20260828", "WiFiduino", "azt1d_full"),
    ("AZT1D s20260829", "WiFiduino", "s20260829_full"),
    ("AZT1D s20260830", "WiFiduino", "s20260830_full"),
    ("OhioT1DM", "WiFiduino", "ohio_full"),
    ("DiaTrend", "DevKitC", "devkitc_diatrend17_full"),
    ("AZT1D s20260828", "DevKitC", "devkitc_azt1d_full"),
]
BUF = re.compile(r"BUF (\S+) addr=(\S+) bytes=(\d+) internal=(\d)")
HEAP = re.compile(r"HEAP (\S+) int_free=(\d+) int_largest=(\d+) int_min=(\d+) spiram_free=(\d+)")
SECTIONS = (".iram0.vectors", ".iram0.text", ".dram0.data", ".dram0.bss", ".flash.text", ".flash.rodata")


def boot_info(run_dir: Path) -> dict:
    boots = []
    for f in sorted(run_dir.glob("boot_*.log")):
        txt = f.read_bytes().decode("utf-8", "replace")
        bufs = [{"name": m[0], "addr": m[1], "bytes": int(m[2]), "internal": int(m[3])} for m in BUF.findall(txt)]
        heap = {m[0]: {"int_free": int(m[1]), "int_largest": int(m[2]), "int_min": int(m[3]),
                       "spiram_free": int(m[4])} for m in HEAP.findall(txt)}
        boots.append({"file": f.name, "bufs": bufs, "heap": heap})
    return {"n_boots": len(boots), "first": boots[0] if boots else None,
            "all_internal": all(b["internal"] == 1 for x in boots for b in x["bufs"])}


def sections(elf: Path) -> dict:
    out = subprocess.run([str(SIZE), "-A", str(elf)], capture_output=True, text=True, check=True).stdout
    sz = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2 and p[0] in SECTIONS:
            sz[p[0]] = int(p[1])
    sz["static_ram"] = sz[".dram0.data"] + sz[".dram0.bss"]
    sz["iram_code"] = sz[".iram0.vectors"] + sz[".iram0.text"]
    return sz


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, board, d in RUNS:
        a = json.loads((AUDIT / d / "audit.json").read_text())
        b = boot_info(AUDIT / d)
        h = b["first"]["heap"]
        rows.append({
            "set": name, "board": board, "dir": d, "windows": a["n"], "time": a["time"],
            "evt_cells_disagree": a["evt_cells_disagree"], "mem": a["mem"],
            "n_boots": b["n_boots"], "all_internal": b["all_internal"],
            "bufs": b["first"]["bufs"], "heap_stages": h,
            "init_drop_int": h["before_init"]["int_free"] - h["after_init"]["int_free"],
            "init_drop_spiram": h["before_init"]["spiram_free"] - h["after_init"]["spiram_free"],
            "net_drop_int": h["before_wifi"]["int_free"] - h["after_web"]["int_free"],
            "net_drop_spiram": h["before_wifi"]["spiram_free"] - h["after_web"]["spiram_free"],
        })
    res = {
        "runs": rows,
        "overall": {
            "int_min_min": min(r["mem"]["int_min_min"] for r in rows),
            "int_largest_min": min(r["mem"]["int_largest_min"] for r in rows),
            "stack_infer_hw_min": min(r["mem"]["stack_infer_hw_min"] for r in rows),
            "stack_serial_hw_min": min(r["mem"]["stack_serial_hw_min"] for r in rows),
            "windows": sum(r["windows"] for r in rows),
        },
        "sections": {k: sections(v) for k, v in ELF.items()},
    }
    (OUT / "r2_8_mem.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        m = r["mem"]
        print(f"{r['set']:16s} {r['board']:9s} n={r['windows']:6d} boots={r['n_boots']} internal={r['all_internal']} "
              f"init_drop={r['init_drop_int']}/{r['init_drop_spiram']} net_drop={r['net_drop_int']}/{r['net_drop_spiram']} "
              f"after_web={r['heap_stages']['after_web']['int_free']}/{r['heap_stages']['after_web']['int_largest']} "
              f"run_min={m['int_min_min']}/{m['int_largest_min']} stack={m['stack_infer_hw_min']}/{m['stack_serial_hw_min']}")
    print(json.dumps(res["overall"]))
    print(json.dumps(res["sections"], indent=1))


if __name__ == "__main__":
    main()
