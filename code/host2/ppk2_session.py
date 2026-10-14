"""One PPK2 power cycle around a Wi-Fi energy plan.

Starts ppk2_record.py (source mode, 3.30 V, power on), waits until the node
answers on Wi-Fi, runs run_energy_wifi.py, then stops the recording, which
turns the output off. The output is turned off even when the plan fails.

Usage:
  python ppk2_session.py --tag <tag> --plan <plan> --ip <node IP>
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_energy_wifi import request
from run_wireless import token

HERE = Path(__file__).resolve().parent
PY = sys.executable


def wait_node(ip: str, tok: str, rec: subprocess.Popen, seconds: float = 90.0) -> None:
    t0 = time.time()
    while time.time() - t0 < seconds:
        if rec.poll() is not None:
            raise RuntimeError(f"ppk2_record exited early with code {rec.returncode}")
        try:
            st, _ = request(ip, tok, "GET", "/api/energy", timeout=2.0)
            if st == 200:
                print(f"node reachable after {time.time() - t0:.1f} s", flush=True)
                return
        except OSError:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"node not reachable within {seconds:.0f} s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Record one PPK2 power cycle while a Wi-Fi energy plan runs.")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--ip", required=True)
    args = ap.parse_args()

    rec = subprocess.Popen(
        [PY, str(HERE / "ppk2_record.py"), "--tag", args.tag, "--power-on", "--power-off-at-end"],
        cwd=HERE,
    )
    rc = 1
    try:
        wait_node(args.ip, token(), rec)
        rc = subprocess.call(
            [PY, str(HERE / "run_energy_wifi.py"), "--ip", args.ip, "--plan", args.plan, "--tag", args.tag],
            cwd=HERE,
        )
    finally:
        if rec.poll() is None:
            rec.send_signal(signal.SIGTERM)
            try:
                rec.wait(timeout=60)
            except subprocess.TimeoutExpired:
                rec.kill()
                rec.wait()
        print(f"session {args.tag} plan={args.plan} rc={rc} record={rec.returncode}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
