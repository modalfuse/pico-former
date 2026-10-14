#!/bin/bash
# Three short power cycles (E1-E6) and one cycle power cycle (E7a, E7b).
# Usage: run_ppk2_sessions.sh <node-ip>
set -euo pipefail
IP="${1:?node ip required}"
cd "$(dirname "$0")"
PY="${PY:?set PY to the python interpreter}"
for k in 1 2 3; do
    echo "=== short r${k} $(date '+%Y-%m-%d %H:%M:%S') ==="
    "$PY" ppk2_session.py --tag "energy_short_r${k}" --plan short --ip "$IP"
done
echo "=== cycles $(date '+%Y-%m-%d %H:%M:%S') ==="
"$PY" ppk2_session.py --tag energy_cycles_r1 --plan cycles --ip "$IP"
echo "=== done $(date '+%Y-%m-%d %H:%M:%S') ==="
