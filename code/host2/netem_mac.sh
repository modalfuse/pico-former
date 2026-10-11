#!/bin/zsh
# R2-10 T6: inject packet loss and added latency (dummynet) into TCP traffic sent from the Mac to the board; restore after the test.
# Applies only to the Mac→board direction; the board→Mac direction is not controlled (stated in the results). Requires sudo.
# Rules are loaded into the anchor com.apple/pf2_t6 already referenced by the system pf.conf; /etc/pf.conf and the Clash Verge configuration are not modified.
# Usage:
#   sudo ./netem_mac.sh up <board IP> <loss rate, e.g. 0.01> <added latency in ms, e.g. 50>
#   sudo ./netem_mac.sh down
#   sudo ./netem_mac.sh show
set -eu
ANCHOR=com.apple/pf2_t6
TOKEN_FILE=/tmp/pf2_t6.token

case ${1:-} in
  up)
    IP=$2; PLR=$3; DELAY=$4
    dnctl pipe 1 config plr $PLR delay $DELAY
    echo "dummynet out quick proto tcp from any to $IP pipe 1" | pfctl -a $ANCHOR -f -
    pfctl -E 2>&1 | awk '/Token/ {print $NF}' > $TOKEN_FILE
    echo "T6 up: to $IP plr=$PLR delay=${DELAY}ms"
    ;;
  down)
    pfctl -a $ANCHOR -F all 2>/dev/null || true
    dnctl -q flush || true
    [[ -s $TOKEN_FILE ]] && pfctl -X $(cat $TOKEN_FILE) 2>/dev/null || true
    rm -f $TOKEN_FILE
    echo "T6 down"
    ;;
  show)
    pfctl -a $ANCHOR -s dummynet 2>/dev/null || true
    dnctl list || true
    ;;
  *)
    echo "usage: $0 up <ip> <plr> <delay_ms> | down | show" >&2
    exit 2
    ;;
esac
