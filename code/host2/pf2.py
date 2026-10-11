"""Host side of the revised Pico-Former audit protocol (PFW2 window frame, PFR2 result frame, PFD1 debug).

Also provides the host mirror of the node's input checks (R1-R8, same rules and order as ceqt_infer.cpp)
and the host reference forward pass on exactly the packed tensors sent to the board.
"""

from __future__ import annotations

import json
import os
import pickle
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial
import torch

BASE = Path(os.environ.get("PF_WORKSPACE", "."))
SRC = BASE / "rice-former/src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from ceqt.metrics import event_labels  # noqa: E402
from ceqt.models.ceqt import CEQTNet, pack_marks  # noqa: E402
from ceqt.ops.ink import InertiaKernel, ink_stack  # noqa: E402

N_CGM, N_MARK, N_OUT, N_EVT = 72, 24, 48, 16
BAUD = 460800
OK, BUSY, BAD_FRAME = 0, 1, 2
NO_PRED_GAP, NO_PRED_STALE, NO_PRED_INVALID = 10, 11, 12
FAULT_OUTPUT, FAULT_WEIGHTS, FAULT_INFER, FAULT_MEMORY = 20, 21, 22, 23
FLAG_TRUNC, FLAG_DROPPED, FLAG_STALE, FLAG_CLAMPED = 1, 2, 4, 8
CGM_MIN, CGM_MAX, CGM_TOL, STALE_INPUT_S = np.float32(40.0), np.float32(400.0), np.float32(2.0), 900
LOG1P_MAX = {0: np.float32(5.525453), 1: np.float32(3.433987), 2: np.float32(3.433987), 3: np.float32(3.583519)}

RES_FMT = "<IBBBB16sf4I48f16fI6IbBH4IBH"
RES_LEN = struct.calcsize(RES_FMT)
RES_FIELDS = ("seq", "status", "flags", "core", "n_dropped", "evt", "g", "us_total", "us_enc", "us_cross",
              "us_heads")


def xor_bytes(b: bytes) -> int:
    x = 0
    for v in b:
        x ^= v
    return x


def pack_window(seq, n_marks, quality, flags, t0_s, t_last_s, y_hist, feat, mask) -> bytes:
    payload = struct.pack("<HBBII", int(min(n_marks, 65535)), int(quality), int(flags), int(t0_s) & 0xFFFFFFFF,
                          int(t_last_s) & 0xFFFFFFFF)
    payload += np.asarray(y_hist, "<f4").reshape(-1).tobytes()
    payload += np.asarray(feat, "<f4").reshape(-1).tobytes()
    payload += np.asarray(mask, "<f4").reshape(-1).tobytes()
    body = struct.pack("<I", int(seq)) + payload
    return b"PFW2" + body + bytes([xor_bytes(body)])


def parse_result(frame: bytes) -> dict:
    body = frame[4:4 + RES_LEN]
    if xor_bytes(body) != frame[4 + RES_LEN]:
        raise ValueError("result xor mismatch")
    v = struct.unpack(RES_FMT, body)
    r = dict(zip(RES_FIELDS, v[:11]))
    r["evt"] = np.frombuffer(r["evt"], dtype=np.uint8).astype(np.int32)
    r["yhat"] = np.asarray(v[11:59], dtype=np.float32)
    r["logits"] = np.asarray(v[59:75], dtype=np.float32)
    rest = v[75:]
    keys = ("t0", "int_free", "int_min", "int_largest", "spiram_free", "stack_infer", "stack_serial", "rssi", "link",
            "reconnects", "http_req", "http_fail", "https_req", "https_fail", "reset_reason", "reset_count")
    r.update(dict(zip(keys, rest)))
    return r


class Board:
    def __init__(self, port: str, reset: bool = False, baud: int = BAUD):
        self.ser = serial.Serial(port, baud, timeout=0.05, dsrdtr=False, rtscts=False)
        self.ser.dtr = False
        self.ser.rts = False
        time.sleep(0.05)
        if reset:
            self.ser.rts = True
            time.sleep(0.08)
            self.ser.rts = False
        self.ser.reset_input_buffer()
        self.banner = ""
        self.ip = ""
        self.text = ""  # text lines seen while waiting for frames

    def close(self):
        self.ser.close()

    def _read(self) -> bytes:
        n = self.ser.in_waiting
        return self.ser.read(n if n else 1)

    def wait_ready(self, timeout: float = 40.0, marker: str = "Pico-Former ready") -> str:
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline:
            buf += self._read()
            text = buf.decode("utf-8", errors="replace")
            if marker in text:
                break
        self.banner = buf.decode("utf-8", errors="replace")
        for line in self.banner.splitlines():
            if line.startswith("wifi "):
                self.ip = line.split()[1]
        return self.banner

    def _wait_frame(self, seq: int, timeout: float) -> dict:
        deadline = time.time() + timeout
        buf = b""
        raw = b""
        why = []
        while time.time() < deadline:
            chunk = self._read()
            buf += chunk
            raw += chunk
            i = buf.find(b"PFR2")
            while i >= 0 and len(buf) >= i + 5 + RES_LEN:
                try:
                    r = parse_result(buf[i:i + 5 + RES_LEN])
                except ValueError as exc:
                    r = None
                    why.append(str(exc))
                if r is not None and r["seq"] == seq:
                    self.text += buf[:i].decode("utf-8", errors="replace")
                    return r
                if r is not None:
                    why.append(f"seq {r['seq']} status {r['status']}")
                buf = buf[i + 4:]
                i = buf.find(b"PFR2")
        self.last_raw = raw
        raise TimeoutError(f"no PFR2 for seq={seq}; raw {len(raw)} B; {why}")

    def infer(self, seq, n_marks, quality, flags, t0_s, t_last_s, y_hist, feat, mask, timeout: float = 5.0) -> dict:
        self.ser.write(pack_window(seq, n_marks, quality, flags, t0_s, t_last_s, y_hist, feat, mask))
        return self._wait_frame(seq, timeout)

    def debug(self, cmd: int, arg: int = 0, timeout: float = 5.0, expect: str = "STATUS ") -> str:
        body = struct.pack("<BI", cmd, arg & 0xFFFFFFFF)
        self.ser.write(b"PFD1" + body + bytes([xor_bytes(body)]))
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline:
            buf += self._read()
            if expect and expect.encode() in buf and buf.endswith(b"\n"):
                break
        return buf.decode("utf-8", errors="replace")

    def status(self) -> dict:
        text = self.debug(4)
        for line in text.splitlines():
            if line.startswith("STATUS "):
                return json.loads(line[7:])
        raise RuntimeError(f"no status: {text[-300:]}")

    def burn(self, n: int) -> dict:
        self.ser.write(b"PFB1" + struct.pack("<H", n))
        deadline = time.time() + n * 1.0 + 10
        buf = b""
        while time.time() < deadline:
            buf += self._read()
            if b"BURN done" in buf and buf.endswith(b"\n"):
                break
        text = buf.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("BURN done"):
                return {k: int(v) for k, v in (p.split("=") for p in line.split()[2:])}
        raise RuntimeError(text[-300:])


def attach(port: str, reset: bool = True, timeout: float = 40.0) -> Board:
    b = Board(port, reset=reset)
    if reset:
        b.wait_ready(timeout)
    return b


# ---------------- windows, guard and host reference ----------------

def load_cache(path: Path) -> dict:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def frames(batch) -> dict:
    """Packed tensors exactly as sent to the board (float32), plus side fields."""
    feat, mask = pack_marks(batch)
    q = np.array([int(np.any(np.isfinite(np.asarray(a, float)) & (np.asarray(a, float) > 0))) if len(a) else 0
                  for a in batch.mark_a], dtype=np.int32)
    n = np.asarray(batch.n_marks, dtype=np.int64)
    return {
        "y": np.asarray(batch.y_hist, np.float32), "feat": feat.astype(np.float32), "mask": mask.astype(np.float32),
        "n_marks": n, "q": q, "flags": (n > N_MARK).astype(np.int32) * FLAG_TRUNC,
        "t0_s": np.round(np.asarray(batch.t0, float) * 60).astype(np.int64),
        "t_last_s": np.round(np.asarray(batch.t_hist, float)[:, -1] * 60).astype(np.int64),
        "iner": ink_stack(InertiaKernel(), batch.t_hist, batch.y_hist, batch.t0, batch.horizons).astype(np.float64),
    }


def guard(y, feat, mask, q, flags, t0_s, t_last_s):
    """Host mirror of validate() in ceqt_infer.cpp for one window.

    Returns (status, y, feat, mask, q, flags, n_dropped); y has sensor-limit readings clamped to 40/400 mg/dL."""
    y = np.asarray(y, np.float32)
    if not (np.isfinite(y).all() and np.isfinite(feat).all() and np.isfinite(mask).all()):
        return NO_PRED_INVALID, y, feat, mask, q, flags, 0
    if (y < CGM_MIN - CGM_TOL).any() or (y > CGM_MAX + CGM_TOL).any():
        return NO_PRED_GAP, y, feat, mask, q, flags, 0
    if (y < CGM_MIN).any() or (y > CGM_MAX).any():
        y = np.clip(y, CGM_MIN, CGM_MAX)
        flags = flags | FLAG_CLAMPED
    if t0_s and t_last_s and (t0_s < t_last_s or t0_s - t_last_s > STALE_INPUT_S):
        return NO_PRED_STALE, y, feat, mask, q, flags, 0
    feat = feat.copy()
    mask = mask.copy()
    nd = 0
    for j in range(N_MARK):
        if mask[j] == 0:
            continue
        typf = np.float32(feat[j, 0])
        typ = int(typf)
        lg = np.float32(feat[j, 1])
        ok = (typf == np.float32(typ) and 0 <= typ <= 3 and lg > 0 and lg <= LOG1P_MAX[typ]
              and np.float32(feat[j, 2]) >= 0 and mask[j] == 1)
        if not ok:
            mask[j] = 0
            feat[j] = 0
            nd += 1
    if nd:
        q = 0
        flags = flags | FLAG_DROPPED
    return OK, y, feat, mask, q, flags, nd


def load_model(ckpt: Path):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = CEQTNet()
    net.load_state_dict(blob["state"])
    net.eval()
    return net, float(blob["y_mean"]), float(blob["y_std"]), np.asarray(blob["thresholds"], dtype=np.float32)


@torch.no_grad()
def host_forward(net, y_mean, y_std, y, feat, mask, n_marks, q, iner, device="cpu", dtype=torch.float64):
    """Host reference on the packed tensors: logits, yhat (firmware gate: g = n>0 ? (q ? 1 : 0.45) : 0)."""
    dev = torch.device(device)
    m = net.to(dev).to(dtype)
    yn = torch.from_numpy(((np.asarray(y, np.float64) - y_mean) / y_std)).to(dev, dtype)
    f = torch.from_numpy(np.asarray(feat, np.float64)).to(dev, dtype)
    k = torch.from_numpy(np.asarray(mask, np.float64)).to(dev, dtype)
    outs_d, outs_l = [], []
    for s in range(0, len(yn), 2048):
        d, lg = m(yn[s:s + 2048], f[s:s + 2048], k[s:s + 2048])
        outs_d.append(d.cpu().numpy())
        outs_l.append(lg.cpu().numpy())
    delta = np.concatenate(outs_d) * y_std
    logits = np.concatenate(outs_l)
    g = np.where(np.asarray(n_marks) > 0, np.where(np.asarray(q) > 0, 1.0, 0.45), 0.0)
    yhat = np.asarray(iner) + g[:, None] * delta
    net.to(torch.float32)
    return logits, yhat, g


def labels(batch) -> np.ndarray:
    return event_labels(batch.y_future, batch.horizons).astype(np.int32)
