# Reproducibility

## Hardware

The reported board audit used an ESP32-S3 QFN56 (dual-core 240 MHz, hardware FP32 FPU, 16 MB Quad Flash, 8 MB in-package PSRAM). Working tensors stay in internal SRAM (about 192 KB). PSRAM is unused.

## Firmware

PlatformIO 6.x with Arduino-ESP32 2.0.x. Production path is FP32.

```bash
cp firmware/wifi.yaml.example firmware/wifi.yaml   # optional; for the HTTP snapshot
# export weights first (docs/data-access.md)
cd firmware
pio run -e s3_n16r8 -t upload
```

Inference is pinned to Core 1 (`xTaskCreatePinnedToCore`). HTTP handlers snapshot the last result on Core 0.

## Host environment

Use the RICE-Former conda environment (Python 3.13) plus `pyserial` for the audit script.

## On-board identity protocol

`host/run_validation.py` streams holdout windows over serial (`PFW1` / `PFR1`):

1. Golden-window max-abs on \(\hat y\) and logits.
2. Stratified 512-window over/under sample (seed 20260831).
3. Full 19,152-window paired FNR/FPR.
4. Dual-core jitter: 128 windows while Core 0 serves `GET /`.

Set `PICO_PORT` to the serial device. Do not commit `wifi.yaml` or `ceqt_weights.h`.

## Reported numbers

These values are locked in `results/validation/report.json` and `results/tables/summary.json`:

- 1 h / 4 h MAE 14.54 / 26.79 mg/dL (public host, same checkpoint)
- Window agreement 19,143 / 19,152 (99.953%)
- 4 h hypo70 FNR/FPR 0.669/0.043 on both sides
- Holdout median latency 398.2 ms
