# Pico-Former: a curve–event glucose Transformer on ESP32-S3

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22218446.svg)](https://doi.org/10.5281/zenodo.22218446)

Repository: [https://github.com/modalfuse/pico-former](https://github.com/modalfuse/pico-former)  
Release: [v0.1.1](https://github.com/modalfuse/pico-former/releases/tag/v0.1.1)  
Zenodo: [https://doi.org/10.5281/zenodo.22218446](https://doi.org/10.5281/zenodo.22218446)

Pico-Former is an on-device curve–event glucose predictor for ESP32-S3. One Core-1 forward pass emits a 48-point 1–4 h trajectory and 16 Flash-threshold event flags. Core 0 serves wireless I/O. The network definition and \(\tau\) protocol follow the public RICE-Former archive.

This package contains the MCU kernels, the dual-core firmware glue, the serial audit host, and aggregate on-board audit tables. Raw CGM records, trained weights, credentials, and caches are not distributed.

## Companion archive

Host protocol, model code, and locked splits: [RICE-Former](https://github.com/modalfuse/rice-former), Zenodo [10.5281/zenodo.22171786](https://doi.org/10.5281/zenodo.22171786).

## Repository contents

- `firmware/`: PlatformIO project for ESP32-S3 (self-attention, cross-attention, residual glue, dual-core I/O).
- `host/`: export and serial-audit scripts. They import `ceqt` from a local RICE-Former checkout.
- `results/tables/`: aggregate AZT1D holdout curve and event tables.
- `results/validation/`: on-board agreement, paired FNR/FPR, latency, and energy-band summaries.
- `docs/`: data access, reproducibility, and GitHub–Zenodo release notes.

## Build the firmware

1. Install [PlatformIO](https://platformio.org/).
2. Obtain an authorized checkpoint and export weights (see `docs/reproducibility.md`). The exporter writes `firmware/include/ceqt_weights.h`, which is gitignored.
3. Copy `firmware/wifi.yaml.example` to `firmware/wifi.yaml` if the HTTP snapshot page is needed.
4. Build and flash:

```bash
cd firmware
pio run -e s3_n16r8 -t upload
```

Inference is pinned to Core 1. A 19,152-window holdout audit on ESP32-S3 QFN56 (240 MHz, hardware FP32) reported median latency 398.2 ms (0.13% of a 5 min CGM beat), a working buffer of about 192 KB internal SRAM, 19,143 / 19,152 window agreement, and identical 4 h hypo70 FNR/FPR 0.669/0.043 on the board and the public host of the same checkpoint.

## Host audit

```bash
export RICE_FORMER_SRC="/absolute/path/to/rice-former/src"
export RICE_CKPT="/absolute/path/to/ceqt.pt"
export RICE_CACHE="/absolute/path/to/azt1d_cache.pkl"
export PICO_PORT="/dev/cu.usbserial-0001"
python host/run_validation.py
```

Python 3.13 with the RICE-Former environment is sufficient for the host scripts.

## Data and model artifacts

This repository contains no raw CGM records, participant event logs, Wi-Fi passwords, trained weights, or caches. Aggregate tables report cohort-level metrics only.

AZT1D remains available from Mendeley Data, <https://doi.org/10.17632/gk9m674wcx.1>. Access terms are set by that provider.

## Citation

Cite both software archives:

> Liu Q, Yang M, Wang Z, Wang S, An X, Lu S, Yang Q, Liu M, Wu Z, Huang D. Pico-Former: a curve–event glucose Transformer on ESP32-S3. Zenodo. https://doi.org/10.5281/zenodo.22218446
>
> Liu Q, Yang M, Wang Z, Wang S, An X, Lu S, Yang Q, Liu M, Wu Z, Huang D. RICE-Former: a curve–event Transformer. Zenodo. https://doi.org/10.5281/zenodo.22171786

Citation metadata are also provided in `CITATION.cff`. Please cite the accompanying manuscript when referring to the scientific findings.

## Licensing

- Source code: MIT License (`LICENSE`).
- Documentation and aggregate tables: CC BY 4.0 (`licenses/CC-BY-4.0.txt`).
