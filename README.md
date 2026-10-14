# Pico-Former: a curve–event glucose Transformer on ESP32-S3

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22218446.svg)](https://doi.org/10.5281/zenodo.22218446)

Repository: [https://github.com/modalfuse/pico-former](https://github.com/modalfuse/pico-former)  
Release: [v1.2.1](https://github.com/modalfuse/pico-former/releases/tag/v1.2.1)  
Zenodo: [https://doi.org/10.5281/zenodo.22218446](https://doi.org/10.5281/zenodo.22218446)

Pico-Former is an on-device curve–event glucose predictor for ESP32-S3. One Core-1 forward pass emits a 48-point 1–4 h trajectory and 16 Flash-threshold event flags. Core 0 serves wireless I/O. The network definition and \(\tau\) protocol follow the public RICE-Former archive.

This package contains the MCU kernels, the dual-core firmware with input checks, fault handling, and HTTP/HTTPS services, the serial audit host, the scripts that produce every table and figure of the revised manuscript and its Supplementary Material, the aggregate results they read, the PPK2 energy measurement scripts and per-step energy terms, and the three trained checkpoints. Raw CGM records, participant windows, firmware images, and dataset archives are not distributed.

## Companion archive

Host protocol, model code, and locked splits: [RICE-Former](https://github.com/modalfuse/rice-former) v1.0.6, Zenodo [10.5281/zenodo.22231231](https://doi.org/10.5281/zenodo.22231231) (concept DOI [10.5281/zenodo.22171786](https://doi.org/10.5281/zenodo.22171786)).

## Repository contents

- `firmware/`: PlatformIO project for ESP32-S3 (self-attention, cross-attention, residual glue, dual-core I/O, input checks, fault handling, HTTP and HTTPS). `env:s3_n16r8` is the node firmware; the other environments are test builds that differ only in build flags.
- `checkpoints/`: the three trained models (seeds 20260828, 20260829, 20260830; seed 20260828 is deployed), CC BY 4.0.
- `code/`: scripts of the revised manuscript (board audit, event tables, quantized baseline, design space, input and fault tests, wireless and HTTPS tests, PPK2 energy measurement, figures) and `code/results/`, the aggregate results they write.
- `host/`: export and serial-audit scripts. They import `ceqt` from a local RICE-Former checkout.
- `results/tables/`: aggregate AZT1D holdout curve and event tables.
- `results/validation/`: on-board summaries of the original submission (v1.0.x).
- `docs/`: data access, reproducibility, release notes, and `reproduce.md`, which lists identifiers, environments, and the command for every table and figure.

## Build the firmware

1. Install [PlatformIO](https://platformio.org/).
2. Export weights from a checkpoint in `checkpoints/` (see `docs/reproducibility.md`). The exporter writes `firmware/include/ceqt_weights.h`, which is gitignored.
3. Copy `firmware/wifi.yaml.example` to `firmware/wifi.yaml` if the HTTP snapshot page is needed.
4. Build and flash:

```bash
cd firmware
pio run -e s3_n16r8 -t upload
```

Inference is pinned to Core 1. Every window of five sets (the AZT1D held-out sets of three independently trained models, OhioT1DM, and DiaTrend; 180,029 windows) was replayed to ESP32-S3 boards (chip revision v0.2, 240 MHz, hardware FP32). All 2,880,464 decision cells agree with the float64 host reference of the same checkpoint (largest logit difference 2.6e-5), the paired FNR/FPR are equal on both sides (4 h hypo70 0.669/0.043 for the deployed model), the median pass time on the deployed-model holdout is 393.3 ms, and the 196,608 B working buffer stays in internal SRAM.

On the 3.3 V rail, measured with a Nordic PPK2 at 100 kS/s, one forward pass draws 93.2 mJ with Wi-Fi off, and the 5 min deployment cycle averages 168 mW (`code/results/energy/energy_tables.json`; commands in `docs/reproduce.md`).

## Host audit

```bash
export RICE_FORMER_SRC="/absolute/path/to/rice-former/src"
export RICE_CKPT="checkpoints/s20260828/ceqt.pt"
export RICE_CACHE="/absolute/path/to/azt1d_cache.pkl"
export PICO_PORT="/dev/cu.usbserial-0001"
python host/run_validation.py
```

Python 3.13 with the RICE-Former environment is sufficient for the host scripts.

## Data and model artifacts

This repository contains the three trained checkpoints (`checkpoints/`) and no raw CGM records, participant windows, firmware images, or dataset archives. Aggregate results report cohort-, participant-, or test-level metrics only.

AZT1D is available from Mendeley Data, <https://doi.org/10.17632/gk9m674wcx.1>; OhioT1DM (2018 and 2020 releases) from Ohio University under a data use agreement; DiaTrend from Synapse, <https://doi.org/10.7303/syn38187184>. Access terms are set by these providers.

## Citation

Cite both software archives:

> Liu Q, Yang M, Wang Z, Wang S, An X, Lu S, Yang Q, Liu M, Wu Z, Huang D. Pico-Former: a curve–event glucose Transformer on ESP32-S3. Zenodo. https://doi.org/10.5281/zenodo.22218446
>
> Liu Q, Yang M, Wang Z, Wang S, An X, Lu S, Yang Q, Liu M, Wu Z, Huang D. RICE-Former: a curve–event Transformer. Zenodo. https://doi.org/10.5281/zenodo.22171786

Citation metadata are also provided in `CITATION.cff`. Please cite the accompanying manuscript when referring to the scientific findings.

## Licensing

- Source code: MIT License (`LICENSE`).
- Documentation, aggregate tables, and trained checkpoints: CC BY 4.0 (`licenses/CC-BY-4.0.txt`). The checkpoints are derived from AZT1D; cite the dataset (doi:10.17632/gk9m674wcx.1) when using them.
