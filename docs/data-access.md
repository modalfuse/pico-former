# Data access

Pico-Former does not redistribute CGM recordings or participant windows. The three trained checkpoints are provided in `checkpoints/` under CC BY 4.0.

## Datasets

| Dataset | Role in this archive | Source |
| --- | --- | --- |
| AZT1D | Holdout audit reported in `results/` | Mendeley Data, https://doi.org/10.17632/gk9m674wcx.1 |

Obtain AZT1D from the original provider and follow the layout in the RICE-Former `docs/data-access.md`.

## Protocol and host code

Clone [rice-former](https://github.com/modalfuse/rice-former) and point host scripts at its `src/` tree:

```bash
export RICE_FORMER_SRC="/absolute/path/to/rice-former/src"
```

Locked splits and \(\tau\) selection follow `protocol.json` and `split_azt1d.json` in that archive (Zenodo [10.5281/zenodo.22171786](https://doi.org/10.5281/zenodo.22171786)).

## Weights

`firmware/include/ceqt_weights.h` is generated locally from a checkpoint in `checkpoints/` (seed 20260828 is the deployed model; SHA-256 values in `checkpoints/README.md`):

```bash
export RICE_CKPT="checkpoints/s20260828/ceqt.pt"
export RICE_CACHE="/absolute/path/to/azt1d_cache.pkl"
python host/export_golden.py
```

The exporter also writes a local golden window under `gold/`. That file contains participant-level series and is gitignored.

The checkpoints are derived from AZT1D; cite the dataset (doi:10.17632/gk9m674wcx.1) together with this archive when using them.
