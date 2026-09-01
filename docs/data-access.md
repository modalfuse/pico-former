# Data access

Pico-Former does not redistribute CGM recordings or trained weights.

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

`firmware/include/ceqt_weights.h` is generated locally from an authorized checkpoint:

```bash
export RICE_CKPT="/absolute/path/to/ceqt.pt"
export RICE_CACHE="/absolute/path/to/azt1d_cache.pkl"
python host/export_golden.py
```

The exporter also writes a local golden window under `gold/`. That file contains participant-level series and is gitignored.
