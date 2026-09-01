# Public-release checklist

- [x] Public tree is a sibling of `rice-former`, not a git init of the lab `Pico-Former/` workspace.
- [x] Firmware copied without `.pio`, weights, or `wifi.yaml`.
- [x] Aggregate tables and on-board audit JSON included.
- [x] Raw participant windows (`gold/window0.json`) excluded.
- [x] Local absolute paths replaced by environment variables.
- [x] LAN IP stripped from `results/validation/report.json`.
- [x] AIM / AIIM manuscript IDs removed from public notes.
- [x] Code and non-code licenses recorded.
- [x] Citation and Zenodo metadata templates included.
- [x] GitHub repository name is `pico-former`.
- [ ] Enable `pico-former` in the Zenodo GitHub integration (https://zenodo.org/account/settings/github/).
- [x] Create GitHub release `v0.1.0`.
- [ ] After Zenodo mints the concept DOI, add it to `CITATION.cff`, `README.md`, and the manuscript Data and code section.
