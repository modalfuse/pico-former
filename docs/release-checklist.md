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
- [x] Enable `pico-former` in the Zenodo GitHub integration.
- [x] Create GitHub release `v0.1.0`, then `v0.1.1` after the hook was enabled.
- [x] Concept DOI `10.5281/zenodo.22218446` added to `CITATION.cff` and `README.md`.
- [x] Zenodo description aligned with the RICE-Former record (https://zenodo.org/records/22171918).
- [ ] GitHub release `v1.0.0` (this version updates the Zenodo landing page).
