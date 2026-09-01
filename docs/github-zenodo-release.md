# GitHub and Zenodo release procedure

GitHub repository name is `pico-former` (lowercase). The display title in README and Zenodo is **Pico-Former: a curve–event glucose Transformer on ESP32-S3**.

## Before publishing

1. Confirm `CITATION.cff` and `.zenodo.json` use `https://github.com/modalfuse/pico-former`.
2. Run the checks in `docs/release-checklist.md`.
3. Push to GitHub `pico-former` and tag `v0.1.0`.

## Connect GitHub to Zenodo

1. Sign in to Zenodo with the GitHub account that owns the repository.
2. Open the Zenodo GitHub integration and enable archiving for `pico-former`.
3. Create a GitHub release from tag `v0.1.0`.
4. Zenodo archives the release and issues a **version DOI** plus a **concept DOI**.

Use the **concept DOI** in the manuscript Data and code section (it always resolves to the latest version). Pair it with the RICE-Former concept DOI `10.5281/zenodo.22171786`.

## After Zenodo issues the DOI

1. Add the concept DOI to `CITATION.cff` as `doi`.
2. Add the DOI badge to `README.md`.
3. Commit the metadata update. Do not cut a new GitHub release solely for that commit.
