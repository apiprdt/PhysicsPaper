# Zenodo Release v2.1.2 Checklist

DOI concept record: [10.5281/zenodo.20534940](https://doi.org/10.5281/zenodo.20534940)

> **Current target version:** 2.1.2 (see [SUBMISSION_CHECKLIST_v2.1.2.md](SUBMISSION_CHECKLIST_v2.1.2.md) for full steps)

## 1. Prerequisites

- Git tag `v2.1.2` pushed to GitHub
- GitHub Release published (triggers PyPI via `.github/workflows/publish.yml`)
- `pytest` and `verify_paper_claims.py` pass on tag commit

## 2. Create new Zenodo version

1. Open the existing Zenodo record → **New version**
2. Upload archive:
   ```powershell
   git archive --format=zip --prefix=adcd-2.1.2/ v2.1.2 -o adcd-2.1.2.zip
   ```
3. Set metadata:
   - **Version:** 2.1.2
   - **Title:** Anomaly-Driven Correction Discovery (ADCD) v2.1.2
   - **Description:** CHANGELOG [2.1.2] highlights (evaluation regimes, 82.8% ± 7.7%, PySR fair 77.8 pp, Tier B+ refresh)
   - **Related identifier:** `https://github.com/apiprdt/PhysicsPaper/tree/v2.1.2` (isSupplementTo)
4. Publish version → confirm DOI resolves

## 3. Post-upload

- README BibTeX `version = {2.1.2}` (updated)
- Optional: upload `paper/main.pdf` as supplementary material

## 4. BibTeX (software)

```bibtex
@software{erdita2026adcd,
  author    = {Erdita, Muhammad Afif},
  title     = {{Anomaly-Driven Correction Discovery (ADCD)}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.1.2},
  doi       = {10.5281/zenodo.20534940},
  url       = {https://doi.org/10.5281/zenodo.20534940}
}
```
