# Zenodo Release v2.1.0 Checklist

DOI concept record: [10.5281/zenodo.20534940](https://doi.org/10.5281/zenodo.20534940)

## 1. Prerequisites

- Git tag `v2.1.0` pushed to GitHub
- GitHub Release published (triggers PyPI via `.github/workflows/publish.yml`)
- `pytest` and `verify_paper_claims.py` pass on tag commit

## 2. Create new Zenodo version

1. Open the existing Zenodo record → **New version**
2. Upload archive: GitHub release asset or `git archive`:
   ```bash
   git archive --format=zip --prefix=adcd-2.1.0/ v2.1.0 -o adcd-2.1.0.zip
   ```
3. Set metadata:
   - **Version:** 2.1.0
   - **Title:** Anomaly-Driven Correction Discovery (ADCD) v2.1.0
   - **Description:** Include CHANGELOG [2.1.0] highlights (binary pulsar v2.1, scale NMSE, Related Work PhySO/LaSR)
   - **Related identifier:** `https://github.com/apiprdt/PhysicsPaper/tree/v2.1.0` (isSupplementTo)
4. Publish version → confirm DOI resolves

## 3. Post-upload

- Update README Zenodo badge if version-specific badge is used
- Update BibTeX `version = {2.1.0}` (done in README for software citation)
- Cite NeurIPS 2024 LaSR if referencing LLM benchmark section

## 4. BibTeX (software)

```bibtex
@software{erdita2026adcd,
  author    = {Erdita, Muhammad Afif},
  title     = {{Anomaly-Driven Correction Discovery (ADCD)}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.1.0},
  doi       = {10.5281/zenodo.20534940},
  url       = {https://doi.org/10.5281/zenodo.20534940}
}
```
