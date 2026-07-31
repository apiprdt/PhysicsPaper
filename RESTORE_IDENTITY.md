# RESTORE IDENTITY AFTER ACCEPTANCE

This file documents every anonymized field that must be restored once the paper
is accepted and before the final camera-ready submission or public release.

---

## Files to Restore

### `src/adcd/__init__.py`
Add back:
```python
__author__ = "Muhammad Afif Erdita"
__email__  = "maeapip10@gmail.com"
__orcid__  = "0009-0004-7597-5221"
```
And add to module docstring:
```
DOI: 10.5281/zenodo.20534940
ORCID: https://orcid.org/0009-0004-7597-5221
```

### `pyproject.toml`
Restore `[project.authors]` and `[project.urls]`:
```toml
authors = [
    { name = "Muhammad Afif Erdita", email = "maeapip10@gmail.com" }
]

[project.urls]
Homepage   = "https://zenodo.org/records/20534940"
Repository = "https://github.com/apiprdt/PhysicsPaper"
"Bug Tracker" = "https://github.com/apiprdt/PhysicsPaper/issues"
DOI        = "https://doi.org/10.5281/zenodo.20534940"
ORCID      = "https://orcid.org/0009-0004-7597-5221"
```

### `paper/main.tex`
- Line ~47: restore `\author{Muhammad Afif Erdita \orcidlink{0009-0004-7597-5221} \\ ...}` block
- Line ~700: restore GitHub URL in Code Availability section

### `paper/supplementary.tex`
- Line ~29: restore `\author{Muhammad Afif Erdita}`

---

## Files that were NOT anonymized (safe to leave as-is)

These files contain identity but are **not submitted to reviewers** and should
remain unchanged:

| File | Reason |
|------|--------|
| `.zenodo.json` | Zenodo release metadata — public, not sent to reviewers |
| `README.md` | GitHub public readme |
| `CHANGELOG.md` | Public changelog |
| `CONTRIBUTING.md` | Community file |
| `CODE_OF_CONDUCT.md` | Community file |
| `mkdocs.yml` | Documentation config |
| `release_v3.0.0_notes.md` | Release notes |
| `zenodo_v3.0.0_description.md` | Zenodo description |

---

## SPARC Data Path (not identity-related — keep as-is)

`src/adcd/sparc_rar_scenario.py` now uses:
```python
os.environ.get("ADCD_SPARC_DATA", "data/sparc/kepler_sparc_clean_v2.csv")
```
After acceptance, upload `kepler_sparc_clean_v2.csv` to the repo at
`data/sparc/kepler_sparc_clean_v2.csv` **or** update the Zenodo record to
include the file and point users to it.

---

*Generated automatically during double-blind anonymization pass — 2026-07-30.*
