# ADCD v2.1.2 — Submission Checklist

Paper polish complete (8/10 path). Execute these steps manually in the browser.

## Pre-flight (local)

```powershell
cd "e:\Physics Project"
py -3.11 scripts/verify_paper_claims.py   # expect [ALL OK]
py -3.11 -m pytest tests/ -q
```

PDF artifact: `paper/main.pdf` (21 pages)

## Step 1 — Git tag and push

```powershell
git tag -a v2.1.2 -m "ADCD v2.1.2 submission-ready paper polish"
git push origin main
git push origin v2.1.2
```

## Step 2 — GitHub Release (triggers PyPI)

1. Open https://github.com/apiprdt/PhysicsPaper/releases/new
2. Tag: **v2.1.2**
3. Title: **ADCD v2.1.2**
4. Paste highlights: evaluation regimes disclosure, PySR fair gap fix (77.8 pp), fresh Tier B+ benchmarks (82.8% ± 7.7%)
5. Publish release

Verify PyPI after workflow completes:

```powershell
pip install adcd==2.1.0
python -c "import adcd; print(adcd.__version__)"
```

## Step 3 — Zenodo new version

1. Open https://zenodo.org/records/20534940 → **New version**
2. Create archive:

```powershell
git archive --format=zip --prefix=adcd-2.1.2/ v2.1.2 -o adcd-2.1.2.zip
```

3. Upload ZIP; set version **2.1.2**; link `https://github.com/apiprdt/PhysicsPaper/tree/v2.1.2`
4. Publish

## Step 4 — arXiv submit

Metadata: see [paper/arxiv_metadata.txt](../paper/arxiv_metadata.txt)

- Primary: **cs.AI**
- Cross-list: **physics.comp-ph**, **cs.LG**
- Upload: `paper/main.pdf` + source tarball (`main.tex`, `figures/`, `generated/`)
- License: CC BY 4.0

## Post-submit

- [ ] Mark arxiv_metadata.txt items 69–70 as `[x]`
- [ ] Add arXiv ID to README when approved
