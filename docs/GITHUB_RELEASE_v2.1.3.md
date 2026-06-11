# GitHub Release v2.1.3

Tag **`v2.1.3`** — submission-ready (paper polish + Tier B+ benchmarks).

## Publish release (triggers PyPI)

1. Open https://github.com/apiprdt/PhysicsPaper/releases/new
2. Choose tag **v2.1.3**
3. Title: **ADCD v2.1.3**
4. Paste notes from [CHANGELOG.md](../CHANGELOG.md) sections [2.1.3] and [2.1.1]
5. Click **Publish release**

Workflow [`.github/workflows/publish.yml`](../.github/workflows/publish.yml) publishes to PyPI on `release: published`.

## Verify PyPI

```powershell
pip install adcd==2.1.3
python -c "import adcd; print(adcd.__version__)"
```

Expected output: `2.1.3`

Full step-by-step: [SUBMISSION_CHECKLIST_v2.1.3.md](SUBMISSION_CHECKLIST_v2.1.3.md)
