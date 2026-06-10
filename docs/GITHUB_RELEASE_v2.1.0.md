# GitHub Release v2.1.0 / v2.1.1

Tag `v2.1.0` has been pushed to `origin`. Subsequent paper fixes and `hybrid_seed42_results.json` are on `main` after `bf0063a` — use tag **`v2.1.1`** for the submission-ready release.

## Publish release (triggers PyPI)

`gh` CLI is not required. Use the GitHub web UI:

1. Open https://github.com/apiprdt/PhysicsPaper/releases/new
2. Choose tag **v2.1.0**
3. Title: **ADCD v2.1.0**
4. Paste notes from [CHANGELOG.md](../CHANGELOG.md) section [2.1.0]
5. Click **Publish release**

The workflow `.github/workflows/publish.yml` publishes to PyPI on `release: published`.

## Verify PyPI

After the workflow completes:

```bash
pip install adcd==2.1.0
python -c "import adcd; print(adcd.__version__)"
```

Expected output: `2.1.0`
