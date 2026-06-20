# Zenodo Release Guide

DOI concept record: [10.5281/zenodo.20534940](https://doi.org/10.5281/zenodo.20534940)
Current version: **2.2.1**

> **Update 2026-06-20:** Integrasi Zenodo-GitHub BELUM pernah aktif untuk repo ini
> (Zenodo masih menunjukkan v2.0.0). `.zenodo.json` sudah ditambahkan ke repo,
> tag `v2.2.1` sudah di-push. Anda hanya perlu **2 langkah di browser** (lihat di bawah).

---

## ⚡ 2 Langkah untuk publish ke Zenodo (otomatis via GitHub)

### Langkah 1 — Enable Zenodo-GitHub integration (sekali saja)

1. Buka https://zenodo.org/account/settings/github/
2. Login dengan akun GitHub `apiprdt`
3. Cari repo **`apiprdt/PhysicsPaper`** → toggle **On**
4. Verifikasi: buka https://github.com/apiprdt/PhysicsPaper/settings/hooks
   → harus ada webhook `zenodo.org`

### Langkah 2 — Buat GitHub Release

1. Buka https://github.com/apiprdt/PhysicsPaper/releases/new
2. Pilih tag: **`v2.2.1`**
3. Release title: `v2.2.1 — Phase 2 Multivariable + Zenodo Metadata`
4. Description (copy-paste):

```markdown
# ADCD v2.2.1 — Phase 2 Multivariable + Zenodo Metadata

## What's new since v2.2.0

- `.zenodo.json` metadata for Zenodo auto-archive integration
- Version string consistency fix (all files now report 2.2.1)

## Phase 2 (released in v2.2.0)

- 🧩 **Buckingham Π group engine** (`buckingham_pi.py`)
- 📐 **Sequential ARC** (`sequential_arc.py`)
- 🔬 **Residual Factorizer v2** (`residual_factorizer_v2.py`)
- 🎯 **Multivariable Orchestrator** (`multivar_orchestrator.py`)
- 📊 Phase 2 benchmark: 2/4 multivariable scenarios solved

## Headline results

| Benchmark | Result |
|-----------|--------|
| 9-scenario (seed=42) | 94.4% structural recovery |
| 5-seed multi-seed (Tier B+) | 82.8% ± 7.7% |
| vs PySR fair @ 5% noise | 88.9% vs 11.1% (77.8 pp gap) |
| Real-world | 4/4 structural (Mercury, Lamb Shift, Muon g-2, Blackbody) |

**Full Changelog**: https://github.com/apiprdt/PhysicsPaper/compare/v2.2.0...v2.2.1
```

5. Klik **Publish release**

### Apa yang terjadi otomatis

1. GitHub webhook → Zenodo
2. Zenodo snapshot repo pada tag `v2.2.1`
3. Zenodo baca `.zenodo.json` → isi metadata otomatis
4. Zenodo buat versi baru dari concept record → **DOI versi baru**
5. Email konfirmasi + DOI baru dalam 1-5 menit

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Zenodo tidak update setelah Release | Cek https://github.com/apiprdt/PhysicsPaper/settings/hooks — ada webhook `zenodo.org`? Kalau tidak, ulangi Langkah 1. |
| Metadata Zenodo masih default | `.zenodo.json` harus ada di commit yang ditag. Tag `v2.2.1` sudah mengarah ke commit yang berisi `.zenodo.json`. ✓ |
| Ingin edit metadata Zenodo pasca-publish | Buka record di Zenodo → Edit → Save. DOI versi tidak berubah. |

---

## Cadangan: Manual upload (tanpa GitHub Release)

Kalau Zenodo-GitHub integration tidak bisa di-enable, upload manual:

1. https://zenodo.org/login/ → ORCID login
2. Buka https://doi.org/10.5281/zenodo.20534940 → **New version**
3. Upload file dari repo (zip bisa dibuat: `git archive --format=zip --prefix=adcd-2.2.1/ v2.2.1 -o adcd-2.2.1.zip`)
4. Edit metadata sesuai `.zenodo.json`
5. Publish

---

## Post-publish checklist

- [ ] DOI versi baru resolve
- [ ] Concept DOI resolve: https://doi.org/10.5281/zenodo.20534940
- [ ] Tab "Versions" menampilkan v2.2.1
- [ ] File bisa di-download

## BibTeX

```bibtex
@software{erdita2026adcd,
  author    = {Erdita, Muhammad Afif},
  title     = {{Anomaly-Driven Correction Discovery (ADCD): Physics-Constrained
                Symbolic Regression for Evolutionary Scientific Discovery}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.2.1},
  doi       = {10.5281/zenodo.20534940},
  url       = {https://doi.org/10.5281/zenodo.20534940}
}
```
