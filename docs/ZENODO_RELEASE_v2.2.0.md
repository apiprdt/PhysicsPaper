# Zenodo Release v2.2.0 Checklist

DOI concept record: [10.5281/zenodo.20534940](https://doi.org/10.5281/zenodo.20534940)
Target version: **2.2.0** (Phase 2 — Multivariable Correction Discovery)

> **Update:** Ada **2 jalur** untuk publish. Jalur A (otomatis, lewat GitHub)
> adalah yang Anda tanyakan. Jalur B (manual upload) sebagai cadangan.

---

## TL;DR — Pilih jalur

| Jalur | Cara kerja | Otomatis? | Butuh `.zenodo.json` |
|-------|-----------|-----------|----------------------|
| **A. GitHub → Zenodo** | Buat GitHub Release → Zenodo webhook auto-archive repo | ✅ Ya | ✅ Ya (sudah dibuat) |
| **B. Manual upload** | Upload zip + PDF via web Zenodo | ❌ Manual | Tidak perlu |

**Status saat ini:**
- ✅ `.zenodo.json` sudah dibuat di repo (metadata untuk auto-archive)
- ✅ Tag `v2.2.0` sudah ada di GitHub (commit `a95a5f3`)
- ❌ **Zenodo-GitHub integration belum di-enable** di akun/repo Anda
- ❌ `.zenodo.json` belum di-commit/push

---

## Jalur A — GitHub → Zenodo (otomatis) ⭐ direkomendasikan

### Langkah 1: Enable Zenodo-GitHub integration (sekali saja)

1. Buka https://zenodo.org/account/settings/github/
2. Login dengan akun GitHub yang sama dengan repo (`apiprdt`)
3. Cari repo `apiprdt/PhysicsPaper` di daftar → toggle **"On"**
4. Zenodo akan memasang webhook di repo Anda (cek:
   `https://github.com/apiprdt/PhysicsPaper/settings/hooks` — ada hook Zenodo)

> Setelah ini, **setiap GitHub Release** yang Anda buat akan otomatis
> di-archive ke Zenodo sebagai versi baru dari concept record yang sama.

### Langkah 2: Commit & push `.zenodo.json`

Supaya metadata Zenodo (title, author, description, license, keywords)
terbaca otomatis saat auto-archive, file ini harus ada di repo:

```powershell
cd "e:\Physics Project"
git add .zenodo.json docs/ZENODO_RELEASE_v2.2.0.md
git commit -m "docs: add .zenodo.json metadata for GitHub auto-archive"
git push origin main
```

> Tanpa `.zenodo.json`, Zenodo akan pakai metadata default dari GitHub
> (README-based), yang kurang rapi. Dengan file ini, semua field
> (title, license MIT, keywords, related identifiers) terisi persis.

### Langkah 3: Buat GitHub Release (trigger auto-archive)

```powershell
cd "e:\Physics Project"
git tag -a v2.2.0-zenodo -m "ADCD v2.2.0 - Phase 2 Multivariable"
# CATATAN: tag v2.2.0 sudah ada. Pakai tag yang sudah ada di langkah 4.
```

**Opsi tanpa `gh` CLI** (karena `gh` belum terinstall):

1. Buka https://github.com/apiprdt/PhysicsPaper/releases/new
2. Pilih tag yang sudah ada: **`v2.2.0`**
3. Isi:
   - **Release title:** `v2.2.0 — Phase 2 Multivariable Correction Discovery`
   - **Description** (copy-paste, lihat §Release notes di bawah)
4. (Opsional) Attach `paper/main.pdf` sebagai binary asset
5. Klik **"Publish release"**

### Langkah 4: Apa yang terjadi otomatis

Setelah Release publish:

1. GitHub mengirim webhook event ke Zenodo
2. Zenodo download snapshot repo pada tag `v2.2.0` (semua file sebagai zip)
3. Zenodo membaca `.zenodo.json` → isi metadata
4. Zenodo buat **versi baru** dari concept record `10.5281/zenodo.20534940`
5. Anda dapat email konfirmasi + DOI versi baru (~1-5 menit)

**Cek hasil:** https://doi.org/10.5281/zenodo.20534940 → tab "Versions"
→ v2.2.0 muncul di atas.

### Troubleshooting Jalur A

| Masalah | Solusi |
|---------|--------|
| Release ter-publish tapi Zenodo tidak update | Cek webhook: `Settings → Webhooks` di repo. Harus ada `zenodo.org`. Re-enable di https://zenodo.org/account/settings/github/ |
| Metadata Zenodo masih default (tidak baca `.zenodo.json`) | Pastikan `.zenodo.json` sudah merge ke `main` **sebelum** Release. Kalau sudah publish, edit manual di Zenodo atau hapus+recreate Release. |
| Zip Zenodo kosong / tidak ada file | Zenodo archive branch default. Pastikan Release pakai tag yang ada di branch `main`. |
| Ingin edit setelah publish | Zenodo: record → "Edit" → ubah → "Save". DOI versi tidak berubah. |

---

## Jalur B — Manual upload (cadangan, tanpa GitHub Release)

### Langkah 1: Login Zenodo

1. https://zenodo.org/login/ → login ORCID
2. Buka https://doi.org/10.5281/zenodo.20534940

### Langkah 2: New Version + Upload

1. Klik **"New version"** (kanan atas)
2. Upload file:

| File | Sumber |
|------|--------|
| `adcd-2.2.0.zip` | `e:\Physics Project\adcd-2.2.0.zip` (4.7 MB, 217 entri) |
| `main.pdf` | `e:\Physics Project\paper\main.pdf` (744 KB, paper) |

### Langkah 3: Edit metadata (copy-paste dari `.zenodo.json`)

Field-field penting:
- **Resource type:** Software
- **Title:** Anomaly-Driven Correction Discovery (ADCD): Physics-Constrained Symbolic Regression for Evolutionary Scientific Discovery
- **Version:** 2.2.0
- **Creator:** Erdita, Muhammad Afif — SMA Negeri 23 Kabupaten Tangerang, Indonesia
- **License:** MIT
- **Description:** copy dari field `description` di `.zenodo.json`
- **Keywords:** symbolic-regression, physics-constrained, equation-discovery, jax, scientific-machine-learning, sciml, physics-informed-ml, anomaly-detection, buckingham-pi, mond, multivariable
- **Related identifiers:**
  - `https://github.com/apiprdt/PhysicsPaper/tree/v2.2.0` → isSupplementTo
  - `https://apiprdt.github.io/PhysicsPaper/` → isPartOf

### Langkah 4: Save → Publish → salin DOI versi baru

---

## Release notes (untuk GitHub Release description)

```markdown
# ADCD v2.2.0 — Phase 2 Multivariable Correction Discovery

Physics-constrained symbolic regression that discovers **correction terms**
rather than learning equations from scratch.

## What's new (Phase 2 — Multivariable Discovery)

- 🧩 **Buckingham Π group engine** (`buckingham_pi.py`): nullspace-based dimensional Π-group generator
- 📐 **Sequential ARC** (`sequential_arc.py`): per-variable asymptotic-limit gate
- 🔬 **Residual Factorizer v2** (`residual_factorizer_v2.py`): variance-decomposition separability detection
- 🎯 **Multivariable Orchestrator** (`multivar_orchestrator.py`): end-to-end multi-input correction search
- 📊 **Phase 2 benchmark:** 2/4 multivariable scenarios solved (Yukawa Mass-Ratio, Turbulent Drag)

## Headline results

| Benchmark | Result |
|-----------|--------|
| 9-scenario (seed=42) | 94.4% structural recovery |
| 5-seed multi-seed (Tier B+) | 82.8% ± 7.7% |
| vs PySR fair @ 5% noise | 88.9% vs 11.1% (77.8 pp gap) |
| Real-world | 4/4 structural (Mercury, Lamb Shift, Muon g-2, Blackbody) |

## Links

- Docs: https://apiprdt.github.io/PhysicsPaper/
- Paper PDF: included as supplementary (`main.pdf`)
- Zenodo: https://doi.org/10.5281/zenodo.20534940

**Full Changelog**: https://github.com/apiprdt/PhysicsPaper/compare/v2.1.3...v2.2.0
```

---

## ⚠️ Catatan inkonsistensi versi di tag v2.2.0

Saat dibuild, di tag `v2.2.0`:
- `pyproject.toml` + `__init__.py` = **2.2.0** ✓
- `CITATION.cff` + README BibTeX = **2.1.3** ✗ (tertinggal)

Ini hanya masalah string di 2 file dokumentasi — tidak mempengaruhi kode.
Fix disarankan di commit berikutnya (bukan blocker untuk publish).

Untuk perbaiki:
```powershell
cd "e:\Physics Project"
# Edit CITATION.cff: version: 2.2.0, date-released: 2026-06-20
# Edit README.md BibTeX: version = {2.2.0}
git add CITATION.cff README.md
git commit -m "docs: sync version strings to 2.2.0"
```

---

## Post-publish checklist

- [ ] DOI versi baru resolve: `https://doi.org/10.5281/zenodo.XXXXXXX`
- [ ] Concept DOI tetap resolve: `https://doi.org/10.5281/zenodo.20534940`
- [ ] Tab "Versions" menampilkan v2.2.0 di atas
- [ ] File utama bisa di-download
- [ ] (Jalur A) Webhook Zenodo ada di repo Settings
- [ ] Update `CITATION.cff` + README BibTeX ke 2.2.0 (fix inkonsistensi)

## BibTeX (software) — pasca publish

```bibtex
@software{erdita2026adcd,
  author    = {Erdita, Muhammad Afif},
  title     = {{Anomaly-Driven Correction Discovery (ADCD): Physics-Constrained
                Symbolic Regression for Evolutionary Scientific Discovery}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.2.0},
  doi       = {10.5281/zenodo.20534940},
  url       = {https://doi.org/10.5281/zenodo.20534940}
}
```

---

## (Opsional) PyPI sync

PyPI dipublish via GitHub Release + Trusted Publishing
(`.github/workflows/publish.yml`). **Membuat GitHub Release otomatis juga
trigger PyPI publish ke 2.2.0** — jadi Jalur A sekali jalan double benefit
(Zenodo + PyPI).

Cek: https://pypi.org/project/adcd/

> Catatan: Anda sebelumnya pilih "tidak perlu push". Jalur A membutuhkan
> push `.zenodo.json` + GitHub Release. Kalau ingin benar-benar tanpa push,
> pakai Jalur B (manual upload).
