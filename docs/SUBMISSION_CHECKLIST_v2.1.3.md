# ADCD v2.1.3 — Panduan Submit Lengkap (Step-by-Step)

Tag **`v2.1.3`** sudah ada di GitHub. Ikuti urutan ini **setelah** commit terbaru (CHANGELOG + version bump) di-push.

---

## Pre-flight (lokal, ~5 menit)

```powershell
cd "e:\Physics Project"
pip install -e ".[dev]"
py -3.11 -m pytest tests/ -q
py -3.11 scripts/verify_paper_claims.py
```

Harapan: pytest hijau, `verify_paper_claims` → **`[ALL OK]`**.

PDF final: `paper\main.pdf` (21 halaman).

---

## Persiapan: update tag jika ada commit baru setelah v2.1.3

Jika Anda baru commit CHANGELOG / version bump **setelah** tag `v2.1.3` dibuat:

```powershell
cd "e:\Physics Project"
git push origin main
git tag -d v2.1.3
git tag -a v2.1.3 -m "ADCD v2.1.3 submission-ready"
git push origin v2.1.3 --force
```

> Aman karena GitHub Release **belum** dipublish. Jangan force-push tag jika Release sudah live.

Buat arsip Zenodo:

```powershell
git archive --format=zip --prefix=adcd-2.1.3/ v2.1.3 -o adcd-2.1.3.zip
```

---

## LANGKAH 1 — GitHub Release (→ PyPI otomatis)

**Waktu:** ~10 menit + tunggu CI ~5–10 menit  
**URL:** https://github.com/apiprdt/PhysicsPaper/releases/new

### 1.1 Buka halaman New Release

1. Login GitHub sebagai `apiprdt`
2. Buka repo **PhysicsPaper**
3. Klik **Releases** (sidebar kanan) → **Draft a new release**

### 1.2 Isi form

| Field | Nilai |
|-------|-------|
| **Choose a tag** | `v2.1.3` (pilih existing tag) |
| **Release title** | `ADCD v2.1.3` |
| **Description** | Paste dari [CHANGELOG.md](../CHANGELOG.md) section `[2.1.3]` + `[2.1.1]` highlights |

**Cuplikan release notes (copy-paste):**

```markdown
## ADCD v2.1.3 — Submission-ready paper polish

- Evaluation regimes disclosure (Primary Mock 5-seed / Supplementary Hybrid / Real-world)
- Fresh Tier B+ benchmarks: **82.8% ± 7.7%** mean structural recovery
- PySR fair gap fix: **77.8 pp** at 5% noise (abstract + body consistent)
- Related Work: PhySO + LaSR (v2.1.1)
- `verify_paper_claims.py` all claims pass

See CHANGELOG.md for full details.
```

### 1.3 Publish

1. Centang **Set as the latest release** (jika ada opsi)
2. **Jangan** centang "pre-release" kecuali masih draft
3. Klik **Publish release**

### 1.4 Verifikasi PyPI

1. Buka https://github.com/apiprdt/PhysicsPaper/actions → workflow **Publish to PyPI**
2. Tunggu status hijau
3. Lokal:

```powershell
pip install --upgrade adcd==2.1.3
python -c "import adcd; print(adcd.__version__)"
```

Harapan: `2.1.3`

> **Prasyarat sekali:** PyPI Trusted Publishing sudah dikonfigurasi (repo → workflow `publish.yml` → environment `pypi`). Jika belum, lihat https://docs.pypi.org/trusted-publishers/

---

## LANGKAH 2 — Zenodo (arsip software + DOI versi baru)

**Waktu:** ~15 menit  
**URL:** https://zenodo.org/records/20534940

### 2.1 Login & buat versi baru

1. Login Zenodo (pakai akun yang punya akses ke record `20534940`)
2. Buka https://zenodo.org/records/20534940
3. Klik tombol **New version** (kanan atas)

### 2.2 Upload file

1. Klik **Files** → upload `adcd-2.1.3.zip` (dari perintah `git archive` di atas)
2. Opsional: upload `paper/main.pdf` sebagai supplementary

### 2.3 Metadata

| Field | Nilai |
|-------|-------|
| **Publication date** | Hari publish |
| **Version** | `2.1.3` |
| **Title** | Anomaly-Driven Correction Discovery (ADCD) v2.1.3 |
| **Description** | Paste highlights CHANGELOG [2.1.3] + link paper |

**Related identifier:**

- Identifier: `https://github.com/apiprdt/PhysicsPaper/tree/v2.1.3`
- Relation: **is supplement to** / **is published in** (pilih yang sesuai template Zenodo-GitHub)

### 2.4 Publish

1. Review metadata
2. Klik **Publish**
3. Catat DOI versi baru (jika berbeda dari concept DOI `10.5281/zenodo.20534940`)
4. Pastikan https://doi.org/10.5281/zenodo.20534940 resolve

---

## LANGKAH 3 — arXiv (paper preprint)

**Waktu:** ~30 menit form + 1–2 hari moderasi  
**URL:** https://arxiv.org/submit

### 3.1 Persiapan file

**PDF:** `paper\main.pdf`

**Source tarball** (PowerShell):

```powershell
cd "e:\Physics Project\paper"
tar -czf ..\adcd-paper-source-v2.1.3.tar.gz main.tex figures\ generated\
```

Atau zip manual: `main.tex`, folder `figures/`, folder `generated/`.

### 3.2 Metadata (dari `paper/arxiv_metadata.txt`)

| Field | Nilai |
|-------|-------|
| **Title** | Anomaly-Driven Correction Discovery (ADCD): Physics-Constrained Symbolic Regression for Evolutionary Scientific Discovery |
| **Authors** | Muhammad Afif Erdita |
| **Abstract** | Copy dari `paper/main.tex` abstract (tanpa LaTeX `\textbf`) |
| **Comments** | 21 pages, 6 figures, 8 tables. Code: https://doi.org/10.5281/zenodo.20534940 |
| **Primary category** | cs.AI |
| **Cross-list** | physics.comp-ph, cs.LG |
| **License** | CC BY 4.0 |

### 3.3 Langkah di website arXiv

1. **Login** atau daftar akun arXiv
2. **Start new submission**
3. **Upload PDF** → `main.pdf`
4. **Upload source** → tarball/zip
5. **Metadata:**
   - Paste title, abstract, authors
   - Primary: **cs.AI**
   - Secondary: centang **physics.comp-ph** dan **cs.LG**
6. **License:** Creative Commons Attribution (CC BY 4.0)
7. **Review** semua field
8. **Submit**
9. Tunggu email moderasi (biasanya 1–2 hari kerja)

### 3.4 Setelah approve

1. Catat arXiv ID (mis. `arXiv:2606.xxxxx`)
2. Update README dengan link arXiv
3. Tandai checklist di `paper/arxiv_metadata.txt`:

```
[x] Submit at Zenodo (upload ZIP)
[x] arXiv submit (cs.AI + cross-list)
```

---

## Checklist ringkas

| # | Langkah | Status |
|---|---------|--------|
| 0 | `verify_paper_claims` ALL OK | ☐ |
| 1 | Push main + tag `v2.1.3` | ☐ |
| 2 | GitHub Release published | ☐ |
| 3 | PyPI `adcd==2.1.3` installable | ☐ |
| 4 | Zenodo v2.1.3 uploaded | ☐ |
| 5 | arXiv submitted | ☐ |
| 6 | arXiv ID added to README | ☐ |

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| PyPI workflow gagal | Cek Trusted Publishing di pypi.org; pastikan tag exact `v2.1.3` |
| LaTeX arXiv compile error | Upload `generated/` + semua `figures/`; jangan upload `main.log` |
| Zenodo "version exists" | Gunakan 2.1.3 bukan 2.1.0 |
| Abstract arXiv > 1920 char | Potong kalimat Hybrid di abstract jika perlu |
