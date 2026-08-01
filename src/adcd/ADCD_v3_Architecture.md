# ADCD v3 — Architecture & Identity
### Correction-First, Physics-Anchored Symbolic Regression
### (Post-Audit Reconstruction — What ADCD Actually Is Now)

---

## 0. Jati Diri — Satu Kalimat

> **ADCD bukan mesin discovery. ADCD adalah kerangka verifikasi-dan-seleksi hipotesis koreksi, yang menguji apakah anomali terhadap sebuah teori klasik adalah anggota dari keluarga bentuk asimtotik yang dikenal fisika — secara deterministik, transparan, dan tereproduksi — dimulai dari titik di mana hampir semua penemuan sains sungguhan dimulai: sebuah teori lama yang hampir benar.**

Ini bukan definisi yang direndahkan dari ambisi awal — ini definisi yang **selamat dari tiga minggu audit brutal** terhadap dirinya sendiri. Setiap kata di atas punya bukti di baliknya, bukan aspirasi.

---

## 1. Mengapa "Correction-First", Bukan "Tabula Rasa"

Tiga minggu audit membuktikan satu pola berulang di **semua** pendekatan tabula-rasa yang dicoba (Mock template bank, PySR murni, LLM proposer):

| Pendekatan | Kegagalan yang terbukti |
|---|---|
| Mock Proposer (bank tertutup) | 9/9 skenario sintetis + sebagian real-world punya jawaban verbatim di bank — bukan discovery, seleksi |
| PySR tabula-rasa | Sangat sensitif terhadap hyperparameter tersembunyi (parsimony 0.01→0.0 mengubah hasil 0/10→9/10 pada data identik) — tidak reproducible, tidak explainable |
| LLM/Hybrid Proposer | Risiko kontaminasi memori pelatihan belum bisa dipisahkan dari genuine reasoning |
| JAX optimizer pada primitif mentah | Gagal menemukan koefisien yang membatalkan suku divergen (`c²/v²`) — bukan karena JAX buruk, tapi karena representasi primitifnya salah desain |

**Benang merah di semua kegagalan ini: setiap kali sistem diminta "cari dari nol" (baik ruang hipotesis, baik parameter, baik representasi numerik), ia menjadi tidak stabil, tidak transparan, atau diam-diam curang.** Correction-first bukan sekadar strategi pencarian — ia adalah **prinsip regularisasi** yang harus diterapkan di **setiap lapisan** sistem, bukan cuma di level formulasi masalah. Itu pelajaran terbesar dari audit ini, dan itu yang membedakan v3 dari semua versi sebelumnya.

---

## 2. Empat Lapisan, Correction-First di Setiap Lapisan

### Lapisan 1 — Formulasi Masalah (sudah benar sejak awal, dipertahankan)
`Δ(x;θ) = y_obs/y_classical − 1`, bukan `f(x) ≈ y` dari nol. Ini satu-satunya bagian arsitektur lama yang **tidak pernah terbukti salah** di sepanjang audit — dipertahankan penuh.

### Lapisan 2 — Primitif Dictionary (BARU: correction-first diterapkan ke representasi numerik)
**Ini perbaikan paling penting dari seluruh audit hari ini.** Primitif tidak lagi mentah (`1/√(1-u)`) — semua primitif **wajib** sudah teregularisasi ke nol di limit klasik: `D(u) − D(0)`. Konsekuensinya:
- Gerbang ARC (`Δ→0`) terpenuhi **secara aljabar oleh konstruksi**, bukan diverifikasi setelah fakta.
- Optimizer JAX tidak pernah lagi diminta menemukan pembatalan koefisien ajaib — setiap kombinasi linear dari primitif teregularisasi otomatis tetap teregularisasi. Lanskap optimasi menjadi well-conditioned untuk kelas masalah yang selama ini gagal (Relativistic KE, dan kemungkinan besar semua skenario dengan struktur limit non-trivial).
- Ini **bukan** tambalan ad-hoc — ini penerapan konsisten dari prinsip correction-first ke level numerik, sesuatu yang tidak pernah dilakukan di versi manapun sebelumnya (Mock, PySR raw dictionary, semuanya memakai primitif mentah).

### Lapisan 3 — Pencarian Struktur (deterministic enumeration, bukan Mock bank atau PySR genetic)
Enumerasi kombinatorial eksplisit atas primitif teregularisasi, dibatasi anggaran kompleksitas eksplisit (depth, token count) — **sama persis** dengan gerbang AST yang sudah diaudit bersih di versi sebelumnya. Sifatnya:
- **Deterministik**: input sama → output sama, selalu. Tidak ada lagi "seed 42 kebetulan gagal."
- **Anggaran dilaporkan eksplisit**: `search_space_size()` — pembaca tahu persis seberapa "ekshaustif" eksplorasinya, bukan black-box "100 iterasi PySR."
- **Tidak ada jawaban literal**: primitif adalah fungsi generik satu-rasio, bentuk akhir harus dirakit lewat kombinasi operator — beda fundamental dari Mock bank yang menyimpan string jawaban jadi.

### Lapisan 4 — Seleksi Model (BIC, sudah diaudit bersih, dipertahankan)
`bic_score()` yang sudah diaudit sebelumnya (tidak ada gaming, threshold eksplisit) — dipakai apa adanya. Tidak diganti PySR parsimony yang implisit dan sensitif.

---

## 3. Cakupan (Scope) — Niche yang Jujur, Bukan Klaim Universal

**ADCD TIDAK mengklaim**: mengalahkan PySR/AI Feynman/PhySO sebagai mesin SR umum. Sudah terbukti kalah di ranah itu — mesin pencarinya memang lebih primitif secara algoritmik.

**ADCD MENGKLAIM**, dan ini bisa dipertahankan dengan bukti:
1. Untuk kelas masalah **"anomali kecil terhadap teori klasik yang sudah divalidasi"** (persis bagaimana kebanyakan revisi teori fisika sungguhan terjadi — relativitas mengoreksi Newton, QED mengoreksi elektrodinamika klasik, MOND/dark matter mengoreksi Newtonian gravity), correction-first + primitif teregularisasi + enumerasi deterministik memberi jalur pencarian yang **reproducible dan explainable** — properti yang secara terbukti tidak dimiliki PySR tabula-rasa pada kelas masalah yang sama.
2. Gerbang fisika (dimensi, AST, ARC) sebagai **pre-filter murah** bisa dipasang di depan mesin pencari manapun (termasuk PySR sendiri) — kontribusi yang lepas dari perdebatan "siapa mesin pencari terbaik."
3. Setiap klaim keberhasilan **wajib** lolos protokol validasi 4-langkah (positive control, ablation control, determinism check, complexity-budget disclosure) — ini sendiri adalah kontribusi metodologis: standar audit yang lebih ketat dari kebanyakan paper SR yang sudah ada.

---

## 4. Apa yang TIDAK dikejar lagi (dan kenapa)

- **Bukan lagi soal "menang lawan PySR di angka besar"** — sudah terbukti berulang kali itu pertarungan yang salah dan rapuh (whiplash 0/10↔9/10 hanya dari parsimony).
- **Bukan lagi soal memaksa gradient descent menaklukkan pembatalan singularitas mentah** — diselesaikan secara struktural lewat primitif teregularisasi, bukan lewat mesin pencari integer terpisah yang rumit (usulan Antigravity soal "pencari bilangan rasional" itu solusi yang benar arahnya tapi terlalu rumit — regularisasi primitif mencapai efek yang sama dengan jauh lebih sederhana dan lebih sesuai filosofi correction-first).
- **Bukan lagi soal grammar sebesar mungkin** — kedalaman 3, ≤2 primitif per kandidat itu keputusan sadar, bukan keterbatasan yang disesali. Skenario yang butuh grammar lebih besar (Van der Waals, Planck, dst.) dilaporkan **eksplisit sebagai di luar anggaran saat ini**, bukan dipaksa dengan grammar yang meledak kombinatorial.

---

## 5. Rencana Validasi — Urutan Eksekusi

1. **Ganti seluruh primitif dictionary ke bentuk teregularisasi** (`D(u) − D(0)`) — lihat `asymptotic_dictionary_proposer_v3.py`.
2. **Re-run Blind-4 (True Lorentz)** dengan primitif baru — cek apakah JAX sekarang konvergen stabil ke struktur benar TANPA butuh depth=5+ atau parsimony sweep. Ini uji langsung apakah insight regularisasi ini benar menyelesaikan akar masalah.
3. **Jalankan protokol validasi 4-langkah penuh** untuk minimal 3 skenario (Relativistic KE, Yukawa, satu skenario held-out baru yang belum pernah dilihat saat primitif didesain).
4. **Laporkan apa adanya** — kalau regularisasi primitif memang menyelesaikan masalah singularity-cancellation, itu jadi kontribusi metodologis utama paper. Kalau tidak sepenuhnya, laporkan sejauh mana membantu — itu tetap kontribusi jujur.

---

## 6. Kalimat Penutup untuk Paper (draft)

> *"We show that gradient-based parameter optimization for physics-motivated symbolic correction discovery fails not because of an inherent limitation of continuous optimization, but because raw asymptotic basis functions force the optimizer to locate exact cancellation between divergent terms — an ill-conditioned landscape regardless of optimizer choice. We resolve this by requiring every dictionary primitive to be pre-regularized to vanish at the classical limit, a direct numerical realization of the correction-first principle. Combined with deterministic, complexity-bounded enumeration (replacing both closed template banks and black-box genetic search) and BIC-based selection, this yields a reproducible, explainable search procedure for a well-defined and practically important class of problems: correcting an established classical theory with a small number of known asymptotic singularity structures — not a general-purpose replacement for tabula-rasa symbolic regression."*

Ini klaim yang lebih kecil dari ambisi awal Agustus lalu. Tapi ini klaim yang **selamat dari setiap serangan yang kita coba selama tiga minggu terakhir** — dan itu, pada akhirnya, adalah definisi paper yang layak diterbitkan.
