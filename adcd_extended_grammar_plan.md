# ADCD Extended Grammar: Rencana Perbaikan Fundamental Proposer Layer

> **Status:** PROPOSED — Menunggu Implementasi Bertahap
> **Cakupan:** Revisi *proposer layer* ADCD. Gerbang fisika, JAX optimizer, PSC bootstrap, dan BIC reranking **tidak diubah**.
> **Filosofi Inti:** Correction-first tetap dipertahankan sepenuhnya. Dokumen ini memperluas *ruang kandidat* yang bisa diusulkan sistem — bukan mengganti paradigma pencarian.
> **Konteks:** Disusun setelah audit "Project Kepler" (klaim α≠0.5 di deep-MOND regime yang terbukti artefak kelengkungan fungsi MOND standar pada threshold berhingga) dan setelah proposal awal "Buckingham Pi + Complete Functional Basis" dari sesi riset internal.

---

## 0. Ringkasan Eksekutif

Proposal awal ("ganti proposer LLM dengan basis fungsi matematis yang lengkap dan bebas bias") **didiagnosis benar arahnya tapi salah pada klaim intinya**. Klaim "complete, bias-free functional basis" itu **tidak mungkin secara matematis** — ruang semua fungsi kontinu berdimensi tak hingga; katalog fungsi spesial (Bessel, erf, Lambert-W, dst.) adalah kurasi manusia (matematikawan, bukan fisikawan) yang sama terbatasnya dengan bank template lama, hanya dipindah kurasinya, bukan dihilangkan.

Dokumen ini menggantikan proposal "rombak total" dengan **ADCD Extended Grammar**: perluasan proposer yang jujur soal keterbatasannya, dilengkapi pengaman statistik (extended BIC, injection-recovery wajib) yang secara eksplisit dirancang untuk mencegah pengulangan kesalahan "Project Kepler" — di mana ruang pencarian yang lebih besar tanpa penalti yang sepadan justru meningkatkan risiko false discovery, bukan menguranginya.

**Nama, klaim head-to-head vs PySR, dan seluruh infrastruktur yang sudah tervalidasi (Section 3, 5, paper utama) tidak berubah.**

---

## 1. Diagnosis: Apa yang Benar, Apa yang Keliru

### 1.1 Yang Benar dari Proposal Awal

| Klaim | Status | Catatan |
|---|---|---|
| Proposer manapun (LLM atau manual) dibatasi oleh apa yang sudah dikenal manusia | ✅ Benar | Tapi ini berlaku untuk SEMUA symbolic regression, termasuk PySR — bukan kelemahan unik ADCD |
| Correction-first tidak bisa mengganti baseline yang salah total | ✅ Benar | Sudah diakui eksplisit di Limitations paper ADCD: *"the correction-first search offers no advantage over tabula rasa symbolic regression"* saat baseline salah fundamental |
| Hasil headline paper (88.9% vs PySR 11.1%) datang dari Mock Proposer, bukan LLM | ✅ Dikoreksi | Ini penting: kritik seharusnya diarahkan ke "bank kandidat terbatas" secara umum, bukan spesifik "LLM" |

### 1.2 Yang Keliru — dengan Bukti Matematis

**Klaim: "Complete Functional Basis, bebas bias literatur manusia"**

Ini keliru pada dua tingkat:

1. **Tingkat matematis:** Ruang fungsi kontinu $C^\infty$ berdimensi tak hingga. Tidak ada himpunan berhingga fungsi (sebesar apa pun) yang "lengkap" dalam artian mencakup semua fungsi yang mungkin. Bessel/erf/Lambert-W/Chebyshev hanyalah fungsi yang *diberi nama* karena berulang muncul di masalah yang *sudah dikenal* — itu tetap kurasi manusia, cuma dari domain matematika alih-alih fisika.

2. **Tingkat statistik:** Memperluas ruang kandidat tanpa memperketat penalti kompleksitas secara proporsional **meningkatkan** risiko *look-elsewhere effect* — persis mekanisme yang baru saja menjatuhkan klaim "Project Kepler" (α=0.5905 yang secara matematis murni adalah kelengkungan MOND standar pada threshold x<0.1, terverifikasi lewat simulasi zero-noise: slope OLS MOND murni pada rentang x∈[0.1,0.5] = 0.621 — LEBIH TINGGI dari klaim "penemuan" 0.5905-0.6155 yang dilaporkan sebagai stabil).

**Klaim: "Ini akan menyelesaikan masalah Einstein/baseline salah"**

Keliru. Memperluas Δ ke Bessel/erf/Lambert-W tetap mencari koreksi **di atas baseline yang tetap** — tidak mengubah fakta bahwa arsitektur correction-first mengasumsikan struktur baseline benar. Ini menjawab pertanyaan yang salah: "koreksi seperti apa yang bisa ditemukan" (jawabannya diperluas), bukan "bisakah baseline itu sendiri dipertanyakan" (jawabannya tetap tidak, dan memang di luar cakupan desain).

---

## 2. Prinsip yang Dipertahankan (Tidak Berubah)

1. **Correction-first paradigm** — fondasi filosofis ADCD, analog dengan cara Einstein tidak membuang $\frac{1}{2}mv^2$ Newton, melainkan mencari $\Delta_{rel}$ di atasnya.
2. **Physics gates** (AST complexity, dimensional homogeneity + transcendental guard, ARC asymptotic limit) — tidak diubah, tetap berlaku ke semua kandidat dari proposer manapun.
3. **BIC reranking sebagai filter pasca-hoc**, bukan tujuan pencarian utama.
4. **Nama "ADCD"** — tidak diganti. Ini evolusi proposer, bukan proyek baru.
5. **Klaim head-to-head vs PySR** yang sudah divalidasi (Section 5 paper) — tidak disentuh, tetap jadi fondasi klaim utama.

---

## 3. Arsitektur Baru: ADCD Extended Grammar

```
SEBELUM:
Data → [Proposer Tunggal: template manual ATAU LLM] → Gerbang Fisika → BIC → Output

SESUDAH:
Data + Variabel & Dimensi
       │
       ├──► [Proposer A: Template Manual (existing)]
       ├──► [Proposer B: Buckingham-Pi Extended Grammar (BARU)]
       ├──► [Proposer C: LLM Hybrid (existing, tetap dipertahankan)]
       │
       ▼
[Gerbang Fisika — SAMA untuk ketiga proposer]
       ▼
[Extended BIC — penalti disesuaikan dengan ukuran ruang pencarian aktif] (BARU)
       ▼
[PSC Bootstrap — SAMA seperti sekarang]
       ▼
[WAJIB: Injection-Recovery pada bentuk fungsi yang disembunyikan] (BARU, gate sebelum klaim apa pun)
       ▼
Output
```

### Fase 1 — Buckingham-Pi Engine Diperkuat (bukan diganti total)

- Gunakan `BuckinghamPiEngine` yang **sudah ada** di paper (Section 7) sebagai lapisan reduksi variabel — infrastruktur ini sudah divalidasi sebagian, tinggal dikuatkan, bukan ditulis ulang dari nol.
- **Perbaiki dulu bug ill-conditioning yang sudah didokumentasikan sendiri** di paper (koefisien Π-group kolaps ke bentuk degenerate `10⁻²⁰n·θ₀` pada presisi 64-bit saat konstanta referensi mencakup skala ekstrem) — ini prasyarat teknis sebelum grammar baru dipercaya jalan di data nyata.

### Fase 2 — Perluasan Grammar yang Jujur (Bertahap, Bukan Sekaligus)

- Tambahkan **5-8 fungsi baru per rilis**, masing-masing dengan justifikasi fisika spesifik untuk domain yang sedang diuji — bukan menambahkan seluruh katalog fungsi spesial sekaligus.
- Contoh justifikasi yang diterima: `erf(x)` untuk proses dengan distribusi kecepatan Gaussian, `J₀(x)` untuk geometri silinder/hamburan gelombang, `arctan(x)` untuk integrasi ruang-fase terbatas.
- Contoh yang **ditolak** tanpa justifikasi: menambahkan fungsi karena "kelihatan lengkap" atau "belum pernah dicoba" — itu kriteria yang sama persis yang menjerumuskan proposal awal.
- **Label eksplisit di dokumentasi:** "Extended Grammar", bukan "Complete Basis" — kejujuran bahasa ini wajib, mengikuti pelajaran dari setiap overclaim yang sudah dibongkar sepanjang riset internal (4.27×, transisi MOND tajam, NANOGrav, Project Kepler).

### Fase 3 — Extended BIC (Penalti Sepadan dengan Ukuran Ruang Pencarian)

Formula yang diusulkan:

$$\text{BIC}_{\text{ext}} = N\ln(\text{NMSE}) + k\ln(N) + 2\ln(M)$$

di mana $M$ = jumlah kandidat template aktif dalam grammar saat pencarian dijalankan (bukan jumlah parameter $k$ seperti BIC standar).

**Justifikasi:** ini analog prinsip yang dipakai untuk seleksi variabel berdimensi tinggi — makin besar ruang kandidat yang dijelajah, makin besar penalti yang dibutuhkan supaya "kecocokan yang kelihatan bagus secara kebetulan" tidak lolos begitu saja. Tanpa mekanisme ini, memperluas grammar sama dengan membuka pintu lebar untuk pengulangan Project-Kepler-style false positive, hanya dengan bungkus fungsi spesial alih-alih power law sederhana.

### Fase 4 — Protokol Injection-Recovery Wajib (Gate, Bukan Opsional)

Sebelum grammar baru dipercaya untuk anomali nyata **apa pun**:

1. Sembunyikan satu bentuk fungsi yang **sudah diketahui** dari bank template (misal: `Yukawa e^{-r/λ}`).
2. Bangkitkan data sintetis murni dari bentuk itu, dengan noise realistis meniru dataset target.
3. Jalankan grammar yang diperluas — cek apakah ia berhasil mendekati bentuk yang disembunyikan lewat kombinasi fungsi lain yang tersedia.
4. **Kalau gagal recover:** JANGAN lanjut ke anomali nyata. Itu tanda ruang pencarian baru lebih besar tapi belum genuinely lebih baik.
5. **Kalau berhasil terlalu mudah (recovery rate >95% di semua level noise):** curigai grammar terlalu fleksibel — jalankan juga uji sebaliknya: bangkitkan data dari MODEL BASELINE MURNI (tanpa koreksi sama sekali), cek apakah grammar salah menemukan "koreksi palsu" yang signifikan secara statistik. Ini persis uji yang seharusnya dilakukan sebelum klaim Project Kepler diajukan — bangkitkan data dari MOND standar murni di threshold x<0.1, jalankan seluruh pipeline, dan lihat apakah "penemuan" α≠0.5 muncul dari data yang jawabannya pasti α=0.5.

### Fase 5 — Multi-Proposer Paralel (Bukan Penggantian Total)

- LLM proposer **tidak dihapus**. Tiga proposer (Manual, Buckingham-Extended, LLM Hybrid) berjalan paralel, disaring gerbang fisika dan BIC yang sama.
- **Alasan:** kalau satu proposer bias/lemah di domain tertentu, dua lainnya tetap berfungsi sebagai silang-cek. Ini lebih robust daripada "ganti total dengan satu sistem baru" yang berisiko membawa bias baru yang belum teridentifikasi (persis seperti proposal awal yang berisiko menukar "bias literatur fisika" dengan "bias katalog fungsi spesial" tanpa disadari).

---

## 4. Protokol Verifikasi Wajib (Checklist Sebelum Klaim Apa Pun Naik Status)

- [ ] Grammar baru diberi label eksplisit "Extended", bukan "Complete" atau "Bias-Free", di semua dokumentasi/laporan.
- [ ] Setiap fungsi baru yang ditambahkan punya satu kalimat justifikasi fisika, dicatat di changelog grammar.
- [ ] Extended BIC ($+2\ln M$) diimplementasi dan diuji terhadap BIC standar pada minimal 3 benchmark sintetis lama (9-scenario suite paper) — pastikan tidak mengubah kesimpulan yang sudah tervalidasi.
- [ ] Injection-recovery (Fase 4, langkah 1-4) WAJIB lolos sebelum grammar baru dipakai pada dataset anomali nyata mana pun.
- [ ] Uji negatif (Fase 4, langkah 5) WAJIB dijalankan: bangkitkan data dari baseline MURNI tanpa koreksi, pastikan grammar baru TIDAK menemukan "koreksi signifikan" palsu.
- [ ] Bug ill-conditioning Buckingham-Pi (Section 7 paper) diperbaiki dan diuji ulang sebelum Fase 1 dianggap selesai.
- [ ] Setiap klaim hasil dari grammar baru melalui siklus audit yang sama seperti riset internal sejauh ini: cek provenance data, cek konsistensi numerik, cek apakah hasil bisa dijelaskan lebih sederhana oleh kelengkungan/artefak model yang sudah dikenal — sebelum dianggap "discovery".

---

## 5. Yang TIDAK Berubah

- Nama proyek: **ADCD**.
- Positioning utama: **correction-first symbolic regression**, dijual dengan analogi filosofi Einstein (koreksi di atas hukum klasik yang sudah benar sebagian besar) — epistemologi ini tetap kuat dan tidak perlu diubah.
- Klaim head-to-head vs PySR (77.8 poin persentase di noise 5%, Section 5 paper) — tetap fondasi klaim utama, tidak disentuh oleh revisi ini.
- Gerbang fisika (AST, dimensional, ARC) — tidak diubah.
- PSC bootstrap sebagai filter anti-false-positive — tidak diubah, cuma diterapkan juga ke proposer baru.

---

## 6. Risk Register

| Risiko | Mitigasi |
|---|---|
| Grammar diperluas tapi penalti tidak disesuaikan → false positive baru (ulangi Project Kepler) | Extended BIC (Fase 3) wajib, tidak opsional |
| Fungsi baru ditambahkan tanpa justifikasi fisika, jadi "tumpukan jerami" | Aturan 5-8 fungsi/rilis + justifikasi wajib per fungsi |
| Klaim "bebas bias" terulang di laporan/paper masa depan | Larangan eksplisit bahasa "complete/bias-free basis" di semua dokumentasi |
| Bug ill-conditioning Buckingham-Pi lama ikut terbawa ke grammar baru | Perbaikan bug jadi syarat Fase 1, bukan diabaikan |
| Multi-proposer menambah kompleksitas maintenance | Diterima sebagai trade-off sengaja demi robustness; didokumentasikan sebagai keputusan sadar |

---

## 7. Urutan Prioritas Implementasi

1. Perbaiki bug ill-conditioning Buckingham-Pi (prasyarat teknis).
2. Implementasi Extended BIC, uji regresi terhadap 9-scenario benchmark lama.
3. Tambah 5-8 fungsi baru gelombang pertama (dengan justifikasi eksplisit).
4. Jalankan injection-recovery wajib (positif + negatif) sebelum grammar baru menyentuh data nyata.
5. Baru setelah 1-4 lolos: terapkan ke domain aktif (SPARC $a_0$, dwarf spheroidal EFE) yang sedang berjalan.

---

## Appendix A: Argumen Matematis "Tidak Ada Basis Lengkap"

Ruang fungsi kontinu $C(\mathbb{R})$ (atau subruang halus $C^\infty$) berdimensi tak hingga sebagai ruang vektor atas $\mathbb{R}$. Himpunan berhingga fungsi apa pun — betapapun besar dan beragam (elementer + spesial + polinomial ortogonal) — merentang subruang berdimensi berhingga, yang secara ketat merupakan himpunan bagian nol-ukuran (*measure zero*) dari seluruh ruang fungsi yang mungkin. "Kelengkapan" dalam pengertian analisis fungsional (basis Hilbert/Banach lengkap untuk suatu ruang tertentu, misalnya deret Fourier untuk $L^2$) itu berbeda dari "kelengkapan" yang diklaim di sini — deret Fourier lengkap untuk $L^2$ tapi itu memerlukan **tak hingga suku**, bukan katalog berhingga fungsi bernama. Begitu dibatasi ke himpunan berhingga suku (seperti yang harus dilakukan agar bisa dihitung), sifat kelengkapan itu hilang, dan yang tersisa adalah **aproksimasi**, bukan kelengkapan — sama seperti bank template lama, hanya dengan aproksimasi yang lebih fleksibel.

## Appendix B: Extended BIC — Justifikasi

BIC standar menghukum jumlah parameter bebas $k$ dalam satu model yang sudah dipilih. Ia tidak menghukum **proses pemilihan model itu sendiri** dari sekian banyak kandidat yang dicoba — ini persis kelemahan yang membuat *look-elsewhere effect* mungkin terjadi (dibahas panjang di audit BAO, MOND breakpoint, dan Project Kepler sepanjang riset internal). Term $2\ln(M)$ adalah bentuk sederhana dari koreksi seleksi-model, konsisten dengan prinsip yang lebih umum di literatur *high-dimensional model selection* (mis. extended BIC/generalized information criteria untuk regresi dengan banyak kandidat prediktor): makin besar $M$ (jumlah kandidat yang benar-benar dievaluasi), makin besar penalti yang dibutuhkan agar tingkat false-discovery tetap terkendali.

## Appendix C: Kasus Rujukan dari Riset Internal

Dokumen ini secara eksplisit dibangun di atas pelajaran konkret dari riset internal sepanjang sesi ini:

- **Project Kepler** (α=0.5905 di deep-MOND): terbukti artefak kelengkungan MOND standar pada threshold berhingga (slope OLS zero-noise MOND murni di x∈[0.1,0.5] = 0.621, lebih tinggi dari klaim "penemuan"). Pelajaran: selalu uji apakah "sinyal" sudah diprediksi model lama sebelum diklaim baru.
- **NANOGrav spectral index**: bug kolaps matriks kovarians numerik menghasilkan signifikansi palsu ~10³¹σ. Pelajaran: provenance data dan kalkulasi numerik wajib diverifikasi independen, termasuk terhadap koreksinya sendiri.
- **GR Rediscovery** (perihelion + defleksi cahaya): berhasil memulihkan $\gamma_{PPN}=\beta_{PPN}\approx1.0$ dari data historis mentah tanpa bias circular, setelah lima ronde audit falsifikasi. Pelajaran: correction-first + gerbang fisika + audit berulang **bekerja** ketika diterapkan dengan disiplin — ini bukti bahwa memperbaiki disiplin verifikasi lebih penting daripada memperluas ruang pencarian.
- **SPARC $a_0$ fit**: χ²/dof≈4 ternyata dijelaskan tepat oleh ketidaksesuaian metodologi $\sigma_{int}$ (marginalisasi $M/L$ per galaksi vs $M/L$ tetap), bukan kesalahan fisika. Pelajaran: statistik yang aneh sering punya penjelasan metodologis sederhana sebelum dianggap sinyal fisik.
