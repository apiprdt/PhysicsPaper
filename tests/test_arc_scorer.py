import pytest
import sympy as sp
from adcd.arc_scorer import AsymptoticRegime, ARCScorer, _resolve_limit


def test_classical_kinetic_energy():
    """
    KASUS 1: Energi Kinetik Klasik (Lolos Batas Normal)
    Target: E_k -> 0 ketika v -> 0
    Kandidat Sah: 0.5 * m * v^2
    Ekspektasi: ARC Score mendekati atau sama dengan 1.0
    """
    regime = AsymptoticRegime(
        variable="v",
        limit_target=0,
        ground_truth_expr="0",
        weight=1.0
    )
    scorer = ARCScorer(regimes=[regime])
    
    candidate = "0.5 * m * v**2"
    score = scorer.score(candidate)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_pathological_kinetic_energy():
    """
    KASUS 2: Konstruksi Eksplisit Persamaan Cacat Fisik (Counterexample Teorema)
    Target: E_k -> 0 ketika v -> 0
    Kandidat Patologis: 0.5 * m * v^2 + (m * c^3) / v
    Catatan: Formula ini lolos uji dimensi fisis, namun runtuh (meledak ke tak hingga) pada batas asimtotik.
    Ekspektasi: ARC Score wajib bernilai MUTLAK 0.0 (Tier 3 Divergence Check Gate)
    """
    regime = AsymptoticRegime(
        variable="v",
        limit_target=0,
        ground_truth_expr="0",
        weight=1.0
    )
    scorer = ARCScorer(regimes=[regime])
    
    # Formula cacat yang lolos analisis dimensi fisis pembawa penyakit ekstrapolasi katastrofik
    pathological_candidate = "0.5 * m * v**2 + (m * c^3) / v"
    score = scorer.score(pathological_candidate)
    assert score == 0.0


def test_newtonian_gravity():
    """
    KASUS 3: Hukum Gravitasi Universal Newton
    Target: Gaya tarik F -> 0 ketika jarak antar benda r -> tak hingga (oo)
    Kandidat Sah: (G * M * m) / r^2
    Ekspektasi: ARC Score mendekati atau sama dengan 1.0
    """
    regime = AsymptoticRegime(
        variable="r",
        limit_target=sp.oo,
        ground_truth_expr="0",
        weight=1.0
    )
    scorer = ARCScorer(regimes=[regime])
    
    candidate = "(G * M * m) / r**2"
    score = scorer.score(candidate)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_multiple_regimes_weighting():
    """
    Pengujian multi-regime untuk memastikan pembobotan prior Bayesian berfungsi linear.
    """
    r1 = AsymptoticRegime(variable="v", limit_target=0, ground_truth_expr="0", weight=2.0)
    r2 = AsymptoticRegime(variable="r", limit_target=sp.oo, ground_truth_expr="0", weight=1.0)
    
    scorer = ARCScorer(regimes=[r1, r2])
    
    # Kandidat ini lolos r2 (r->oo hasilnya 0) tapi gagal di r1 (v->0 hasilnya oo)
    mixed_candidate = "(G * M * m) / (r**2 * v)"
    
    # r1 (bobot 2) kesamaan = 0.0 -> 2.0 * 0.0 = 0.0
    # r2 (bobot 1) kesamaan = 1.0 -> 1.0 * 1.0 = 1.0
    # Total nilai diharapkan = 1.0 / (2.0 + 1.0) = 0.33333...
    score = scorer.score(mixed_candidate)
    assert score == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_yukawa_decay_positive_theta_regression():
    """
    REGRESI A2: Famili eksponensial dengan parameter tak bertanda (Yukawa / screened Coulomb).

    Sebelum perbaikan: ``sp.limit(exp(-r/theta_1), r, oo)`` memunculkan
    ``NotImplementedError`` karena tanda ``theta_1`` tak tentu, sehingga gerbang
    ARC menolak kandidat (termasuk literal ground-truth) dengan skor 0.0.
    Seharusnya: peluruhan eksponensial terhadap parameter skala positif ``theta_i``
    (dipaksakan positif oleh optimiser JAX via log-parameterisasi) wajib menuju 0
    ketika ``r -> oo`` dan mendapat skor 1.0.
    """
    # Tingkat helper: limit eksplisit harus teratasi menjadi 0
    r = sp.Symbol("r")
    resolved = _resolve_limit(sp.exp(-r / sp.Symbol("theta_1")), r, sp.oo)
    assert resolved == 0

    regime = AsymptoticRegime(variable="r", limit_target=sp.oo, ground_truth_expr="0", weight=1.0)
    scorer = ARCScorer(regimes=[regime])

    # Kandidat sah: gembur (screened) Yukawa yang meluruh ke nol
    assert scorer.score("exp(-r/theta_1)") == pytest.approx(1.0, abs=1e-5)
    # Varian polinomial-eksponensial juga harus diterima
    assert scorer.score("exp(-r/theta_2) / r**2") == pytest.approx(1.0, abs=1e-5)


def test_yukawa_divergence_still_rejected():
    """
    REGRESI A2 (countercheck): koreksi yang benar-benar divergen harus tetap ditolak.

    Memastikan asumsi theta positif hanya menyelamatkan kasus peluruhan, bukan
    menutupi divergensi nyata: ``exp(+r/theta_1)`` tetap harus mendapat skor 0.0.
    """
    regime = AsymptoticRegime(variable="r", limit_target=sp.oo, ground_truth_expr="0", weight=1.0)
    scorer = ARCScorer(regimes=[regime])
    assert scorer.score("exp(r/theta_1)") == 0.0
