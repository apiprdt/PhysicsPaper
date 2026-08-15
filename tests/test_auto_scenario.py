import os
import tempfile
import numpy as np
import pytest
from adcd.auto_scenario import (
    UnitExtractor,
    UnitParseError,
    DomainTaxonomyError,
    build_scenario_from_csv,
    verify_name_invariance,
)

def test_bracket_formats_near_universal():
    # Includes units that were NEVER in the old hand-written dictionary
    var, unit, vec, scale, raw = UnitExtractor.parse_header("orbital_velocity [km/s]")
    assert var == "orbital_velocity" and vec == (0, 1, -1, 0, 0) and np.isclose(scale, 1000.0)

    var, unit, vec, scale, raw = UnitExtractor.parse_header("pressure [Pa]")
    assert var == "pressure"
    assert vec == (1, -1, -2, 0, 0)  # Pascal = kg/(m*s^2), never in old dict

def test_bug1_suffix_invariance_now_passes():
    assert verify_name_invariance(["radius_km", "velocity_kms", "acceleration_ms2"]) is True

def test_bug1_bracket_invariance_still_passes():
    assert verify_name_invariance([
        "mercury_velocity [km/s]", "perihelion_distance (au)", "solar_mass [kg]",
    ]) is True

def test_bug2_oov_fails_loud():
    with pytest.raises(UnitParseError):
        UnitExtractor.parse_header("weird_column_xyz123")

def test_bug4_ambiguous_short_suffix_not_guessed():
    with pytest.raises(UnitParseError):
        UnitExtractor.parse_header("sample_min")

def test_bug5_second_bracket_found():
    var, unit, vec, scale, raw = UnitExtractor.parse_header("velocity (approx) [km/s]")
    assert vec == (0, 1, -1, 0, 0)
    assert np.isclose(scale, 1000.0)

def test_domain_taxonomy_validated():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = f.name
        f.write("radius_km,velocity_kms,acceleration_ms2\n10000,7.5,5.625\n")
    try:
        with pytest.raises(DomainTaxonomyError):
            build_scenario_from_csv(
                csv_path=csv_path, scenario_name="x", target_col="acceleration_ms2",
                classical_expr="velocity**2 / radius", domain="gravity",  # invalid, old silent-default bug
            )
    finally:
        os.remove(csv_path)

def test_bug3_multiplicative_residual_matches_eq1():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = f.name
        f.write("radius_km,velocity_kms,acceleration_ms2\n10000,7.5,6.75\n")  # 20% off classical
    try:
        scenario = build_scenario_from_csv(
            csv_path=csv_path, scenario_name="x", target_col="acceleration_ms2",
            classical_expr="velocity**2 / radius", domain="gravity_orbital",
            classical_limit_variable="velocity",
        )
        scenario.correction_type = "multiplicative"
        X, y_obs, y_classical, residual = scenario.generate_data()
        # y_classical = 7500^2/1e7 = 5.625 ; y_obs = 6.75 -> ratio - 1 = 0.2
        assert np.isclose(y_classical[0], 5.625)
        assert np.isclose(residual[0], 0.2)   # NOT 1.2 (the old buggy behavior)
    finally:
        os.remove(csv_path)

def test_full_integration_unchanged_behavior():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = f.name
        f.write("radius_km,velocity_kms,acceleration_ms2\n")
        f.write("10000,7.5,5.625\n20000,5.3,1.4045\n30000,4.33,0.6249\n")
    try:
        scenario = build_scenario_from_csv(
            csv_path=csv_path, scenario_name="Mock Orbit Anomaly",
            target_col="acceleration_ms2", classical_expr="velocity**2 / radius",
            domain="gravity_orbital", classical_limit_variable="velocity",
        )
        assert scenario.variables_with_units["radius"] == "meter"  # canonical SI base unit
        X, y_obs, y_classical, residual = scenario.generate_data()
        assert np.isclose(X["radius"][0], 10000.0 * 1000.0)
        assert np.isclose(X["velocity"][0], 7500.0)
        assert np.isclose(y_classical[0], 5.625)
        assert np.isclose(residual[0], 0.0)
    finally:
        os.remove(csv_path)

# --- Test mitigasi dari kritik user ---

def test_mitigasi1_trusted_bare_suffix_opt_in():
    # Tanpa opt-in: tetap diblokir (default aman)
    with pytest.raises(UnitParseError):
        UnitExtractor.parse_header("radius_m")
    # Dengan opt-in eksplisit: berhasil, dan memicu warning
    with pytest.warns(UserWarning):
        var, unit, vec, scale, raw = UnitExtractor.parse_header(
            "radius_m", trusted_bare_suffixes={"m"}
        )
    assert var == "radius" and vec == (0, 1, 0, 0, 0)

def test_mitigasi1_error_message_actionable():
    with pytest.raises(UnitParseError) as exc_info:
        UnitExtractor.parse_header("radius_m")
    msg = str(exc_info.value)
    assert "trusted_bare_suffixes" in msg and "[unit]" in msg

def test_mitigasi2_mol_column_gracefully_excluded():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = f.name
        f.write("radius_km,velocity_kms,concentration [mol/m^3],acceleration_ms2\n")
        f.write("10000,7.5,2.5,5.625\n")
    try:
        with pytest.warns(UserWarning):
            scenario = build_scenario_from_csv(
                csv_path=csv_path, scenario_name="x", target_col="acceleration_ms2",
                classical_expr="velocity**2 / radius", domain="gravity_orbital",
                classical_limit_variable="velocity",
            )
        # kolom mol dikecualikan, TAPI ingest CSV lain tetap sukses (bukan crash total)
        assert "concentration [mol/m^3]" in scenario.excluded_columns
        assert "radius" in scenario.variables_with_units
        X, y_obs, y_classical, residual = scenario.generate_data()
        assert np.isclose(y_classical[0], 5.625)
    finally:
        os.remove(csv_path)

def test_mitigasi2_mol_as_target_still_hard_fails():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = f.name
        f.write("radius_km,concentration [mol/m^3]\n10000,2.5\n")
    try:
        with pytest.raises(UnitParseError):
            build_scenario_from_csv(
                csv_path=csv_path, scenario_name="x", target_col="concentration [mol/m^3]",
                classical_expr="radius", domain="gravity_orbital",
            )
    finally:
        os.remove(csv_path)

def test_mitigasi3_missing_pint_gives_actionable_message():
    import subprocess
    import sys
    # Simulasikan environment tanpa pint terpasang -- pastikan pesan errornya
    # actionable (menyebut nama package & perintah install), bukan traceback mentah
    code = "import builtins; real_import = builtins.__import__\n" \
           "def fake_import(name, *a, **k):\n" \
           "    if name == 'pint': raise ModuleNotFoundError('no pint')\n" \
           "    return real_import(name, *a, **k)\n" \
           "builtins.__import__ = fake_import\n" \
           "import importlib, adcd.auto_scenario\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
    assert "pip install pint" in result.stderr
