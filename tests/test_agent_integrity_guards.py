"""
tests/test_agent_integrity_guards.py
====================================
Automated integrity and compliance guards for AI agents and human researchers.

Enforces:
1. No false novelty claims (all candidate formulas must have verified provenance).
2. Zero-memory quantitative reporting (markdown reports must match CSV numbers).
3. Core module freeze (src/adcd must never import from spike/).
4. Branch isolation (README.md must remain untouched on branch Exploring).
5. Formal CandidateProvenance validation contracts in ClaimCard.
"""

from __future__ import annotations

import ast
import csv
import os
import subprocess
import pytest
from pydantic import ValidationError

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPARC_CSV_PATH = os.path.join(REPO_ROOT, "reports", "real_world_sparc_fits.csv")
PHASE7_REPORT_PATH = os.path.join(REPO_ROOT, "reports", "phase7_final_synthesis.md")
SPARC_REPORT_PATH = os.path.join(REPO_ROOT, "reports", "real_world_sparc_audit.md")
README_PATH = os.path.join(REPO_ROOT, "README.md")
SRC_DIR = os.path.join(REPO_ROOT, "src", "adcd")

from adcd.claim_card import CandidateProvenance, ClaimCard  # noqa: E402


def test_sparc_candidates_have_explicit_provenance_and_no_false_novelty() -> None:
    """Verify that every candidate in SPARC audit has non-empty citation and is_novel_claim is False."""
    assert os.path.exists(SPARC_CSV_PATH), f"Missing CSV: {SPARC_CSV_PATH}"
    with open(SPARC_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) >= 6, f"Expected at least 6 models, found {len(rows)}"
    valid_origins = {"literature", "dictionary_primitive", "symbolic_regression", "postulate_derivation"}

    for row in rows:
        expr = row["model_expr"]
        family = row["family"]
        origin = row["origin_type"]
        citation = row["citation"]
        is_novel = row["is_novel_claim"]

        assert origin in valid_origins, f"Invalid origin_type '{origin}' for {expr}"
        assert len(citation.strip()) > 0, f"Candidate {expr} must have a non-empty citation"
        assert is_novel == "False", f"Candidate {expr} cannot be claimed as novel without proof (found {is_novel})"

        # Specific guard for Smooth-Max CES (L_n-Crossover / Churchill-Usagi)
        if family == "family_smooth_max_ces":
            assert origin == "literature", "Smooth-Max CES must be attributed to literature"
            assert "Churchill" in citation, "Smooth-Max CES must cite Churchill & Usagi (1972)"


def test_reports_match_csv_ground_truth_numbers() -> None:
    """Verify that markdown reports quote BIC and delta-BIC numbers matching CSV within rounding."""
    assert os.path.exists(SPARC_CSV_PATH), f"Missing CSV: {SPARC_CSV_PATH}"
    assert os.path.exists(PHASE7_REPORT_PATH), f"Missing report: {PHASE7_REPORT_PATH}"
    assert os.path.exists(SPARC_REPORT_PATH), f"Missing report: {SPARC_REPORT_PATH}"

    with open(SPARC_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    with open(PHASE7_REPORT_PATH, "r", encoding="utf-8") as f:
        phase7_text = f.read()

    with open(SPARC_REPORT_PATH, "r", encoding="utf-8") as f:
        sparc_report_text = f.read()

    for row in csv_rows:
        bic_val = float(row["bic"])
        dbic_best = float(row["delta_bic_vs_best"])

        # String representations for rounding check
        bic_1dec = f"{bic_val:.1f}"
        bic_2dec = f"{bic_val:.2f}"

        # Ensure either 1-dec or 2-dec appears in both reports
        assert (bic_1dec in phase7_text or bic_2dec in phase7_text), (
            f"BIC {bic_1dec} for {row['family']} missing in phase7_final_synthesis.md"
        )
        assert (bic_1dec in sparc_report_text or bic_2dec in sparc_report_text), (
            f"BIC {bic_1dec} for {row['family']} missing in real_world_sparc_audit.md"
        )

        # For non-best models, verify delta-BIC is quoted accurately
        if dbic_best > 1.0:
            dbic_1dec = f"{dbic_best:.1f}"
            dbic_2dec = f"{dbic_best:.2f}"
            assert (dbic_1dec in phase7_text or dbic_2dec in phase7_text), (
                f"Delta-BIC {dbic_1dec} for {row['family']} missing in phase7_final_synthesis.md"
            )


def test_core_adcd_never_imports_spike() -> None:
    """Verify that no module in src/adcd/ imports from spike/."""
    for root, _, files in os.walk(SRC_DIR):
        for file in files:
            if not file.endswith(".py"):
                continue
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content, filename=filepath)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("spike"), (
                            f"Illegal import in production module {file}: imports '{alias.name}'"
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        assert not node.module.startswith("spike"), (
                            f"Illegal import in production module {file}: imports from '{node.module}'"
                        )


def test_readme_untouched_on_exploring_branch() -> None:
    """Verify that README.md has no unstaged/staged modifications and remains clean."""
    res = subprocess.run(
        ["git", "status", "--porcelain", "README.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert res.stdout.strip() == "", "README.md must not be modified on branch Exploring"

    # Also verify that README does not contain branch-specific leak text
    with open(README_PATH, "r", encoding="utf-8") as f:
        readme_content = f.read()
    assert "Smooth-Max CES" not in readme_content, "README.md must remain clean of exploration artifacts"


def test_claim_card_provenance_validation() -> None:
    """Verify CandidateProvenance model validation rules."""
    # 1. Valid literature provenance
    prov = CandidateProvenance(
        expression="y = 1 / (1 - exp(-sqrt(x)))",
        family_name="family_mcgaugh_rar",
        origin_type="literature",
        citation_or_source="McGaugh et al. (2016)",
        is_novel_claim=False,
    )
    assert prov.is_novel_claim is False

    # 2. Rejection of literature marked as novel
    with pytest.raises(ValidationError) as exc_info:
        CandidateProvenance(
            expression="y = (1^n + b^n)^(1/n)",
            family_name="family_ces",
            origin_type="literature",
            citation_or_source="Churchill & Usagi (1972)",
            is_novel_claim=True,
        )
    assert "cannot be True" in str(exc_info.value)

    # 3. Rejection of empty citation
    with pytest.raises(ValidationError) as exc_info:
        CandidateProvenance(
            expression="y = x + 1",
            family_name="family_poly",
            origin_type="literature",
            citation_or_source="   ",
            is_novel_claim=False,
        )
    assert "non-empty" in str(exc_info.value)

    # 4. ClaimCard integration
    card = ClaimCard(
        baseline_expr="1.0",
        limit_variable="x",
        provenance=[prov],
    )
    assert card.provenance is not None
    assert len(card.provenance) == 1
    assert card.provenance[0].citation_or_source == "McGaugh et al. (2016)"
