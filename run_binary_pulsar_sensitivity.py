"""
Binary pulsar benchmark sensitivity study.

Tests whether ADCD recovery degrades as more physical parameters are exposed
(P only → P+e → P+e+M → full legacy scan), addressing reviewer concerns that
the v2.1 reduced-variable formulation is an easier benchmark.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from adcd.real_data_loader import load_binary_pulsar_decay, binary_pulsar_prefactor
from adcd.real_scenarios import RealAnomalyScenario
from run_correction_discovery import run_scenario_benchmark

VARIANTS = [
    ("P_only", ["P"], "Reduced: fixed M, a, e; only P varies"),
    ("P_e", ["P", "e"], "Intermediate: P and eccentricity vary"),
    ("P_e_M", ["P", "e", "M"], "Hard: P, e, and total mass vary"),
    ("full", ["P", "M", "a", "e"], "Legacy: full multi-parameter scan"),
]


@dataclass
class PulsarVariantScenario(RealAnomalyScenario):
    """RealAnomalyScenario with configurable binary pulsar loader variant."""
    loader_variant: str = "P_only"

    def generate_data(
        self,
        n_points: int = 200,
        noise_level: float = 0.0,
        seed: int = 42,
    ) -> Tuple[Dict, object, object, object]:
        return load_binary_pulsar_decay(seed=seed, variant=self.loader_variant)


def _base_pulsar_scenario() -> RealAnomalyScenario:
    from adcd.real_scenarios import get_real_scenarios
    return [s for s in get_real_scenarios() if "Pulsar" in s.name][0]


def make_variant_scenario(variant: str, variables: List[str]) -> PulsarVariantScenario:
    base = _base_pulsar_scenario()
    units = {v: base.variables_with_units.get(v, "dimensionless") for v in variables}
    return PulsarVariantScenario(
        name=f"Real: Binary Pulsar ({variant})",
        tier="real_data",
        domain=base.domain,
        classical_expr=base.classical_expr,
        classical_variables=variables,
        classical_constants=base.classical_constants,
        correction_type=base.correction_type,
        correction_expr=base.correction_expr,
        correction_constants={"theta_0": binary_pulsar_prefactor()},
        anomaly_regime=base.anomaly_regime,
        variables_with_units=units,
        classical_limit_variable="P",
        classical_limit_direction="oo",
        correction_class=base.correction_class,
        loader_variant=variant,
    )


def main():
    results = []
    print("=" * 72)
    print("  BINARY PULSAR SENSITIVITY STUDY (Mock Proposer, extended=True)")
    print("=" * 72)
    print(f"{'Variant':<12} {'Vars':<16} {'Class':^6} {'NMSE':>10} {'Discovered':<30}")
    print("-" * 72)

    for variant, variables, description in VARIANTS:
        scenario = make_variant_scenario(variant, variables)
        result = run_scenario_benchmark(
            scenario,
            noise_level=0.0,
            max_iter=4,
            proposer_type="mock",
            seed=42,
            extended=True,
        )
        result["variant"] = variant
        result["variables"] = variables
        result["description"] = description
        results.append(result)

        match = "OK" if result["class_match"] else "FAIL"
        print(
            f"{variant:<12} {','.join(variables):<16} [{match:^4}] "
            f"{result['nmse_full']:>10.3e} {result['discovered_expr'][:28]}"
        )

    out = "binary_pulsar_sensitivity.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
