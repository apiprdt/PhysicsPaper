"""Command-line interface for ADCD (Asymptotic Dictionary Correction Discovery)."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from adcd import __version__
from adcd.anomaly_scenarios import get_all_scenarios
from adcd.asymptotic_dictionary_proposer_v3 import PRIMITIVE_REGISTRY
from adcd.quickfit import DOMAIN_TAXONOMY


def _cmd_list_scenarios() -> int:
    scenarios = get_all_scenarios()
    print(f"{'Scenario Name':<28} {'Domain Key':<30} {'Limit':<12} {'Baseline'}")
    print("-" * 95)
    for sc in scenarios:
        lim = f"{sc.classical_limit_variable}->{sc.classical_limit_direction}"
        print(f"{sc.name:<28} {sc.domain:<30} {lim:<12} {sc.classical_expr}")
    return 0


def _cmd_list_taxonomy() -> int:
    print(f"{'Phenomenon Taxonomy Key':<34} {'Permitted Primitives'}")
    print("-" * 70)
    for key, prims in DOMAIN_TAXONOMY.items():
        print(f"{key:<34} {', '.join(prims)}")
    return 0


def _cmd_list_primitives() -> int:
    print(f"{'Primitive':<14} {'Cost':<6} {'Domain Note':<28} {'Template D(u)'}")
    print("-" * 95)
    for name, prim in PRIMITIVE_REGISTRY.items():
        tmpl = prim.string_template.format(u="u")
        print(f"{name:<14} {prim.token_cost:<6} {prim.domain_note:<28} {tmpl}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="adcd",
        description="ADCD: Asymptotic Dictionary Correction Discovery CLI",
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"adcd {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # adcd validate
    val_parser = subparsers.add_parser(
        "validate", help="Run the 4-step ADCD validation protocol"
    )
    val_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of Pareto front candidates"
    )
    val_parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run a single scenario by name (default: all canonical scenarios)",
    )
    val_parser.add_argument(
        "--no-taxonomy",
        action="store_false",
        dest="taxonomy",
        help="Disable Domain Taxonomy Prior for Stage 1",
    )
    val_parser.set_defaults(taxonomy=True)

    # adcd list-scenarios
    subparsers.add_parser(
        "list-scenarios", help="List built-in physics benchmark scenarios"
    )

    # adcd list-taxonomy
    subparsers.add_parser(
        "list-taxonomy", help="List Phenomenon-Specific Taxonomy keys and priors"
    )

    # adcd list-primitives
    subparsers.add_parser(
        "list-primitives", help="List algebraically regularized primitives"
    )

    args = parser.parse_args(argv)

    if args.command == "list-scenarios":
        return _cmd_list_scenarios()
    if args.command == "list-taxonomy":
        return _cmd_list_taxonomy()
    if args.command == "list-primitives":
        return _cmd_list_primitives()
    if args.command == "validate":
        from adcd.run_adcd_v3_validation_blind import main as run_val_main

        fwd_args = ["--top-k", str(args.top_k)]
        if args.scenario:
            fwd_args.extend(["--scenario", args.scenario])
        if not args.taxonomy:
            fwd_args.append("--no-taxonomy")
        sys.argv = ["adcd validate"] + fwd_args
        run_val_main()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
