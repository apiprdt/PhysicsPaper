"""
ADCD CLI (Command-Line Interface).

Provides clean CLI commands:
  - adcd discover: Run correction discovery on custom or predefined scenarios.
  - adcd fit: Fit physical correction terms on local CSV/JSON datasets.
  - adcd run-experiment: Run Muon g-2 or SPARC physical experiments.
"""

import sys
import argparse
import json
from adcd.api import fit
from adcd.experiments.muon_g2_validation import run_validation_demo
from adcd.experiments.sparc_stacking import run_sparc_discovery

def cmd_fit(args):
    # Data ingestion for custom local data
    print(f"Loading dataset from: {args.data}")
    
    if args.data.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(args.data)
    elif args.data.endswith(".json"):
        df = pd.read_json(args.data)
    else:
        print("Error: Dataset must be a CSV or JSON file.")
        sys.exit(1)
        
    # Ingest inputs
    if args.y_obs not in df.columns or args.y_classical not in df.columns:
        print(f"Error: Columns '{args.y_obs}' and '{args.y_classical}' must exist in the dataset.")
        sys.exit(1)
        
    y_obs = df[args.y_obs].values
    y_classical = df[args.y_classical].values
    
    # Ingest X variables
    x_cols = [c for c in df.columns if c not in [args.y_obs, args.y_classical]]
    if args.x_vars:
        x_cols = [v.strip() for v in args.x_vars.split(",")]
        
    X = {col: df[col].values for col in x_cols}
    
    print(f"Ingested X variables: {list(X.keys())}")
    print(f"Ingested data points: {len(y_obs)}")
    
    res = fit(
        X=X,
        y_obs=y_obs,
        y_classical=y_classical,
        limit_variable=args.limit_var,
        limit_direction=args.limit_dir,
        classical_expr=args.classical_expr,
        proposer="mock",
        verbose=True
    )
    
    print("\n=== ADCD Fit Results ===")
    print(f"Discovered Correction: {res.search_result.best_expr}")
    print(f"Fitted Parameters: {res.search_result.best_theta}")
    print(f"Residual NMSE: {res.search_result.best_nmse_residual:.6f}")
    
    if args.output:
        out_data = {
            "best_expr": res.search_result.best_expr,
            "best_theta": res.search_result.best_theta,
            "nmse_residual": float(res.search_result.best_nmse_residual),
            "nmse_full": float(res.search_result.best_nmse_full),
            "converged": bool(res.search_result.converged)
        }
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Results saved to: {args.output}")

def cmd_experiment(args):
    if args.name == "muon-g2":
        run_validation_demo()
    elif args.name == "sparc":
        run_sparc_discovery()
    else:
        print(f"Unknown experiment: {args.name}. Available: muon-g2, sparc")

def main():
    parser = argparse.ArgumentParser(
        prog="adcd",
        description="ADCD: Anomaly-Driven Correction Discovery"
    )
    subparsers = parser.add_subparsers(dest="command")
    
    # Command: fit
    parser_fit = subparsers.add_parser("fit", help="Fit physical correction on local dataset")
    parser_fit.add_argument("--data", required=True, help="Path to CSV/JSON dataset file")
    parser_fit.add_argument("--y-obs", required=True, help="Column name of observed outputs")
    parser_fit.add_argument("--y-classical", required=True, help="Column name of classical baseline")
    parser_fit.add_argument("--x-vars", help="Comma-separated column names for independent variables (defaults to all other columns)")
    parser_fit.add_argument("--limit-var", help="Variable governing asymptotic classical limit")
    parser_fit.add_argument("--limit-dir", default="0", help="Classical limit direction (0 or oo)")
    parser_fit.add_argument("--classical-expr", default="0", help="Formula representation of classical baseline")
    parser_fit.add_argument("--output", help="Save output JSON summary to path")
    
    # Command: experiment
    parser_exp = subparsers.add_parser("experiment", help="Run predefined physical validation/discovery experiments")
    parser_exp.add_argument("name", choices=["muon-g2", "sparc"], help="Name of experiment")
    
    args = parser.parse_args()
    
    if args.command == "fit":
        cmd_fit(args)
    elif args.command == "experiment":
        cmd_experiment(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
