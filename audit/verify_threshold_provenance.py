import subprocess
import sympy as sp


def get_depth(expr: sp.Expr) -> int:
    if not expr.args:
        return 1
    return 1 + max(get_depth(a) for a in expr.args)


def get_tokens(expr: sp.Expr) -> int:
    return len(list(sp.preorder_traversal(expr)))


def q1_find_default_introduction():
    """When did max_depth=7, max_tokens=25 first appear as ASTValidator defaults?"""
    print("=" * 70)
    print("Q1: git history of ASTValidator default values")
    print("=" * 70)
    try:
        out = subprocess.run(
            ["git", "log", "-p", "-S", "max_depth: int = 7", "--",
             "src/adcd/dimensional_checker.py"],
            capture_output=True, text=True, check=False,
        )
        print(out.stdout[:4000] if out.stdout else "(no matches -- try a different search string)")
        if out.stderr:
            print("stderr:", out.stderr[:500])
    except FileNotFoundError:
        print("git not available in this environment -- run manually and paste output.")


def q2_check_if_leaky_function_was_ever_called():
    """Search full history for ANY call site of set_threshold_relative_to,
    not just its definition -- a defined-but-never-called function cannot
    have contaminated the shipped defaults."""
    print("\n" + "=" * 70)
    print("Q2: was set_threshold_relative_to() ever actually CALLED anywhere?")
    print("=" * 70)
    try:
        out = subprocess.run(
            ["git", "log", "-p", "-S", "set_threshold_relative_to("],
            capture_output=True, text=True, check=False,
        )
        # Filter to lines that look like a CALL (has an opening paren with an
        # argument), not the `def` line itself
        call_lines = [
            line for line in out.stdout.splitlines()
            if "set_threshold_relative_to(" in line
            and "def set_threshold_relative_to" not in line
        ]
        if call_lines:
            print(f"Found {len(call_lines)} non-definition occurrence(s):")
            for line in call_lines[:20]:
                print(" ", line.strip())
        else:
            print(
                "No call sites found anywhere in history -- this function "
                "appears to have been DEFINED but NEVER CALLED. If true, it "
                "could not have contaminated the shipped default values; "
                "it was dead code, still worth removing (which was already "
                "done), but not proof of an actual leak into the numbers "
                "used in any reported result."
            )
    except FileNotFoundError:
        print("git not available -- run manually.")


def q3_check_coincidence_with_scenarios():
    """Do 7/25 match target_depth+2 / target_tokens+5 for the project's
    actual scenario ground truths? Checked independently of git history --
    this is a pure math check anyone can rerun."""
    print("\n" + "=" * 70)
    print("Q3: does 7/25 match (ground_truth_depth+2)/(ground_truth_tokens+5) "
          "for actual scenarios?")
    print("=" * 70)
    try:
        from adcd.anomaly_scenarios import get_all_scenarios
    except ImportError:
        print("Could not import anomaly_scenarios -- run with PYTHONPATH=src.")
        return

    matches = []
    for s in get_all_scenarios():
        try:
            expr = sp.sympify(s.correction_expr)
            d = get_depth(expr)
            t = get_tokens(expr)
            implied_depth = d + 2
            implied_tokens = t + 5
            is_match = (implied_depth == 7 and implied_tokens == 25)
            print(
                f"{s.name:<40} depth={d:<3} tokens={t:<3} "
                f"-> implied_max_depth={implied_depth:<3} "
                f"implied_max_tokens={implied_tokens:<3}  "
                f"{'** MATCHES SHIPPED DEFAULTS **' if is_match else ''}"
            )
            if is_match:
                matches.append(s.name)
        except Exception as e:
            print(f"{s.name:<40} (could not parse: {e})")

    print()
    if len(matches) == 0:
        print(
            "CONCLUSION: no scenario's ground truth, run through the leaky "
            "formula, produces exactly 7/25. This weakens the 'defaults were "
            "derived from this function' theory -- more likely 7/25 were "
            "chosen independently (e.g. round numbers, or tuned against a "
            "DIFFERENT scenario not in the current set, or just a generic "
            "engineering choice)."
        )
    elif len(matches) == 1:
        print(
            f"CONCLUSION: exactly one scenario ({matches[0]}) matches. "
            f"Plausible this WAS the scenario used to set the leaky "
            f"defaults during development -- worth specifically disclosing "
            f"in the paper as a limitation for that scenario, without "
            f"necessarily implying ALL scenarios' thresholds are tainted."
        )
    else:
        print(
            f"CONCLUSION: {len(matches)} scenarios match simultaneously "
            f"({matches}). This is much harder to explain as coincidence -- "
            f"strengthens the case that these ARE the scenarios the "
            f"threshold was originally tuned against."
        )


if __name__ == "__main__":
    q1_find_default_introduction()
    q2_check_if_leaky_function_was_ever_called()
    q3_check_coincidence_with_scenarios()
