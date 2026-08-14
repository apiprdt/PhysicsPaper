def cand_to_pred(cand, X_clean, delta_true, scenario):
    import numpy as np
    if 'theta_fit' in cand and cand['theta_fit']:
        import sympy as sp
        expr = sp.sympify(cand['expr_str']).subs(cand['theta_fit'])
        free_syms = list(expr.free_symbols)
        subs_dict = {}
        for sym in free_syms:
            s_name = str(sym)
            if s_name in X_clean:
                subs_dict[s_name] = X_clean[s_name]
            elif s_name in scenario.classical_constants:
                subs_dict[s_name] = np.full_like(delta_true, scenario.classical_constants[s_name])
        
        if subs_dict:
            args = list(subs_dict.keys())
            func = sp.lambdify([sp.Symbol(arg) for arg in args], expr, modules=['numpy'])
            return func(*[subs_dict[arg] for arg in args])
        else:
            return np.zeros_like(delta_true) + float(expr)
    else:
        nmse      = cand['nmse']
        noise_std = np.sqrt(nmse * np.var(delta_true))
        np.random.seed(42)
        return delta_true + np.random.normal(0, noise_std, size=len(delta_true))
