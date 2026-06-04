import json

# ====== ABLATION AUDIT ======
with open('ablation_results.json') as f:
    abl = json.load(f)

conditions = {}
for r in abl:
    c = r['condition']
    if c not in conditions:
        conditions[c] = []
    conditions[c].append(r['class_match'])

print('=== ABLATION RESULTS (5% noise) ===')
for cond, matches in conditions.items():
    n = len(matches)
    hits = sum(matches)
    print(f'  {cond}: {hits}/{n} = {100*hits/n:.1f}%')

full = sum(conditions['Full_ADCD'])
no_gates = sum(conditions['No_Gates'])
print(f'  => Drop from Full_ADCD to No_Gates: {(full - no_gates)/9*100:.1f} pp')
print()

# ====== MLP BASELINE AUDIT ======
with open('mlp_baseline_results.json') as f:
    mlp = json.load(f)

print(f'  MLP keys sample: {list(mlp[0].keys())}')
noise0 = [r for r in mlp if r['noise'] == 0.0]
noise5 = [r for r in mlp if r['noise'] == 0.05]
print(f'  MLP 0% noise records: {len(noise0)}, 5% records: {len(noise5)}')
if noise0:
    avg_nmse_0 = sum(r['nmse_full'] for r in noise0) / len(noise0)
    print(f'  0% noise avg NMSE: {avg_nmse_0:.3e}  (paper claims 8.56e-5)')
if noise5:
    avg_nmse_5 = sum(r['nmse_full'] for r in noise5) / len(noise5)
    print(f'  5% noise avg NMSE: {avg_nmse_5:.3e}  (paper claims 8.01e-3)')
print()

# ====== ADCD AVERAGE NMSE AUDIT ======
with open('reproducibility_results.json') as f:
    rep = json.load(f)

seed42 = [r for r in rep if r.get('seed', 42) == 42]
adcd_0 = [r for r in seed42 if r['noise'] == 0.0]
adcd_5 = [r for r in seed42 if r['noise'] == 0.05]
adcd_avg_0 = sum(r['nmse_full'] for r in adcd_0) / max(len(adcd_0), 1)
adcd_avg_5 = sum(r['nmse_full'] for r in adcd_5) / max(len(adcd_5), 1)
print('=== ADCD NMSE (reproducibility_results.json, seed=42) ===')
print(f'  0% noise avg NMSE: {adcd_avg_0:.3e}  (paper claims 5.51e-12)')
print(f'  5% noise avg NMSE: {adcd_avg_5:.3e}  (paper claims 4.56e-3)')
print()

# ====== OVERALL ACCURACY AUDIT ======
print('=== OVERALL ACCURACY AUDIT (seed=42) ===')
for noise_pct in [0.0, 0.01, 0.05, 0.1]:
    rows = [r for r in seed42 if r['noise'] == noise_pct]
    hits = sum(r['class_match'] for r in rows)
    n = len(rows)
    print(f'  Noise={int(round(noise_pct*100))}%: {hits}/{n} = {100*hits/n:.1f}%')
total = sum(r['class_match'] for r in seed42)
total_n = len(seed42)
print(f'  TOTAL: {total}/{total_n} = {100*total/total_n:.1f}%  (paper claims 94.4% = 34/36)')
print()

# ====== PYSR AUDIT ======
with open('pysr_baseline_results.json') as f:
    pysr = json.load(f)

print('=== PYSR ACCURACY AUDIT ===')
paper_pysr = {0: '22.2%', 1: '66.7%', 5: '44.4%', 10: '44.4%'}
for noise_pct in [0.0, 0.01, 0.05, 0.1]:
    rows = [r for r in pysr if r['noise'] == noise_pct]
    hits = sum(r['class_match'] for r in rows)
    n = len(rows)
    key = int(round(noise_pct*100))
    claimed = paper_pysr.get(key, '?')
    print(f'  Noise={key}%: {hits}/{n} = {100*hits/n:.1f}%  | Paper claims: {claimed}')
print()

# ====== SCAN FOR SPECIFIC PROBLEMATIC CLAIMS ======
print('=== ABLATION: Does removing any single gate change the score? ===')
baseline_hits = sum(conditions['Full_ADCD'])
for cond in ['No_ARC', 'No_AST', 'No_Dim', 'No_DataGate']:
    hits = sum(conditions[cond])
    diff = hits - baseline_hits
    print(f'  {cond}: {hits}/9  delta={diff:+d}  ({"no change" if diff == 0 else "CHANGED"})')
