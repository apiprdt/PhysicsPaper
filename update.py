import sys

def replace_in_file(filepath, old_text, new_text):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if old_text in content:
        content = content.replace(old_text, new_text)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Replaced in {filepath}')
    else:
        print(f'Not found in {filepath}: {old_text}')

# 1. paper
replace_in_file('paper/neurips_paper.tex', 
    'evidence of blind discovery:', 
    'evidence of domain-guided discovery:')

replace_in_file('paper/neurips_paper.tex', 
    'Rank-1 blind-search candidate', 
    'Rank-1 domain-guided search candidate')

replace_in_file('paper/neurips_paper.tex', 
    'Rank-1 blind-search BIC;', 
    'Rank-1 domain-guided search BIC;')

replace_in_file('paper/neurips_paper.tex', 
    'achieved without any symbolic hint of the solution.', 
    'achieving a clean exact match through domain-guided search.')

replace_in_file('paper/neurips_paper.tex', 
    'Rank-1 candidate (blind search)', 
    'Rank-1 candidate (domain-guided search)')


# 2. python script
with open('src/adcd/run_adcd_v3_validation_blind.py', 'r', encoding='utf-8') as f:
    content = f.read()

# default argparse
content = content.replace(
    'parser.add_argument("--taxonomy", action="store_true", help="Use Domain Taxonomy Prior for Stage 1")',
    'parser.add_argument("--no-taxonomy", action="store_false", dest="taxonomy", help="Disable Domain Taxonomy Prior for Stage 1")\n    parser.set_defaults(taxonomy=True)'
)

# default function signature
content = content.replace(
    'def run_scenario_protocol(scenario, seed: int = 42, top_k_val: int = 5, use_taxonomy_prior: bool = False) -> ProtocolResult:',
    'def run_scenario_protocol(scenario, seed: int = 42, top_k_val: int = 5, use_taxonomy_prior: bool = True) -> ProtocolResult:'
)

# rename dictionary key
content = content.replace('"blind_search"', '"primary_search"')

# write back
with open('src/adcd/run_adcd_v3_validation_blind.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Updated python script')
