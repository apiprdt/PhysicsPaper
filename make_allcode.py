import os

output_file = 'allcode.txt'
exclude_dirs = {'.venv', '.git', '__pycache__', 'figures', 'generated'}
# Extensions to include, can add more if needed (e.g. .md, .tex)
include_exts = {'.py', '.ps1'}

with open(output_file, 'w', encoding='utf-8') as out:
    out.write('# ADCD — All Core Source Code\n\n')
    
    # Walk starting from root directory
    for root, dirs, files in os.walk('.'):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for f in files:
            # Skip the script itself and the output file
            if f in [output_file, 'make_allcode.py', 'strip_docs.py', 'strip_docs2.py', 'strip_docs3.py', 'fix_flake.py']:
                continue
                
            if any(f.endswith(ext) for ext in include_exts):
                path = os.path.join(root, f)
                
                # Format path nicely
                clean_path = path.replace('.\\', '').replace('\\', '/')
                
                out.write('='*80 + '\n')
                out.write(f'FILE: {clean_path}\n')
                out.write('='*80 + '\n')
                try:
                    with open(path, 'r', encoding='utf-8') as infile:
                        out.write(infile.read())
                except Exception as e:
                    out.write(f'<< Error reading file: {e} >>\n')
                out.write('\n\n')

print(f'{output_file} generated successfully, containing all codebase files.')
