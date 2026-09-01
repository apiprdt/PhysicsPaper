import os

dirs_to_scan = ['src', 'ADCDEngine/src', 'eval', 'paper']
exts = ['.py', '.jl', '.tex']

with open('allcode.txt', 'w', encoding='utf-8') as outfile:
    for d in dirs_to_scan:
        for root, _, files in os.walk(d):
            for file in files:
                if any(file.endswith(ext) for ext in exts):
                    path = os.path.join(root, file)
                    outfile.write(f'\n\n{"="*80}\n')
                    outfile.write(f'FILE: {path}\n')
                    outfile.write(f'{"="*80}\n\n')
                    try:
                        with open(path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f'<Error reading file: {e}>\n')
print('Generated allcode.txt')
