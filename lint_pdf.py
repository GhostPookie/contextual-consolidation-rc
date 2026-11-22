# lint_pdf.py — render-level audit: paper.pdf + build output.
# Run: python lint_pdf.py   (paper.pdf built; pdftotext + tectonic on PATH)
import re
import subprocess
import sys

F = []

def check(ok, msg):
    print(('PASS ' if ok else 'FAIL ') + msg)
    if not ok:
        F.append(msg)

# ---- 1) text extraction ----
txt = subprocess.run(['pdftotext', '-enc', 'UTF-8', 'paper.pdf', '-'],
                     capture_output=True, text=True, encoding='utf-8',
                     errors='replace').stdout
norm = re.sub(r'\s+', ' ', txt)
check(len(txt) > 30000, f'pdftotext extraction ({len(txt)} chars)')

# ---- 2) captions exactly once (alphanumeric-normalized) ----
def alnum(s):
    return re.sub(r'[^a-z0-9]+', '', s.lower())

tex = open('paper.tex', encoding='utf-8').read()
caps = re.findall(r'\\caption\{(.*?)\}\s*\n?\\label', tex, re.S)
norm_a = alnum(norm)
missing, dup = [], []
for c in caps:
    plain = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^]]*\])?', '', c)
    needle = alnum(plain[:40])
    if len(needle) < 15:
        continue
    n = norm_a.count(needle)
    if n == 0:
        missing.append(plain[:40].strip())
    elif n > 1:
        dup.append(f'{n}x {plain[:40].strip()}')
check(not missing, f'{len(caps)} captions found in PDF (missing: {missing})')
check(not dup, f'no caption duplicates (dup: {dup})')

# ---- 3) caption tails: tail text appears only inside caption (exactly 1x) ----
TAILS = ['F=1001). λ=1', 'F3-onset= NMSE over', '(laddered, not factorial);',
         'baseline to printed precision', 'cap alone recovers',
         'are representative values', 'the residual gap',
         'improves on the baseline in 10/10 and on v2',
         'the ungated fast path']
bad = [t for t in TAILS if norm_a.count(alnum(t)) != 1]
check(not bad, f'caption-tail exactly-once rule (broken: {bad})')

# ---- 4) C1 control characters + broken glyphs ----
c1 = [f'U+{ord(c):04X}' for c in txt if 0x80 <= ord(c) <= 0x9F]
check(not c1, f'no C1 control characters ({set(c1)})')

# ---- 5) leftover tokens ----
check(r'\*' not in txt, 'no literal \\*')
check('??' not in txt, 'no unresolved ?? refs')
check('\\ref' not in txt, 'no \\ref leftovers')
check('textasciitilde' not in txt, 'no textasciitilde leftovers')
check(not re.search(r'(Table|Fig\.?|Sec\.?|Tables|Figs\.?)~\S', txt),
      'no literal ~ refs (Table~/Fig.~/Sec.~)')
check(not re.search(r'[0-9a-z]\*', txt) or 'marked*' not in txt.replace(' ', ''),
      'no literal * in prose')
check('Seventh,' not in txt, 'no Seventh numbering')
check('Sixth,' in txt, 'Sixth limitation present')
check('Index Terms' not in txt, 'no Index Terms (IEEE leftover)')
check('Keywords' in txt, 'Keywords present')
check('0.456' not in txt, 'stale 0.456 absent')
check('0.455' in txt, 'naiveOnset 0.455 rendered')
check(re.search(r'2\.3e[-−]?0?5', txt), 'FullRLS F3-end 2.3e-05 rendered')

# ---- 6) do macro values appear in the PDF ----
mac = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]+)\}',
                      open('numbers_macros.tex', encoding='utf-8').read()))
flat = norm.replace('×', 'x').replace('−', '-')
absent = [k for k, v in mac.items() if v.replace('x', 'x') not in flat
          and v not in flat]
check(not absent, f'all macro values rendered (absent: {absent})')

# ---- 7) overfull hbox > 20pt -> error ----
import glob
TEC = glob.glob(r'C:\Users\*\AppData\Local\Temp\tectonic_bin\tectonic.exe')
TEC = TEC[0] if TEC else 'tectonic'
log = subprocess.run([TEC, 'paper.tex'], capture_output=True,
                     text=True, encoding='utf-8', errors='replace').stderr
over = [float(m) for m in
        re.findall(r'Overfull \\hbox \(([\d.]+)pt', log)]
check(max(over or [0]) <= 20,
      f'overfull hbox <=20pt (max={max(over or [0]):.1f}pt)')
check('Missing character' not in log, 'no Missing character (log)')
check('Undefined' not in log and 'Error' not in log,
      'no undefined/error (log)')

print(f"\n{'='*40}\n{len(F)} FAIL" if F else f"\n{'='*40}\nALL CHECKS PASSED")
sys.exit(1 if F else 0)
