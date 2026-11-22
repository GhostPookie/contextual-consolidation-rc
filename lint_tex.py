import re
import sys
from collections import Counter

t = open('paper.tex', encoding='utf-8').read()
begins = Counter(re.findall(r'\\begin\{(\w+)\}', t))
ends = Counter(re.findall(r'\\end\{(\w+)\}', t))
out = []
for env in set(begins) | set(ends):
    if begins[env] != ends[env]:
        out.append(f'UNBALANCED {env}: {begins[env]}v{ends[env]}')
depth, ln = 0, 1
for ch in t:
    if ch == '\n':
        ln += 1
    elif ch == '{':
        depth += 1
    elif ch == '}':
        depth -= 1
        if depth < 0:
            out.append('extra } @L' + str(ln))
            depth = 0
if depth:
    out.append('unclosed { x' + str(depth))
out.append('envs: ' + str(dict(begins)))
sys.stdout.buffer.write('\n'.join(out).encode('utf-8', 'replace'))
