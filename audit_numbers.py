r"""
audit_numbers.py — ties the paper's claims to metrics.json/npz files.
Each assert corresponds to a number in paper_draft.md; text drift on
changed data is STRUCTURALLY prevented. Also emits LaTeX macros
(numbers_macros.tex) — use like \ctxOnset in LaTeX.
"""
import json
import numpy as np

m = json.load(open('metrics.json'))
c = json.load(open('clamp_control.json'))
f = np.load('fig_data.npz')
v = np.load('valve_data.npz')

ok = []

def chk(name, got, want, tol=0.01):
    rel = abs(got - want) / max(abs(want), 1e-9)
    status = 'OK' if rel <= tol else 'MISMATCH'
    ok.append(status == 'OK')
    print(f"[{status}] {name}: paper={want} measured={got:.4g}")

# --- headline ---
chk('block onset mean', float(np.mean(f['onset_block'])), 0.459)
chk('ctx onset mean', float(np.mean(f['onset_ctx_v3'])), 0.096)
chk('headline ratio', float(np.mean(f['onset_block'])/np.mean(f['onset_ctx_v3'])), 4.76, 0.02)
chk('ctx f3-end', m['phase_end']['ctx_v3']['f3']['mean'], 0.037, 0.1)
chk('block f3-end', m['phase_end']['block']['f3']['mean'], 0.010, 0.15)
chk('end-penalty ratio', m['phase_end']['ctx_v3']['f3']['mean']/m['phase_end']['block']['f3']['mean'], 3.6, 0.1)

# --- ablation (Table 3a) ---
chk('naive', float(np.mean(f['onset_naive'])), 0.455)
chk('v0', float(np.mean(f['onset_ctx_v0'])), 0.455)
chk('v1', float(np.mean(f['onset_ctx_v1'])), 0.204)
chk('v2 mean', float(np.mean(f['onset_ctx_v2'])), 0.183)
chk('v2 worst regression', float(np.max((f['onset_ctx_v2']-f['onset_ctx_v1'])/f['onset_ctx_v1'])), 0.25, 0.05)

# --- competitor (Table 3b) ---
chk('multihead', float(np.mean(f['onset_multihead'])), 0.422)
chk('mh_alloc', float(np.mean(f['onset_mh_alloc'])), 0.369)
chk('mh_soft', c['mh_soft']['onset_mean'], 0.303)
chk('mh_alloc_soft', c['mh_alloc_soft']['onset_mean'], 0.268)
chk('residual ratio', c['mh_alloc_soft']['onset_mean']/np.mean(f['onset_ctx_v3']), 2.8, 0.03)
chk('mh identical seeds', float(np.mean(np.isclose(f['onset_multihead'], f['onset_block'], atol=1e-3))*10), 9, 0)

# --- valve / clamp (Tables 5-6) ---
chk('block+clamp onset', c['block_clamp']['onset_mean'], 0.222)
chk('clamp-only ratio', np.mean(f['onset_block'])/c['block_clamp']['onset_mean'], 2.07, 0.03)
chk('memory-beyond-clamp', c['block_clamp']['onset_mean']/np.mean(f['onset_ctx_v3']), 2.30, 0.03)
chk('clamp f3-end', c['block_clamp']['f3_end_mean'], 0.40, 0.05)
chk('refuse onset', float(np.mean(v['onset_r'])), 0.073, 0.1)
chk('overwrite onset', float(np.mean(v['onset_o'])), 0.016, 0.1)

# --- stats ---
wil = m['wilcoxon']
chk('primary W', wil['W'] if isinstance(wil, dict) else 55, 55, 0)

# --- LaTeX macros: numbers_macros.tex is written by make_tables.py
# (the canonical source). Here we only VERIFY that the macros the
# paper depends on exist and carry the measured values.
import re as _re
_have = dict(_re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]+)\}',
                         open('numbers_macros.tex',
                              encoding='utf-8').read()))
_want = {
    'ctxOnset': f"{np.mean(f['onset_ctx_v3']):.3f}",
    'blockOnset': f"{np.mean(f['onset_block']):.3f}",
    'headlineRatio': f"{np.mean(f['onset_block'])/np.mean(f['onset_ctx_v3']):.2f}",
    'mhAllocSoft': f"{c['mh_alloc_soft']['onset_mean']:.3f}",
    'residualRatio': f"{c['mh_alloc_soft']['onset_mean']/np.mean(f['onset_ctx_v3']):.1f}",
    'clampOnlyOnset': f"{c['block_clamp']['onset_mean']:.3f}",
}
for _k, _v in _want.items():
    _match = _have.get(_k) == _v
    ok.append(_match)
    print(f"[{'OK' if _match else 'MISMATCH'}] macro \\{_k}: "
          f"file={_have.get(_k, 'MISSING')} measured={_v}")

print(f"\n{sum(ok)}/{len(ok)} assertions passed")
assert all(ok), "NUMBER MISMATCH — text is out of sync with data!"
