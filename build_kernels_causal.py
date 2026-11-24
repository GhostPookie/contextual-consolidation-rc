"""Assemble two Kaggle script kernels for the post-leakfix rerun.

kaggle_causal/  -> 'apia-v3-causal': native-LM 250k A-B-A-B-A grid with
                   the causal gate ordering (se_buf fix). All 6 arms x
                   3 seeds, checkpoint-resumable.
kaggle_p1/      -> 'apia-p1-battery': paper-1 saturation/valve/clamp
                   tests with fixed ContextualMemory ordering.
"""
import json
import os

# ---------------- kernel B: apia-v3-causal ----------------
os.makedirs('kaggle_causal', exist_ok=True)
old = open('kaggle_kernel/script.py', encoding='utf-8').read().splitlines(True)
chunk = open('_chunked.py', encoding='utf-8').read().splitlines(True)

head = old[:1789]          # through end of _nb_block_c (line 1790 = blank)
ctxmem = chunk[85:307]     # @njit _nb_ctxmem_c .. _FSTEP_PAD (lines 86-307)
runner = old[2007:2090]    # def run_arm_throttled .. return r (2008-2090)

driver = '''

# ---------- causal-ordering re-verification grid: A->B->A->B->A ----------
OUT3 = '/kaggle/working/native_lm_v3_causal.json'
res3 = json.load(open(OUT3)) if os.path.exists(OUT3) else {}
res3.setdefault('runs', [])
stream, vocab = load_chars()
stream = stream[:250001]
sh = np.concatenate([np.ascontiguousarray(stream[0:50000]),
                     np.ascontiguousarray(stream[50000:100000][::-1]),
                     np.ascontiguousarray(stream[100000:150000]),
                     np.ascontiguousarray(stream[150000:200000][::-1]),
                     np.ascontiguousarray(stream[200000:250000])])
bounds = [0, 50000, 100000, 150000, 200000, 250000]
labels = ['A1', 'B1', 'A2', 'B2', 'A3']
print('v3-CAUSAL-250K stream:', len(sh), 'bounds', bounds, flush=True)


def alt_phase_metrics(r, bounds, labels, win=2000):
    out = {}
    steps, curve = r['curve_steps'], r['bpc_curve']
    for i in range(len(labels)):
        lo, hi = bounds[i], bounds[i + 1]
        on = [c for s, c in zip(steps, curve) if lo < s <= lo + win]
        en = [c for s, c in zip(steps, curve) if hi - win < s <= hi]
        out['onset_' + labels[i]] = float(np.mean(on)) if on else None
        out['end_' + labels[i]] = float(np.mean(en)) if en else None
    return out


print('warming numba...', flush=True)
_wa = _apia_init(vocab, 512, 10, 0)[2]
_wh = np.zeros(512)
_wt = np.ascontiguousarray(np.arange(8, dtype=np.int64) % vocab)
_nb_block_c(_wt, _wa['W_in'], _wa['Wd'], _wa['Wi'], _wa['Wp'], _wa['alpha'],
            _wa['knots'], _wa['degree'], _wa['K'], _wa['W'], _wa['Pb'],
            _wa['Pbias'], _wa['lam'], _wa['trace_bound'], _wa['bias_bound'],
            _wh, vocab)
time.sleep(0.5)
_wS = np.zeros((3,) + _wa['W'].shape)
_wE = np.zeros(3)
_wcc = np.zeros(3, np.int64)
_wst = np.zeros(6)
_wcl = np.empty(8, np.int64)
_wcl2 = np.empty(8, np.int64)
_wcl3 = np.empty(8, np.int64)
_wncl = np.zeros(1, np.int64)
_nb_ctxmem_c(_wt, _wa['W_in'], _wa['Wd'], _wa['Wi'], _wa['Wp'], _wa['alpha'],
             _wa['knots'], _wa['degree'], _wa['K'], _wa['W'], _wa['Pb'],
             _wa['Pbias'], _wa['lam'], _wa['trace_bound'], _wa['bias_bound'],
             _wS, _wE, 1.0, 0.1, 5.0, 0.05, 0.1, 5000, 1.0, 0.5, 1.5,
             _wcc, _wst, _wcl, _wcl2, _wcl3, _wncl, _wh, vocab, 0, -1, 0, _FSTEP_PAD, 0)
_nb_ctxmem_c(_wt, _wa['W_in'], _wa['Wd'], _wa['Wi'], _wa['Wp'], _wa['alpha'],
             _wa['knots'], _wa['degree'], _wa['K'], _wa['W'], _wa['Pb'],
             _wa['Pbias'], _wa['lam'], _wa['trace_bound'], _wa['bias_bound'],
             _wS, _wE, 1.0, 0.1, 5.0, 0.05, 0.1, 5000, 1.0, 0.5, 1.5,
             _wcc, _wst, _wcl, _wcl2, _wcl3, _wncl, _wh, vocab, 0, 0, 1, _FSTEP_PAD, 0)
print('numba warm', flush=True)
for arm in ['apia_block', 'apia_ctxmem', 'apia_ctxmem_hardgate',
            'apia_ctxmem_oracle', 'apia_ctxmem_forced',
            'apia_ctxmem_detect']:
    for seed in [0, 1, 2]:
        if any(r['arm'] == arm and r['seed'] == seed for r in res3['runs']):
            print('skip', arm, seed, flush=True)
            continue
        print('==', arm, 'seed', seed, flush=True)
        r = run_arm_throttled(arm, sh, vocab, seed=seed, cons_every=5000,
                              seg_bounds=bounds)
        r['phases'] = alt_phase_metrics(r, bounds, labels)
        res3['runs'].append(r)
        json.dump(res3, open(OUT3, 'w'), indent=1)
        print('__V3ARM__' + json.dumps(r), flush=True)
        pm = r['phases']
        print('   -> ' + '  '.join('%s.on=%.3f %s.en=%.3f' % (
            l, pm['onset_' + l] or -1, l, pm['end_' + l] or -1)
            for l in labels) + '  cons=%d' % r.get('n_consolidations', 0),
            flush=True)
print('CAUSAL_DONE ->', OUT3, flush=True)
'''

code = ''.join(head) + ''.join(ctxmem) + ''.join(runner) + driver
open('kaggle_causal/script.py', 'w', encoding='utf-8', newline='\n').write(code)
json.dump({
    "id": "ghostsa/apia-v3-causal",
    "title": "apia-v3-causal",
    "code_file": "script.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []},
    open('kaggle_causal/kernel-metadata.json', 'w'), indent=1)
print('kaggle_causal/script.py:', len(code.splitlines()), 'lines,',
      len(code), 'bytes')

# ---------------- kernel A: apia-p1-battery ----------------
os.makedirs('kaggle_p1', exist_ok=True)
MODS = ['reservoir.py', 'rls_engine.py', 'memory.py', 'multihead.py',
        'contextual_memory.py', 'continual_protocol.py']
TESTS = ['saturation_test.py', 'valve_test.py', 'clamp_control.py']

out = ['''# apia-p1-battery - post-leakfix paper-1 rerun on Kaggle CPU.
# Repo modules registered in sys.modules (causal-ordering fix) +
# saturation/valve/clamp drivers in isolated namespaces.
# Outputs -> /kaggle/working.
import json
import os
import sys
import types
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _mkmod(name, src):
    m = types.ModuleType(name)
    m.__file__ = name + '.py'
    exec(compile(src, m.__file__, 'exec'), m.__dict__)
    sys.modules[name] = m
    return m


''']
for m in MODS:
    src = open(m, encoding='utf-8').read().splitlines(True)
    cut = len(src)
    for i, ln in enumerate(src):
        if ln.startswith('if __name__'):
            cut = i
            break
    var = m.replace('.py', '').upper() + '_SRC'
    out.append('%s = %r\n' % (var, ''.join(src[:cut])))
    out.append("_mkmod('%s', %s)\n\n" % (m[:-3], var))

out.append('''
# ============ test drivers (isolated namespaces) ============
''')
for t in TESTS:
    src = open(t, encoding='utf-8').read()
    var = t.replace('.py', '').upper() + '_SRC'
    out.append('%s = %r\n' % (var, src))

out.append('''
os.chdir('/kaggle/working') if os.path.isdir('/kaggle/working') else None
for name, src in [('saturation', SATURATION_TEST_SRC),
                  ('valve', VALVE_TEST_SRC),
                  ('clamp', CLAMP_CONTROL_SRC)]:
    print('\\\\n===== RUN %s =====' % name, flush=True)
    g = {'__name__': '__main__'}
    exec(compile(src, name + '.py', 'exec'), g)
    print('===== DONE %s =====' % name, flush=True)
print('P1_BATTERY_DONE', flush=True)
''')

code2 = ''.join(out)
open('kaggle_p1/script.py', 'w', encoding='utf-8', newline='\n').write(code2)
json.dump({
    "id": "ghostsa/apia-p1-battery",
    "title": "apia-p1-battery",
    "code_file": "script.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []},
    open('kaggle_p1/kernel-metadata.json', 'w'), indent=1)
print('kaggle_p1/script.py:', len(code2.splitlines()), 'lines,',
      len(code2), 'bytes')
