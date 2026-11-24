# Gate-E diagnosis on the alternating A->B->A->B->A run.
# Answers: at each phase boundary, did the oracle-correct slot have
# the lowest E? If not, how long until it did (lag)?
import json, sys
import numpy as np

d = json.load(open(sys.argv[1] if len(sys.argv) > 1
                 else 'kaggle_gatediag_out/native_lm_v3_gatediag.json'))
r = [x for x in d['runs'] if x['arm'] == 'apia_ctxmem_refuse'][0]
E = np.array(r['gate_E'])          # (625, 3) E snapshot at end of each chunk
steps = np.array(r['curve_steps'])  # chunk-end positions (400, 800, ...)
gate = np.array(r['gate_log'])      # argmax slot used per chunk
cl = r['cons_log']

bounds = [(50000, 'A1->B1'), (100000, 'B1->A2'), (150000, 'A2->B2'),
          (200000, 'B2->A3')]
# oracle mapping: A segments -> slot0, B segments -> slot1
oracle_at = lambda t: 0 if (t // 50000) % 2 == 0 else 1

print('cons fills: step=%s target=%s' % (
    [s for s, t_, m in zip(cl['step'], cl['target'], cl['mode']) if m >= 0],
    [t_ for t_, m in zip(cl['target'], cl['mode']) if m >= 0]))
print('chunk granularity: %d tokens\n' % int(steps[0]))

for bnd, name in bounds:
    i0 = int(np.searchsorted(steps, bnd))
    print('=== boundary %s (t=%d) ===' % (name, bnd))
    # E ranking right at boundary and +4k tokens after
    for off in (0, 2, 5, 10, 20):
        i = i0 + off
        if i >= len(E):
            break
        e = E[i]
        correct = oracle_at(bnd + 1)
        rank_ok = int(np.argmin(e) == correct)
        z = -5.0 * e / max(e.min(), 1e-8)
        w = np.exp(z - z.max()); w /= w.sum()
        print(' t=%d | E=[%.3f %.3f %.3f] argmin=slot%d | gate_w=%s | oracle=slot%d %s'
              % (steps[i], e[0], e[1], e[2], int(np.argmin(e)),
                 np.round(w, 3), correct, 'RANK_OK' if rank_ok else 'RANK_WRONG'))
    # lag: first chunk after boundary where argmin(E) == oracle slot
    correct = oracle_at(bnd + 1)
    lag = None
    for i in range(i0, len(E)):
        if np.argmin(E[i]) == correct:
            lag = steps[i] - bnd
            break
    print(' -> correct slot first becomes argmin(E) at t+%s tokens\n' % lag)
