"""Structure gate_E/gate_log from a v3 result json into boundary records.

Usage: python normalize_gate_e.py <result.json> [arm] [seed]
Emits multiseed-native-lm-compatible records:
  {"arm","seed","boundary","t","E":[...],"w":[...],"argmax":int}
and prints the 3-way reading tree (hard vs soft vs oracle onsets).
"""
import json
import sys
import numpy as np

LABELS = ['A1', 'B1', 'A2', 'B2', 'A3']
ETA = 5.0


def soft_w(E):
    e = np.asarray(E, float)
    z = -ETA * e / max(e.min(), 1e-8)
    w = np.exp(z - z.max())
    return (w / w.sum()).tolist()


def boundary_records(run, bounds, window=(0, 2, 5)):
    """records at each phase boundary + chunk offsets (400tok chunks)."""
    steps = np.asarray(run['curve_steps'])
    E = np.asarray(run['gate_E'])
    g = np.asarray(run['gate_log'])
    recs = []
    for i, b in enumerate(bounds[1:-1]):  # internal boundaries
        i0 = int(np.searchsorted(steps, b))
        for off in window:
            j = i0 + off
            if j >= len(steps):
                continue
            recs.append(dict(
                arm=run['arm'], seed=run['seed'], hard_gate=run.get('hard_gate'),
                boundary='%s->%s' % (LABELS[i], LABELS[i + 1]),
                t=int(steps[j]), E=E[j].tolist(),
                w=soft_w(E[j]), argmax=int(g[j])))
    return recs


def main(path, arm=None, seed=None):
    d = json.load(open(path))
    out = []
    table = {}
    for r in d['runs']:
        if arm and r['arm'] != arm:
            continue
        if seed is not None and r['seed'] != seed:
            continue
        bounds = d.get('bounds') or [0, 50000, 100000, 150000, 200000, 250000]
        out += boundary_records(r, bounds)
        table[(r['arm'], r['seed'])] = r['phases']
    print(json.dumps(out, indent=1) if '--json' in sys.argv else '')
    arms = sorted({a for a, _ in table})
    seeds = sorted({s for _, s in table})
    ons = ['onset_A2', 'onset_B2', 'onset_A3']
    for a in arms:
        for s in seeds:
            pm = table.get((a, s))
            if not pm:
                continue
            print('%-26s s%d  ' % (a, s) +
                  ' '.join('%s=%.3f' % (k[6:], pm[k] or -1) for k in ons))
    # reading tree
    for s in seeds:
        try:
            so = table[('apia_ctxmem_refuse', s)]
            ha = table[('apia_ctxmem_hardgate', s)]
            oc = table[('apia_ctxmem_oracle', s)]
        except KeyError:
            continue
        gap_o = np.mean([so[k] - oc[k] for k in ons])
        gap_h = np.mean([so[k] - ha[k] for k in ons])
        verdict = ('HARD~ORACLE (chain repaired)' if gap_h > 0.8 * gap_o else
                   'ORACLE>HARD>SOFT (dilution residual)' if gap_h > 1.2 * np.mean(
                       [0]) + 0.02 else 'HARD~SOFT (oscillation?)')
        print('seed %d: oracle_gain=%.3f hard_gain=%.3f -> %s'
              % (s, gap_o, gap_h, verdict))


if __name__ == '__main__':
    main(sys.argv[1],
         sys.argv[2] if len(sys.argv) > 2 else None,
         int(sys.argv[3]) if len(sys.argv) > 3 else None)
