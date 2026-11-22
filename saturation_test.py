"""
APIA 2.0 - Saturation-policy mini-ablation
==============================================================
4 regimes: sine -> square -> triangle -> sine  (S in {2,3}).

Scenario: square-phase fragmentation at S=3 + a triangle phase = a
NEW context arrives while all slots are populated. Policy:
  'overwrite' -> writes over argmin c_i (least-consolidated).
  'refuse'    -> refuses the write; stored contexts are preserved.

Primary measure: F4-onset NMSE (sine return, first 250 steps) — did
the stored sine memory survive past saturation?
Secondary: eviction (mode=3) / refusal (mode=2) counters.
"""

import numpy as np

from rls_engine import SplineFeatureMap, BlockDiagRLS
from contextual_memory import ContextualMemory
from continual_protocol import (build_stream, make_stream_states,
                                D, K, WASHOUT, LAM, INIT_VAR,
                                WINDUP_MULT, WINDOW)

SEEDS = 10
PHASES = ['sine', 'square', 'triangle', 'sine']
PH_LEN = 1500
T_POST = PH_LEN * len(PHASES)


def make_model(kind):
    base = lambda: BlockDiagRLS(D, K, 1, forgetting=LAM,
                                init_var=INIT_VAR,
                                windup_mult=WINDUP_MULT)
    if kind == 'block':
        return base()
    S, pol = kind[0], kind[1]      # '2o', '2r', '3o', '3r'
    return ContextualMemory(base(), n_slots=int(S), beta=1.0,
                            gamma=0.1, eta=2.0,
                            consolidate_every=250, e_gate=0.5,
                            p_inflate=1.5,
                            saturation='refuse' if pol == 'r'
                            else 'overwrite')


def run(kind, Phi, Y):
    m = make_model(kind)
    errs = np.empty(len(Phi))
    for i in range(len(Phi)):
        errs[i] = float(np.sum(m.update(Phi[i], Y[i]) ** 2))
    return errs, m


def main():
    kinds = ['block', '2o', '2r', '3o', '3r']
    labels = {'block': 'Block-RLS', '2o': 'S=2 overwrite',
              '2r': 'S=2 refuse', '3o': 'S=3 overwrite',
              '3r': 'S=3 refuse'}
    onset, evict, refuse_n = {k: [] for k in kinds}, \
                             {k: [] for k in kinds[1:]}, \
                             {k: [] for k in kinds[1:]}
    fmap = SplineFeatureMap(D, K, 3)

    for seed in range(SEEDS):
        u_all, y_all = build_stream(seed, phases=PHASES,
                                    phase_len=PH_LEN)
        sig_var = float(np.var(y_all))
        Phi = make_stream_states(u_all, seed, fmap)
        Y = y_all[WASHOUT:].reshape(-1, 1)
        lo = 3 * PH_LEN
        for k in kinds:
            errs, m = run(k, Phi, Y)
            onset[k].append(float(np.mean(errs[lo:lo + WINDOW])
                                  / sig_var))
            if k != 'block':
                log = np.array([(t, s, mo) for t, s, mo
                                in m.cons_log])
                evict[k].append(int((log[:, 2] == 3).sum()))
                refuse_n[k].append(int((log[:, 2] == 2).sum()))
        print(f"seed {seed}: " + " | ".join(
            f"{k}={onset[k][-1]:.3f}" for k in kinds))

    print("\n" + "=" * 70)
    print("SATURATION TEST: F4-onset NMSE (sine return, 10 seeds)")
    print("=" * 70)
    for k in kinds:
        v = np.array(onset[k])
        line = f"{labels[k]:<15}: {v.mean():.4f} +/- {v.std():.4f}"
        if k != 'block':
            line += (f"  | evict={np.mean(evict[k]):.1f}"
                     f"  refuse={np.mean(refuse_n[k]):.1f}")
        print(line)
    print("-" * 70)
    b = np.mean(onset['block'])
    for k in kinds[1:]:
        print(f"{labels[k]}: vs block x{b / np.mean(onset[k]):.2f}")

    np.savez('saturation_data.npz',
             **{f'onset_{k}': np.array(onset[k]) for k in kinds},
             **{f'evict_{k}': np.array(evict[k]) for k in kinds[1:]},
             **{f'refuse_{k}': np.array(refuse_n[k])
                for k in kinds[1:]})
    print("Saved: saturation_data.npz")


if __name__ == '__main__':
    main()
