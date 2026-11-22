"""
APIA 2.0 - Main-protocol valve control + C-attribution variants
=================================================================
1) HEADLINE CONFOUND CONTROL: block+clamp on the main protocol
   (sine>square>sine). R = median ||W_fast|| at ctx_v3's
   consolidation events (pooled over 10 seeds) — protocol-specific,
   a matched-norm control NOT tuned on the test.
2) C-attribution: multihead+soft_predict (normalized gate only)
   and mh_alloc+soft (alloc + soft gate) — whether the remaining
   ~3.8x comes from gate quality or from transfer.
3) Lossy-vs-lossless: F3-end error of the clamp variants —
   prediction: clamp destroys information -> end-error worse than
   CTX.
"""

import numpy as np
from scipy import stats

from rls_engine import SplineFeatureMap, BlockDiagRLS
from contextual_memory import ContextualMemory
from multihead import MultiHeadMemory
from continual_protocol import (build_stream, make_stream_states,
                                D, K, WASHOUT, PHASE_LEN, LAM,
                                INIT_VAR, WINDUP_MULT, WINDOW)

SEEDS = 10


def ctx():
    base = BlockDiagRLS(D, K, 1, forgetting=LAM, init_var=INIT_VAR,
                        windup_mult=WINDUP_MULT)
    return ContextualMemory(base, n_slots=3, beta=1.0, gamma=0.1,
                            eta=2.0, consolidate_every=250,
                            e_gate=0.5, p_inflate=1.5)


def mh(alloc, soft):
    return MultiHeadMemory(
        lambda: BlockDiagRLS(D, K, 1, forgetting=LAM,
                             init_var=INIT_VAR,
                             windup_mult=WINDUP_MULT),
        n_slots=3, alloc_thresh=alloc, soft_predict=soft)


def run(m, Phi, Y, cap=None, log_wnorm_at_cons=False):
    errs = np.empty(len(Phi))
    cons_norms = []
    prev_cons = 0
    for i in range(len(Phi)):
        errs[i] = float(np.sum(m.update(Phi[i], Y[i]) ** 2))
        if cap is not None:
            w = m.get_weights()
            n = float(np.linalg.norm(w))
            if n > cap:
                m.set_weights(w * (cap / n))
        if log_wnorm_at_cons and m.n_consolidations > prev_cons:
            cons_norms.append(
                float(np.linalg.norm(m.model.get_weights())))
            prev_cons = m.n_consolidations
    return errs, cons_norms


def main():
    fmap = SplineFeatureMap(D, K, 3)
    lo = 2 * PHASE_LEN

    # --- 1) R calibration: ctx consolidation norms (10 seeds) ---
    cons_norms = []
    for seed in range(SEEDS):
        u_all, y_all = build_stream(seed)
        Phi = make_stream_states(u_all, seed, fmap)
        Y = y_all[WASHOUT:].reshape(-1, 1)
        m = ctx()
        prev = 0
        for i in range(len(Phi)):
            m.update(Phi[i], Y[i])
            if m.n_consolidations > prev:
                cons_norms.append(
                    float(np.linalg.norm(m.model.get_weights())))
                prev = m.n_consolidations
    R = float(np.median(cons_norms))
    print(f"R (ctx consolidation median ||W_fast||, {len(cons_norms)} events)"
          f": {R:.3f}")

    # --- 2) Main protocol: block / block+clamp / ctx / mh variants ---
    kinds = ['block', 'block_clamp', 'ctx', 'mh_soft', 'mh_alloc_soft']
    ons = {k: [] for k in kinds}
    end3 = {k: [] for k in kinds}
    for seed in range(SEEDS):
        u_all, y_all = build_stream(seed)
        sv = float(np.var(y_all))
        Phi = make_stream_states(u_all, seed, fmap)
        Y = y_all[WASHOUT:].reshape(-1, 1)
        for k in kinds:
            if k == 'block':
                m = BlockDiagRLS(D, K, 1, forgetting=LAM,
                                 init_var=INIT_VAR,
                                 windup_mult=WINDUP_MULT)
                errs, _ = run(m, Phi, Y)
            elif k == 'block_clamp':
                m = BlockDiagRLS(D, K, 1, forgetting=LAM,
                                 init_var=INIT_VAR,
                                 windup_mult=WINDUP_MULT)
                errs, _ = run(m, Phi, Y, cap=R)
            elif k == 'ctx':
                errs, _ = run(ctx(), Phi, Y)
            elif k == 'mh_soft':
                errs, _ = run(mh(None, True), Phi, Y)
            else:
                errs, _ = run(mh(0.5, True), Phi, Y)
            ons[k].append(float(np.mean(errs[lo:lo + WINDOW]) / sv))
            hi = 3 * PHASE_LEN
            end3[k].append(float(np.mean(errs[hi - WINDOW:hi]) / sv))
        print(f"seed {seed}: " + " | ".join(
            f"{k}={ons[k][-1]:.3f}" for k in kinds))

    print("\n=== MAIN-PROTOCOL VALVE CONTROL ===")
    for k in kinds:
        v, e = np.array(ons[k]), np.array(end3[k])
        print(f"{k:>14}: F3-onset={v.mean():.4f}+/-{v.std():.4f} "
              f"| F3-end={e.mean():.2e}")
    b, c = np.array(ons['block']), np.array(ons['block_clamp'])
    w, p = stats.wilcoxon(b, c, alternative='greater')
    print(f"block vs block+clamp: W={w}, p={p:.4f} "
          f"(one-sided); mean ratio x{b.mean()/c.mean():.2f}")
    cc = np.array(ons['ctx'])
    w2, p2 = stats.wilcoxon(c, cc, alternative='greater')
    print(f"block+clamp vs ctx: W={w2}, p={p2:.4f} "
          f"| remaining ratio x{c.mean()/cc.mean():.2f}")

    import json
    out = {'R': R, 'seeds': SEEDS,
           **{k: {'onset_mean': float(np.mean(ons[k])),
                  'onset_std': float(np.std(ons[k])),
                  'f3_end_mean': float(np.mean(end3[k])),
                  'per_seed': [float(v) for v in ons[k]]}
              for k in kinds}}
    json.dump(out, open('clamp_control.json', 'w'), indent=2)
    print("Saved: clamp_control.json")


if __name__ == '__main__':
    main()
