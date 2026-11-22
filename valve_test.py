"""
APIA 2.0 - Valve (pressure-release) confound test + Fig6
==============================================================
The open question left by the B finding: does the F4 gain come from
the stored context, or from the gamma-decay drainage of W_fast
(implicit regularization)?

Two tests:
  1) CORRELATION: ||W_fast|| at the phase-4 boundary vs F4-onset
     NMSE per run (all config x seed points).
  2) INTERVENTION: refuse + norm-clamp (w_fast_cap=R). If recall
     returns, the cause is MAGNITUDE; if not, it is STORAGE.
     R = the median ||W_fast|| of the overwrite run.

Fig6: top panel ||W_fast||(t) refuse vs overwrite (log);
      bottom panel rolling NMSE (phase-4 return window highlighted).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from rls_engine import SplineFeatureMap, BlockDiagRLS
from contextual_memory import ContextualMemory
from continual_protocol import (build_stream, make_stream_states,
                                D, K, WASHOUT, LAM, INIT_VAR,
                                WINDUP_MULT, WINDOW)

SEEDS = 10
PHASES = ['sine', 'square', 'triangle', 'sine']
PL = 1500


def make_model(cfg, cap=None):
    base = BlockDiagRLS(D, K, 1, forgetting=LAM, init_var=INIT_VAR,
                        windup_mult=WINDUP_MULT)
    if cfg == 'block':
        return base
    sat = {'o': 'overwrite', 'r': 'refuse', 'rc': 'refuse'}[cfg[0]]
    return ContextualMemory(base, n_slots=2, beta=1.0, gamma=0.1,
                            eta=2.0, consolidate_every=250,
                            e_gate=0.5, p_inflate=1.5,
                            saturation=sat, w_fast_cap=cap)


def _wnorm(m):
    w = m.model.get_weights() if hasattr(m, 'model') else m.get_weights()
    return float(np.linalg.norm(w))


def run(cfg, Phi, Y, cap=None, log_norm=False):
    m = make_model(cfg, cap)
    errs = np.empty(len(Phi))
    norms = np.empty(len(Phi)) if log_norm else None
    for i in range(len(Phi)):
        errs[i] = float(np.sum(m.update(Phi[i], Y[i]) ** 2))
        if log_norm:
            norms[i] = _wnorm(m)
    return errs, m, norms


def main():
    # --- 1) R calibration: overwrite median norm (seed 0) ---
    fmap = SplineFeatureMap(D, K, 3)
    u_all, y_all = build_stream(0, phases=PHASES, phase_len=PL)
    Phi0 = make_stream_states(u_all, 0, fmap)
    Y0 = y_all[WASHOUT:].reshape(-1, 1)
    _, _, norms_o = run('o', Phi0, Y0, log_norm=True)
    R = float(np.median(norms_o[PL:]))          # median after phase 1
    print(f"R (overwrite median ||W_fast||, P1+): {R:.3f}")

    # --- 2) All configs: onset + boundary norm + phase-ends ---
    cfgs = ['block', 'o', 'r', 'rc']
    onset = {c: [] for c in cfgs}
    norm_b = {c: [] for c in cfgs}   # ||W_fast|| @ P4 boundary (t=3PL)
    end_err = {c: {p: [] for p in range(4)} for c in cfgs}
    for seed in range(SEEDS):
        u_all, y_all = build_stream(seed, phases=PHASES, phase_len=PL)
        sv = float(np.var(y_all))
        Phi = make_stream_states(u_all, seed, fmap)
        Y = y_all[WASHOUT:].reshape(-1, 1)
        for c in cfgs:
            errs, m, nr = run(c, Phi, Y, cap=R if c == 'rc' else None,
                              log_norm=True)
            onset[c].append(float(np.mean(errs[3 * PL:3 * PL + WINDOW])
                                  / sv))
            norm_b[c].append(nr[3 * PL - 1])   # norm at P4 boundary
            for p in range(4):
                end_err[c][p].append(
                    float(np.mean(errs[(p + 1) * PL - WINDOW:(p + 1) * PL])
                          / sv))
        print(f"seed {seed}: " + " | ".join(
            f"{c}={onset[c][-1]:.3f}" for c in cfgs))

    print("\n=== VALVE TEST ===")
    for c in cfgs:
        v = np.array(onset[c])
        print(f"{c:>6}: F4-onset={v.mean():.4f}+/-{v.std():.4f} "
              f"| ||W_fast||@boundary={np.mean(norm_b[c]):.2f}")

    # --- correlation: norm@boundary vs onset (all points, log-norm) ---
    xs, ys = [], []
    for c in cfgs:
        xs += list(np.log10(np.array(norm_b[c]) + 1e-12))
        ys += list(onset[c])
    r_pear, p_pear = stats.pearsonr(xs, ys)
    r_spe, p_spe = stats.spearmanr(xs, ys)
    print(f"correlation log10||W_fast|| vs F4-onset: "
          f"pearson r={r_pear:.3f} (p={p_pear:.2e}), "
          f"spearman rho={r_spe:.3f} (p={p_spe:.2e})")

    # --- phase-end context (baseline's forgetting cost) ---
    print("baseline phase-end NMSE:", [
        f"{np.mean(end_err['block'][p]):.3f}" for p in range(4)])

    # --- 3) Fig6: seed-0 trajectories ---
    _, _, n_r = run('r', Phi0, Y0, log_norm=True)
    _, _, n_rc = run('rc', Phi0, Y0, cap=R, log_norm=True)
    errs_o, _, _ = run('o', Phi0, Y0)
    errs_r, _, _ = run('r', Phi0, Y0)
    errs_rc, _, _ = run('rc', Phi0, Y0, cap=R)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.5),
                                 sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1]})
    t = np.arange(len(norms_o))
    a1.semilogy(t, norms_o, lw=1.4, color='#1f77b4',
                label='overwrite')
    a1.semilogy(t, n_r, lw=1.4, color='#d62728', label='refuse')
    a1.semilogy(t, n_rc, lw=1.2, ls='--', color='#2ca02c',
                label='refuse+clamp')
    a1.axhline(R, color='k', ls=':', lw=1)
    a1.text(len(t) * 0.62, R * 1.3, f'clamp R={R:.2f}', fontsize=8)
    for x in (PL, 2 * PL, 3 * PL):
        a1.axvline(x, color='k', lw=0.8, alpha=0.5)
    a1.set_ylabel(r'$\|W_{\mathrm{fast}}\|$')
    a1.legend(loc='upper left', fontsize=8)
    a1.set_title('Fast-weight magnitude: consolidation as drainage valve')

    roll = lambda e: (np.convolve(e, np.ones(100) / 100, 'valid')
                      / float(np.var(y_all)))
    tt = np.arange(len(errs_o) - 99)
    a2.semilogy(tt, roll(errs_o), lw=1.4, color='#1f77b4')
    a2.semilogy(tt, roll(errs_r), lw=1.4, color='#d62728')
    a2.semilogy(tt, roll(errs_rc), lw=1.2, ls='--', color='#2ca02c')
    for x in (PL, 2 * PL, 3 * PL):
        a2.axvline(x, color='k', lw=0.8, alpha=0.5)
    a2.set_xlabel('step $t$')
    a2.set_ylabel('rolling NMSE (w=100)')
    fig.tight_layout()
    fig.savefig('fig6_valve.png', dpi=300)
    plt.close(fig)

    np.savez('valve_data.npz',
             **{f'onset_{c}': np.array(onset[c]) for c in cfgs},
             **{f'normb_{c}': np.array(norm_b[c]) for c in cfgs},
             norms_overwrite=norms_o, norms_refuse=n_r,
             norms_refuse_clamp=n_rc, R=R)
    print("Saved: fig6_valve.png + valve_data.npz")


if __name__ == '__main__':
    main()
