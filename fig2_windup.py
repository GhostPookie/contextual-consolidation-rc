"""
APIA 2.0 - Fig2: covariance windup and trace bound
===================================================
FullRLS lambda=0.98 UNBOUNDED vs BOUNDED (windup_mult=1e4), MG-17,
seed 0. Dual axis: trace(P) (log) + rolling error (log).
The windup divergence is not overflow: accumulated rounding error
breaks P's positive-definiteness -> divergence. The delay between
trace reaching the bound and NMSE diverging is marked.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rls_engine import (SplineFeatureMap, FullRLS, mackey_glass)
from reservoir import ReservoirEngine

D, K = 100, 10
WASHOUT, TRAIN = 500, 5000
LAM, INIT_VAR, WMULT = 0.98, 1e2, 1e4
BOUND = INIT_VAR * (D * K + 1) * WMULT   # FullRLS trace_bound formula


def prepare(seed):
    series = mackey_glass(WASHOUT + TRAIN, seed=seed)
    u_all = (series - series.mean()) / series.std()
    y_all = np.roll(u_all, -1)
    res = ReservoirEngine(1, D, 0.1, 0.3, 0.8, seed=seed)
    fmap = SplineFeatureMap(D, K, 3)
    Phi = np.empty((TRAIN, fmap.n_features))
    Y = np.empty((TRAIN, 1))
    for t in range(WASHOUT + TRAIN):
        x = res.step(np.array([u_all[t]]))
        if t >= WASHOUT:
            Phi[t - WASHOUT] = fmap.transform(x)
            Y[t - WASHOUT, 0] = y_all[t]
    return Phi, Y, float(np.var(y_all))


def run(Phi, Y, bound_mult):
    m = FullRLS(D * K + 1, 1, forgetting=LAM, init_var=INIT_VAR,
                windup_mult=bound_mult)
    errs = np.empty(len(Phi))
    tr = np.empty(len(Phi))
    md = np.empty(len(Phi))
    for i in range(len(Phi)):
        e = m.update(Phi[i], Y[i])
        errs[i] = float(np.sum(e ** 2))
        tr[i] = float(np.trace(m.P))
        md[i] = float(np.min(np.diag(m.P)))
    return errs, tr, md


def main():
    Phi, Y, sv = prepare(0)
    e_u, tr_u, md_u = run(Phi, Y, None)    # unbounded
    e_b, tr_b, md_b = run(Phi, Y, WMULT)   # bounded
    neg_t = int(np.argmax(md_u < 0)) if (md_u < 0).any() else -1
    print(f"first negative diag(P): t={neg_t}")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1]})
    t = np.arange(len(tr_u))
    a1.semilogy(t, tr_u, lw=1.3, color='#d62728',
                label='unbounded trace($P$)')
    a1.semilogy(t, tr_b, lw=1.3, color='#1f77b4',
                label='trace-bounded')
    a1.axhline(BOUND, color='k', ls=':', lw=1)
    a1.text(len(t) * 0.02, BOUND * 3, f'bound={BOUND:.0e}', fontsize=8)
    cross = int(np.argmax(tr_u > BOUND))
    a1.axvline(cross, color='gray', ls=':', lw=1)
    a1.set_ylabel(r'trace$(P_t)$')
    a1.legend(loc='upper left', fontsize=8)
    a1.set_title(r'Covariance windup, $\lambda=0.98$: trace bound vs unbounded')

    roll = lambda e: np.convolve(e, np.ones(100) / 100, 'valid') / sv
    tt = np.arange(len(e_u) - 99)
    a2.semilogy(tt, roll(e_u), lw=1.3, color='#d62728')
    a2.semilogy(tt, roll(e_b), lw=1.3, color='#1f77b4')
    # sustained divergence: the moment ru permanently exceeds 10x the
    # bounded curve
    ru, rb = roll(e_u), roll(e_b)
    above = ru > 10 * rb
    div = int(np.max(np.where(~above))) + 1
    a1.axvline(div, color='k', ls='--', lw=1)
    a2.axvline(div, color='k', ls='--', lw=1)
    if neg_t >= 0:
        a1.axvline(neg_t, color='purple', ls='-.', lw=1)
        a2.axvline(neg_t, color='purple', ls='-.', lw=1)
    a2.annotate(f'sustained divergence ~t={div} (ru>10x rb)\n'
                f'min diag(P)<0 at t={neg_t}',
                (div, 1e-1), fontsize=8, color='#d62728')
    a2.set_xlabel('step $t$')
    a2.set_ylabel('rolling NMSE (w=100)')
    fig.tight_layout()
    fig.savefig('fig2_windup.png', dpi=300)
    plt.close(fig)

    print(f"unbounded: trace max={tr_u.max():.2e}, NMSE end={np.mean(e_u[-100:])/sv:.2e}")
    print(f"bounded  : trace max={tr_b.max():.2e}, NMSE end={np.mean(e_b[-100:])/sv:.2e}")
    print("Saved: fig2_windup.png")


if __name__ == '__main__':
    main()
