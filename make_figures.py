"""
APIA 2.0 - Paper figure + statistics generation (REAL DATA)
==============================================================
In a single run:
  * 10 seeds x 5 variants (block / naive / ctx_v1 / ctx_v2 / ctx_v3)
    -> Fig5 ablation bars + Wilcoxon signed-rank
  * seed-0 error trajectory (block vs v3) -> Fig3
  * seed-0 gate weights (T x 3) -> Fig4 heatmap

Variant semantics (contextual_memory ablation flags):
  ctx_v1: err_mode='total', write_protect=False  -> self-masking + blind argmin
  ctx_v2: err_mode='slow',  write_protect=True, empty_rule='norm'
  ctx_v3: err_mode='slow',  write_protect=True, empty_rule='count' (FINAL)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from reservoir import ReservoirEngine
from rls_engine import SplineFeatureMap, BlockDiagRLS
from memory import DualMemory
from contextual_memory import ContextualMemory
from multihead import MultiHeadMemory
from continual_protocol import (build_stream, make_stream_states,
                                D, K, WASHOUT, PHASE_LEN, LAM, INIT_VAR,
                                WINDUP_MULT, WINDOW)

SEEDS = 10
TOTAL_STEPS = 3 * PHASE_LEN          # post-washout stream length
PLOT_SEED = 0


def make_variant(kind, sig_var):
    base = lambda: BlockDiagRLS(D, K, 1, forgetting=LAM,
                                init_var=INIT_VAR, windup_mult=WINDUP_MULT)
    if kind == 'block':
        return base()
    if kind == 'naive':
        return DualMemory(base(), beta=1.0, gamma=0.1,
                          consolidate_every=250, p_inflate=1.5)
    if kind == 'multihead':
        return MultiHeadMemory(
            lambda: BlockDiagRLS(D, K, 1, forgetting=LAM,
                                 init_var=INIT_VAR,
                                 windup_mult=WINDUP_MULT), n_slots=3)
    if kind == 'mh_alloc':
        # Channel isolation: write-triggered allocation added to the
        # competitor architecture (same 0.5 threshold as tau, fair)
        return MultiHeadMemory(
            lambda: BlockDiagRLS(D, K, 1, forgetting=LAM,
                                 init_var=INIT_VAR,
                                 windup_mult=WINDUP_MULT), n_slots=3,
            alloc_thresh=0.5)
    if kind == 'full':
        from rls_engine import FullRLS
        return FullRLS(D * K + 1, 1, forgetting=LAM,
                       init_var=INIT_VAR, windup_mult=WINDUP_MULT)
    kw = {}
    if kind == 'ctx_v0':
        kw = dict(err_mode='total', write_protect=False)   # original bug
    elif kind == 'ctx_v1':
        kw = dict(write_protect=False)                    # self-mask fix only
    elif kind == 'ctx_v2':
        kw = dict(empty_rule='norm')                      # + write-protect
    return ContextualMemory(base(), n_slots=3, beta=1.0, gamma=0.1,
                            eta=2.0, consolidate_every=250,
                            e_gate=0.5,
                            p_inflate=1.5, **kw)


def run_variant(kind, Phi, Y, sig_var):
    m = make_variant(kind, sig_var)
    errs = np.empty(len(Phi))
    for i in range(len(Phi)):
        errs[i] = float(np.sum(m.update(Phi[i], Y[i]) ** 2))
    return errs, m


def main():
    # full list for metrics.json; Fig5 bars are selected separately
    all_kinds = ['block', 'naive', 'multihead', 'mh_alloc',
                 'ctx_v0', 'ctx_v1', 'ctx_v2', 'ctx_v3', 'full']
    bar_kinds = ['block', 'naive', 'multihead', 'mh_alloc',
                 'ctx_v0', 'ctx_v1', 'ctx_v2', 'ctx_v3']
    labels = {'block': 'Block-RLS\n(baseline)', 'naive': '+Naive\nmemory',
              'multihead': 'MultiHead\n(Bereska)',
              'mh_alloc': 'MultiHead\n+alloc',
              'ctx_v0': 'CTX v0\n(self-mask)', 'ctx_v1': 'CTX v1\n(slow-err)',
              'ctx_v2': 'CTX v2\n(+protect,norm)', 'ctx_v3': 'CTX v3\n(+count) FINAL'}
    onset = {k: [] for k in all_kinds}
    end = {k: {p: [] for p in range(3)} for k in all_kinds}
    wsens = {k: {w: [] for w in (100, 250, 500)}
             for k in ('block', 'ctx_v3')}
    traj, gates, sig0 = {}, None, None
    fmap = SplineFeatureMap(D, K, 3)

    for seed in range(SEEDS):
        u_all, y_all = build_stream(seed)
        sig_var = float(np.var(y_all))
        Phi = make_stream_states(u_all, seed, fmap)
        Y = y_all[WASHOUT:].reshape(-1, 1)
        lo = 2 * PHASE_LEN
        for k in all_kinds:
            errs, m = run_variant(k, Phi, Y, sig_var)
            onset[k].append(float(np.mean(errs[lo:lo + WINDOW]) / sig_var))
            for p in range(3):
                hi = (p + 1) * PHASE_LEN
                end[k][p].append(
                    float(np.mean(errs[hi - WINDOW:hi]) / sig_var))
            if k in wsens:
                for w in wsens[k]:
                    wsens[k][w].append(
                        float(np.mean(errs[lo:lo + w]) / sig_var))
            if seed == PLOT_SEED and k in ('block', 'ctx_v3'):
                traj[k] = errs
            if seed == PLOT_SEED and k == 'ctx_v3':
                gates = np.array(m.gate_log)
                sig0 = sig_var
        print(f"seed {seed}: " + " | ".join(
            f"{k}={onset[k][-1]:.3f}" for k in all_kinds[:8]))

    # ---------------- Wilcoxon ----------------
    b, c = np.array(onset['block']), np.array(onset['ctx_v3'])
    wilw, pilp = stats.wilcoxon(b, c, alternative='greater')
    t_stat, p_t = stats.ttest_rel(b, c)
    print("\n" + "=" * 60)
    print("WILCOXON (block > ctx_v3, one-sided): "
          f"W={wilw:.1f}, p={pilp:.3e}  | paired t: t={t_stat:.2f}, p={p_t:.3e}")
    print(f"block={b.mean():.4f}+/-{b.std():.4f}  "
          f"ctx={c.mean():.4f}+/-{c.std():.4f}  "
          f"ratio x{b.mean() / c.mean():.2f}")
    print("=" * 60)

    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                         'axes.grid': True, 'grid.alpha': 0.3})

    # ---------------- Fig3: error trajectory ----------------
    roll = lambda e: np.convolve(e, np.ones(WINDOW) / WINDOW, 'valid') / sig0
    t = np.arange(TOTAL_STEPS - WINDOW + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(t, roll(traj['block']), ls='--', lw=1.6, color='0.45',
                label='Block-RLS (baseline)')
    ax.semilogy(t, roll(traj['ctx_v3']), lw=2.0, color='#1f77b4',
                label='Block-RLS + CTX (final)')
    for x, lab, col in [(PHASE_LEN, 'sine$\\rightarrow$square', 'r'),
                        (2 * PHASE_LEN, 'square$\\rightarrow$sine', 'g')]:
        ax.axvline(x, color=col, ls=':', lw=1.4)
        ax.text(x + 30, 1.5, lab, color=col, fontsize=8, rotation=90)
    ax.set_xlabel('step $t$'); ax.set_ylabel('rolling NMSE ($w$=250)')
    ax.legend()
    fig.tight_layout(); fig.savefig('fig3_error_trajectory.png', dpi=300)
    plt.close(fig)

    # ---------------- Fig4: gate heatmap ----------------
    fig, ax = plt.subplots(figsize=(7, 2.8))
    im = ax.imshow(gates.T, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=1, origin='lower')
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['slot 0', 'slot 1', 'slot 2'])
    ax.set_xlabel('step $t$'); ax.set_ylabel('gate $w_k$')
    for x in (PHASE_LEN, 2 * PHASE_LEN):
        ax.axvline(x, color='k', lw=1)
    fig.colorbar(im, ax=ax, label='$w_k$')
    fig.tight_layout(); fig.savefig('fig4_slot_heatmap.png', dpi=300)
    plt.close(fig)

    # ---------------- Fig5: ablation ----------------
    # Competitor-ladder data comes from clamp_control.json
    # (mh_alloc_soft was not run in-protocol — from the main-protocol
    # control)
    import json as _json
    cc = _json.load(open('clamp_control.json'))
    extra = np.array(cc['mh_alloc_soft']['per_seed'])
    bar_all = bar_kinds[:4] + ['mh_alloc_soft'] + bar_kinds[4:]
    vals = {**{k: np.array(onset[k]) for k in bar_kinds},
            'mh_alloc_soft': extra}
    lbl = dict(labels); lbl['mh_alloc_soft'] = 'MultiHead\n+alloc+soft'
    means = [np.mean(vals[k]) for k in bar_all]
    stds = [np.std(vals[k]) for k in bar_all]
    nb = len(bar_all)
    # three families: baseline (gray) | competitor ladder (orange,
    # hatch) | CTX ladder (blue) — Table 3a/3b split
    fam = (['#d9d9d9'] * 2 + ['#ff7f0e'] * 3 +
           ['#9ecae1', '#6baed6', '#3182bd', '#1f77b4'])
    hatch = ['', '', '//', '//', '//', '', '', '', '']
    fig, ax = plt.subplots(figsize=(9.5, 4))
    bars = ax.bar(range(nb), means, yerr=stds, capsize=5,
                  color=fam, edgecolor='k', alpha=0.9)
    for b, h in zip(bars, hatch):
        b.set_hatch(h)
    ax.axvline(1.5, color='k', ls=':', lw=1)
    ax.axvline(4.5, color='k', ls=':', lw=1)
    ax.set_xticks(range(nb))
    ax.set_xticklabels([lbl[k] for k in bar_all], fontsize=8)
    ax.set_ylabel('F3-onset NMSE')
    for i, (mu, sd) in enumerate(zip(means, stds)):
        if i:
            ax.annotate(f'x{means[0] / mu:.2f}',
                        (i, mu + sd + 0.02), ha='center',
                        fontsize=9, fontweight='bold', color='#1f77b4')
    ax.set_ylim(0, max(np.array(means) + np.array(stds)) * 1.15)
    fig.tight_layout(); fig.savefig('fig5_ablation_bar.png', dpi=300)
    plt.close(fig)

    # ---------------- window sensitivity ----------------
    print("\nWindow sensitivity (F3 onset, mean NMSE):")
    for ww in (100, 250, 500):
        bw, cw = np.mean(wsens['block'][ww]), np.mean(wsens['ctx_v3'][ww])
        print(f"  w={ww:4d}: block={bw:.4f}  ctx={cw:.4f}  ratio x{bw/cw:.2f}")

    np.savez('fig_data.npz',
             **{f'onset_{k}': np.array(onset[k]) for k in all_kinds},
             **{f'wsens_{k}_{w}': np.array(wsens[k][w])
                for k in wsens for w in wsens[k]},
             traj_block=traj['block'], traj_ctx=traj['ctx_v3'],
             gates=gates, sig0=sig0)

    # ---------------- metrics.json: SINGLE SOURCE OF TRUTH ----------------
    import json
    metrics = {
        'seeds': SEEDS, 'window': WINDOW, 'protocol': 'sine>square>sine',
        'f3_onset': {k: {'mean': float(np.mean(onset[k])),
                         'std': float(np.std(onset[k])),
                         'per_seed': [float(v) for v in onset[k]]}
                     for k in all_kinds},
        'phase_end': {k: {f'f{p+1}': {'mean': float(np.mean(end[k][p])),
                                     'std': float(np.std(end[k][p]))}
                          for p in range(3)} for k in all_kinds},
        'window_sensitivity': {
            str(w): {'block': float(np.mean(wsens['block'][w])),
                     'ctx': float(np.mean(wsens['ctx_v3'][w])),
                     'ratio': float(np.mean(wsens['block'][w])
                                    / np.mean(wsens['ctx_v3'][w]))}
            for w in (100, 250, 500)},
        'wilcoxon': {'W': float(wilw), 'p': float(pilp)},
        'paired_t': {'t': float(t_stat), 'p': float(p_t)},
        'headline_ratio': float(np.mean(onset['block'])
                                / np.mean(onset['ctx_v3'])),
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved: fig3/4/5 png + fig_data.npz + metrics.json")


if __name__ == '__main__':
    main()
