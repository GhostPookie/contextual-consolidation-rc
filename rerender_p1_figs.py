"""Re-render paper-1 figures for README display (bigger fonts, lower dpi).

Reads cached fig_data.npz + clamp_control.json — does NOT rerun the
experiment protocol (make_figures.py remains the data producer).
Same plot code, README-friendly sizing.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from continual_protocol import PHASE_LEN, WINDOW

D = np.load('fig_data.npz')
sig0 = float(D['sig0'])
TOTAL_STEPS = 3 * PHASE_LEN

plt.rcParams.update({'font.family': 'serif', 'font.size': 12,
                     'axes.grid': True, 'grid.alpha': 0.3})

# ---------------- Fig3: error trajectory ----------------
roll = lambda e: np.convolve(e, np.ones(WINDOW) / WINDOW, 'valid') / sig0
t = np.arange(TOTAL_STEPS - WINDOW + 1)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.semilogy(t, roll(D['traj_block']), ls='--', lw=1.8, color='0.45',
            label='Block-RLS (baseline)')
ax.semilogy(t, roll(D['traj_ctx']), lw=2.2, color='#1f77b4',
            label='Block-RLS + CTX (final)')
for x, lab, col in [(PHASE_LEN, 'sine$\\rightarrow$square', 'r'),
                    (2 * PHASE_LEN, 'square$\\rightarrow$sine', 'g')]:
    ax.axvline(x, color=col, ls=':', lw=1.4)
    ax.text(x + 30, 1.5, lab, color=col, fontsize=10, rotation=90)
ax.set_xlabel('step $t$'); ax.set_ylabel('rolling NMSE ($w$=250)')
ax.legend(fontsize=10)
fig.tight_layout(); fig.savefig('fig3_error_trajectory.png', dpi=170)
plt.close(fig)

# ---------------- Fig5: ablation ----------------
cc = json.load(open('clamp_control.json'))
bar_kinds = ['block', 'naive', 'multihead', 'mh_alloc',
             'ctx_v0', 'ctx_v1', 'ctx_v2', 'ctx_v3']
labels = {'block': 'Block-RLS\n(baseline)', 'naive': '+Naive\nmemory',
          'multihead': 'MultiHead\n(Bereska)',
          'mh_alloc': 'MultiHead\n+alloc',
          'ctx_v0': 'CTX v0\n(self-mask)', 'ctx_v1': 'CTX v1\n(slow-err)',
          'ctx_v2': 'CTX v2\n(+protect,norm)', 'ctx_v3': 'CTX v3\n(+count) FINAL'}
extra = np.array(cc['mh_alloc_soft']['per_seed'])
bar_all = bar_kinds[:4] + ['mh_alloc_soft'] + bar_kinds[4:]
vals = {**{k: D[f'onset_{k}'] for k in bar_kinds}, 'mh_alloc_soft': extra}
lbl = dict(labels); lbl['mh_alloc_soft'] = 'MultiHead\n+alloc+soft'
means = [np.mean(vals[k]) for k in bar_all]
stds = [np.std(vals[k]) for k in bar_all]
nb = len(bar_all)
fam = (['#d9d9d9'] * 2 + ['#ff7f0e'] * 3 +
       ['#9ecae1', '#6baed6', '#3182bd', '#1f77b4'])
hatch = ['', '', '//', '//', '//', '', '', '', '']
fig, ax = plt.subplots(figsize=(10, 4.4))
bars = ax.bar(range(nb), means, yerr=stds, capsize=5,
              color=fam, edgecolor='k', alpha=0.9)
for b, h in zip(bars, hatch):
    b.set_hatch(h)
ax.axvline(1.5, color='k', ls=':', lw=1)
ax.axvline(4.5, color='k', ls=':', lw=1)
ax.set_xticks(range(nb))
ax.set_xticklabels([lbl[k] for k in bar_all], fontsize=8.5)
ax.set_ylabel('F3-onset NMSE')
for i, (mu, sd) in enumerate(zip(means, stds)):
    if i:
        ax.annotate(f'x{means[0] / mu:.2f}',
                    (i, mu + sd + 0.02), ha='center',
                    fontsize=10.5, fontweight='bold', color='#1f77b4')
ax.set_ylim(0, max(np.array(means) + np.array(stds)) * 1.15)
fig.tight_layout(); fig.savefig('fig5_ablation_bar.png', dpi=170)
plt.close(fig)

print('re-rendered fig3_error_trajectory.png + fig5_ablation_bar.png '
      'from fig_data.npz (no experiment rerun)')
