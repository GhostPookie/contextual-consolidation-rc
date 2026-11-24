"""Fig1: architecture diagram — thesis-bearing, gated/ungated split."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(9.5, 5.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

def box(x, y, w, h, text, fc='#f5f5f5', fs=10, ec='k', lw=1.2, tc='k'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold')

def arrow(x1, y1, x2, y2, label='', fs=9, color='k', ls='-', lw=1.4, off=0.12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=14, color=color, ls=ls, lw=lw))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + off, label, ha='center', fontsize=fs,
                color=color, style='italic')

# --- input / reservoir / spline chain ---
box(0.1, 2.6, 1.3, 0.8, 'input $u_t$', '#e8f0fe')
box(1.8, 2.6, 1.6, 0.8, 'fixed reservoir\n(ESP, $D$=100)', '#e8f0fe')
box(3.8, 2.6, 1.7, 0.8, 'spline features\n$\\phi_t\\in\\mathbb{R}^{1001}$',
    '#e8f0fe')
arrow(1.4, 3.0, 1.8, 3.0)
arrow(3.4, 3.0, 3.8, 3.0)

# --- ungated fast channel ---
box(6.0, 4.0, 2.0, 0.9, '$W_{\\mathrm{fast}}$ (RLS)\nUNGATED — always active',
    '#fdecea', ec='#c0392b', lw=1.8)
arrow(5.5, 3.15, 6.3, 4.0)

# --- gated slow channel ---
box(6.0, 1.1, 2.0, 1.15, 'slots $W_{slow}^{(1..S)}$\nGATED — suppresible',
    '#e9f7ef', ec='#1e8449', lw=1.8)
arrow(5.5, 2.85, 6.3, 2.25)

# gate block
box(8.3, 1.1, 1.5, 1.15, 'gate\n$w_i\\propto e^{-\\eta E_i/E_{min}}$',
    '#e9f7ef', ec='#1e8449', fs=8.8)
arrow(8.0, 1.67, 8.3, 1.67, '')

# adder
box(8.5, 3.3, 1.3, 0.7, '$\\hat y_t = \\phi^\\top W_{fast}$\n$+\\beta\\phi^\\top\\sum_i w_i W_{slow}^{(i)}$',
    '#fff7e0', fs=8.8)
arrow(8.0, 4.45, 8.75, 4.0)          # fast -> sum
arrow(8.0, 1.95, 8.65, 3.3)          # gated -> sum

# --- transfer oku: ungated -> gated ---
arrow(7.0, 4.0, 7.0, 2.28,
      'consolidation (valve):\n$W_{slow}\\!{+}{=}\\frac{1-\\gamma}{\\beta}W_{fast}$,  '
      '$W_{fast}{\\times}{=}\\gamma$', color='#8e44ad', lw=2.0, off=-0.02)

# --- two separate decision channels ---
box(0.3, 0.3, 4.6, 1.0,
    'decision channels (causal $\\widehat{m_2}=\\mathbb{E}[y^2]$ EMA):\n'
    'predict gate $w_i$  |  write-trigger $E_{min}\\gtrless\\tau\\widehat{m_2}$',
    '#f0f0f0', fs=9)
arrow(4.9, 0.9, 8.45, 1.35, color='gray', ls=':')
arrow(4.9, 0.7, 8.55, 3.28, color='gray', ls=':')

# clamp intervention point (Table 6 control)
ax.annotate('norm cap (lossy,\nTable 6 control)', (8.35, 5.35),
            fontsize=9, color='#c0392b', style='italic', ha='left')
arrow(8.5, 5.25, 7.95, 4.82, color='#c0392b', ls='--')

ax.set_title('Write-protected contextual memory: ungated fast channel '
             '+ gated slow slots', fontsize=12)
fig.tight_layout()
fig.savefig('fig1_architecture.png', dpi=170)
print('saved fig1_architecture.png')
