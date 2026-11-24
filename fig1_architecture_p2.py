"""Fig1 (Phase-2): frozen-LLM + APIA adapter architecture.

Reads left->right: stream text -> frozen backbone (no grad) ->
last-token hidden state phi_t -> spline expansion -> two channels
(ungated W_fast RLS + gated W_slow slots) -> prediction y_hat.
The consolidation arrow is the ONLY write path between channels;
the dashed red X marks where backprop would be (absent by design).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10.2, 5.6))
ax.set_xlim(0, 10.8); ax.set_ylim(0, 6); ax.axis('off')


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


# --- stream -> frozen backbone -> feature chain ---
box(0.1, 2.7, 1.5, 0.8, 'stream text\n$A\\!\\to\\!B\\!\\to\\!A\'$', '#e8f0fe')
box(2.0, 2.7, 1.9, 0.8, 'frozen backbone\nGPT-2 / OPT\nrequires_grad=F',
    '#e8f0fe')
box(4.3, 2.7, 1.9, 0.8, 'last-token hidden\n$\\phi_t\\in\\mathbb{R}^{d}$\n$d\\leq 2048$',
    '#e8f0fe')
box(6.5, 2.7, 1.5, 0.8, 'spline map\n$\\mathbb{R}^{d\\cdot K+1}$',
    '#e8f0fe')
arrow(1.6, 3.1, 2.0, 3.1)
arrow(3.9, 3.1, 4.3, 3.1)
arrow(6.2, 3.1, 6.5, 3.1)

# no-backprop marker on the backbone boundary
ax.text(2.95, 2.28, 'no $\\nabla$ — no backward graph',
        fontsize=9.5, color='#c0392b', style='italic', ha='center')
arrow(2.95, 2.65, 2.95, 2.42, color='#c0392b', ls='--', lw=1.2)

# --- ungated fast channel ---
box(8.3, 4.1, 2.3, 0.95, '$W_{\\mathrm{fast}}$ (block-diag RLS)\n'
    'UNGATED — update every step\nno lr (2nd-order)',
    '#fdecea', ec='#c0392b', lw=1.8, fs=8.8)
arrow(8.0, 3.25, 8.6, 4.1)

# --- gated slow channel ---
box(8.3, 1.0, 2.3, 1.15, 'slots $W_{\\mathrm{slow}}^{(1..S)}$\n'
    'GATED + write-protected\nCPU-resident (~0.4 MB)', '#e9f7ef',
    ec='#1e8449', lw=1.8, fs=8.8)
arrow(8.0, 2.95, 8.6, 2.15)

# gate block
box(6.15, 0.45, 1.95, 1.7, 'gate\n\n$w_i\\propto e^{-\\eta E_i/E_{min}}$\n\n'
    'slot errors exclude\n$W_{\\mathrm{fast}}$ (self-masking)',
    '#e9f7ef', ec='#1e8449', fs=8.2)
arrow(8.1, 1.45, 8.3, 1.5, '')

# adder / prediction
box(8.55, 5.25, 1.9, 0.7,
    '$\\hat y_t = \\phi^\\top W_{fast} + \\beta\\phi^\\top\\sum_i w_i W_{slow}^{(i)}$',
    '#fff7e0', fs=8.2)
arrow(9.45, 5.05, 9.5, 5.25)
arrow(10.15, 2.15, 10.3, 5.25)

# consolidation arrow: fast -> slot
arrow(9.45, 4.1, 9.45, 2.18,
      'consolidate: $W_{slow}{+}{=}\\frac{1-\\gamma}{\\beta}W_{fast}$,'
      '  $W_{fast}{\\times}{=}\\gamma$',
      color='#8e44ad', lw=2.0, off=-0.06, fs=8.2)

# write trigger annotation
box(0.3, 0.5, 5.4, 1.3,
    'write trigger: period elapsed AND '
    '$e_{fast} \\leq e_{gate}\\,\\widehat{m_2}$\n'
    'target: $E_k{<}\\tau\\widehat{m_2}$ reinforce,'
    ' else argmin $c_i$ (emptiest slot)',
    '#f0f0f0', fs=9.2)
arrow(5.7, 1.1, 8.32, 1.35, color='gray', ls=':')

# predict-before-update banner
ax.text(5.4, 5.65, 'predict-before-update, single pass, no replay buffer',
        fontsize=10.5, color='dimgray', style='italic', ha='center')

ax.set_title('APIA on a frozen LLM: forward-only feature-space adaptation '
             'with write-protected contextual memory', fontsize=12.5)
fig.tight_layout()
fig.savefig('fig1_architecture_p2.png', dpi=165)
print('saved fig1_architecture_p2.png')
