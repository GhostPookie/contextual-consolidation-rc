"""
make_tables.py — generates all LaTeX tables from the data files.
Sources: fig_data.npz, metrics.json, clamp_control.json,
         valve_data.npz, saturation_data.npz.
CANON = measured values of runs recorded outside the metrics files
(with provenance notes); kept here as the single source so that no
hand-written table values remain in the paper.
Outputs: gen/table*.tex + numbers_macros.tex  (audit_numbers.py
verifies these files).
"""
import json
import numpy as np
import os

os.makedirs('gen', exist_ok=True)
f = np.load('fig_data.npz')
m = json.load(open('metrics.json'))
c = json.load(open('clamp_control.json'))
v = np.load('valve_data.npz')
s = np.load('saturation_data.npz')

# ---------- CANON: measurements outside metrics.json ----------
# Source: benchmark_block_rls.py (10-seed MG-17 characterization +
# single-seed lambda=0.98 operating-point diagnostic) and
# different_tasks.py (5-seed cross-family) runs; values verified
# against the working notes.
CANON = {
    't1_ridge_nmse': '9e-6', 't1_ridge_upd': 'offline', 't1_ridge_mem': '---',
    't1_full_l1_nmse': '1e-3', 't1_full_l1_upd': '3.4 ms',
    't1_full_mem': '7.6 MB',
    't1_block_l1_nmse': '2e-2', 't1_block_l1_upd': '53 \\textmu s',
    't1_block_mem': '78 kB',
    't1_full_b_nmse': '8.1e-3', 't1_full_b_upd': '3.8 ms',
    't1_block_b_nmse': '0.115', 't1_block_b_upd': '87 \\textmu s',
    # different_tasks (5 seed)
    't4_mg_block': '1.40e-2', 't4_mg_mh': '1.40e-2',
    't4_mg_ctx': '1.38e-2', 't4_mg_gain': '1.01',
    't4_mg_slots': '[0,0,0] 5/5',
    't4_na_block': '0.108', 't4_na_mh': '0.108',
    't4_na_ctx': '0.104', 't4_na_gain': '1.04',
    't4_na_slots': '[0,1,0] 3/5',
    # Table 5: the 4-regime 'block+clamp' run was measured in a
    # separate output (no such config in the valve/saturation npz's)
    't5_bc_onset': '0.051', 't5_bc_std': '0.014', 't5_bc_norm': '3.2',
}

mu = lambda k: float(np.mean(f[f'onset_{k}']))
sd = lambda k: float(np.std(f[f'onset_{k}']))
pe = lambda k, p: m['phase_end'][k][p]['mean']


def w(name, body):
    with open(f'gen/{name}.tex', 'w', encoding='utf-8') as fh:
        fh.write(body)


def fmt(x, nd=3):
    return f'{x:.{nd}f}'


# ---------- Table 1: readout characterization ----------
w('table1', rf"""\begin{{tabular}}{{lcccc}}
\toprule
Method & $\lambda$ & NMSE @5000 & Update & $P$ memory \\
\midrule
Ridge (batch) & --- & $\approx${CANON['t1_ridge_nmse']} & {CANON['t1_ridge_upd']} & {CANON['t1_ridge_mem']} \\
Full RLS & 1.0 & $\approx${CANON['t1_full_l1_nmse']} & {CANON['t1_full_l1_upd']}/step & {CANON['t1_full_mem']} \\
Block-RLS & 1.0 & $\approx${CANON['t1_block_l1_nmse']} & {CANON['t1_block_l1_upd']}/step & {CANON['t1_block_mem']} \\
Full RLS + bound & 0.98 & {CANON['t1_full_b_nmse']} & {CANON['t1_full_b_upd']}/step & {CANON['t1_full_mem']} \\
Block-RLS + bound & 0.98 & {CANON['t1_block_b_nmse']} & {CANON['t1_block_b_upd']}/step & {CANON['t1_block_mem']} \\
\bottomrule
\end{{tabular}}""")

# ---------- Table 2: main protocol ----------
pe_blk, pe_nai = pe('block', 'f3'), pe('naive', 'f3')
pe_mh, pe_mha = pe('multihead', 'f3'), pe('mh_alloc', 'f3')
pe_ctx = pe('ctx_v3', 'f3')
f1_blk, f2_blk = pe('block', 'f1'), pe('block', 'f2')
f2_ctx = pe('ctx_v3', 'f2')
pe_mhs = c['mh_soft']['f3_end_mean']
pe_mhas = c['mh_alloc_soft']['f3_end_mean']
mo = lambda k: m['f3_onset'][k]['mean']
so = lambda k: m['f3_onset'][k]['std']
w('table2', rf"""\begin{{tabular}}{{lcccc}}
\toprule
Method & F1-end & F2-end & F3-onset (primary) & F3-end \\
\midrule
FullRLS+bound & {pe('full','f1'):.1e} & {fmt(pe('full','f2'))} & ${mo('full'):.2f}\pm{so('full'):.2f}$ & {pe('full','f3'):.1e} \\
BlockRLS+bound & {f1_blk:.1e} & {fmt(f2_blk)} & ${fmt(mu('block'))}\pm{fmt(sd('block'),2)}$ & {pe_blk:.1e} \\
+naive memory & {f1_blk:.1e} & {fmt(pe('naive','f2'))} & {fmt(mu('naive'))} & {pe_nai:.1e} \\
+MultiHead (prior art) & {f1_blk:.1e} & {fmt(pe('multihead','f2'))} & ${fmt(mu('multihead'))}\pm{fmt(sd('multihead'),2)}$ & {pe_mh:.1e} \\
+MultiHead+alloc & {f1_blk:.1e} & {fmt(pe('mh_alloc','f2'))} & ${fmt(mu('mh_alloc'))}\pm{fmt(sd('mh_alloc'),2)}$ & {pe_mha:.1e} \\
+MultiHead+alloc+soft & {f1_blk:.1e} & {fmt(pe('mh_alloc','f2'))} & ${c['mh_alloc_soft']['onset_mean']:.3f}\pm{c['mh_alloc_soft']['onset_std']:.2f}$ & {pe_mhas:.1e} \\
\textbf{{+CTX (final)}} & {pe('ctx_v3','f1'):.1e} & {fmt(f2_ctx)} & $\mathbf{{{fmt(mu('ctx_v3'))}\pm{fmt(sd('ctx_v3'),2)}}}$ & {pe_ctx:.1e} \\
\bottomrule
\end{{tabular}}""")

# ---------- Table 3a: CTX ladder ----------
w('table3a', rf"""\begin{{tabular}}{{lcl}}
\toprule
CTX layer & F3-onset NMSE & Interpretation \\
\midrule
Block-RLS & ${fmt(mu('block'))}\pm{fmt(sd('block'),2)}$ & baseline \\
+naive memory & {fmt(mu('naive'))} & consolidation relocates forgetting \\
+CTX v0 (self-masked) & {fmt(mu('ctx_v0'))} & reproduced defect: shared $W_{{\mathrm{{fast}}}}$ masks slot quality \\
+CTX v1 (slow-error) & ${fmt(mu('ctx_v1'))}\pm{fmt(sd('ctx_v1'),2)}$ & unmasking halves onset error \\
+CTX v2 (+protection, norm) & ${fmt(mu('ctx_v2'))}\pm{fmt(sd('ctx_v2'),2)}$ & mean gain n.s.\ ($p{{=}}0.116$); regressions in 3/10 seeds \\
+CTX v3 (+cons.\ count) & $\mathbf{{{fmt(mu('ctx_v3'))}\pm{fmt(sd('ctx_v3'),2)}}}$ & correct target \emph{{and}} low variance \\
\bottomrule
\end{{tabular}}""")

# ---------- Table 3b: competitor ladder ----------
w('table3b', rf"""\begin{{tabular}}{{lcl}}
\toprule
Competitor ladder & F3-onset NMSE & Interpretation \\
\midrule
MultiHead (prior art) & ${fmt(mu('multihead'))}\pm{fmt(sd('multihead'),2)}$ & passive protection never engages \\
MultiHead+alloc & ${fmt(mu('mh_alloc'))}\pm{fmt(sd('mh_alloc'),2)}$ & write-trigger channel: helps seeds 1, 6, 8 \\
MultiHead+alloc+soft & ${c['mh_alloc_soft']['onset_mean']:.3f}\pm{c['mh_alloc_soft']['onset_std']:.2f}$ & +normalized gate: borrows $\times1.6$ of gap \\
CTX (ours) & $\mathbf{{{fmt(mu('ctx_v3'))}\pm{fmt(sd('ctx_v3'),2)}}}$ & remaining $\approx$2.8$\times$ = dual-rate transfer \\
\bottomrule
\end{{tabular}}""")

# ---------- Table 4: cross-family ----------
w('table4', rf"""\begin{{tabular}}{{lccccc}}
\toprule
Protocol (5 seeds) & Block & MultiHead & CTX & Gain & Slots \\
\midrule
MG-17$\to$MG-30$\to$MG-17 & {CANON['t4_mg_block']} & {CANON['t4_mg_mh']} & {CANON['t4_mg_ctx']} & $\times${CANON['t4_mg_gain']} & {CANON['t4_mg_slots']} \\
NARMA-10$\to$MG-17$\to$NARMA-10 & {CANON['t4_na_block']} & {CANON['t4_na_mh']} & {CANON['t4_na_ctx']} & $\times${CANON['t4_na_gain']} & {CANON['t4_na_slots']} \\
\bottomrule
\end{{tabular}}""")

# ---------- Table 5: saturation ----------
w('table5', rf"""\begin{{tabular}}{{lcc}}
\toprule
Configuration & F4-onset NMSE & $\|W_{{\mathrm{{fast}}}}\|$ at boundary \\
\midrule
Block-RLS (memoryless) & ${np.mean(v['onset_block']):.3f}\pm{np.std(v['onset_block']):.3f}$ & {np.mean(v['normb_block']):.1f} \\
Block-RLS + clamp & ${CANON['t5_bc_onset']}\pm{CANON['t5_bc_std']}$ & {CANON['t5_bc_norm']} \\
$S{{=}}2$, refuse & ${np.mean(v['onset_r']):.3f}\pm{np.std(v['onset_r']):.3f}$ & {np.mean(v['normb_r']):.1f} \\
$S{{=}}2$, refuse + clamp & ${np.mean(v['onset_rc']):.3f}\pm{np.std(v['onset_rc']):.3f}$ & {np.mean(v['normb_rc']):.1f} \\
$S{{=}}2$, overwrite & $\mathbf{{{np.mean(v['onset_o']):.3f}\pm{np.std(v['onset_o']):.3f}}}$ & {np.mean(v['normb_o']):.2f} \\
$S{{=}}3$, overwrite & ${np.mean(s['onset_3o']):.3f}\pm{np.std(s['onset_3o']):.3f}$ & --- \\
$S{{=}}3$, refuse & ${np.mean(s['onset_3r']):.3f}\pm{np.std(s['onset_3r']):.3f}$ & --- \\
\bottomrule
\end{{tabular}}""")

# ---------- Table 6: drainage control ----------
w('table6', rf"""\begin{{tabular}}{{lcc}}
\toprule
Configuration & F3-onset NMSE & F3-end NMSE \\
\midrule
Block-RLS & ${fmt(mu('block'))}\pm{fmt(sd('block'))}$ & {pe_blk:.3f} \\
Block-RLS + clamp ($R{{=}}0.94$) & ${c['block_clamp']['onset_mean']:.3f}\pm{c['block_clamp']['onset_std']:.3f}$ & {c['block_clamp']['f3_end_mean']:.3f} \\
CTX (full) & $\mathbf{{{fmt(mu('ctx_v3'))}\pm{fmt(sd('ctx_v3'))}}}$ & {pe_ctx:.3f} \\
\bottomrule
\end{{tabular}}""")

# ---------- Table 7: cost envelope ----------
w('table7', r"""\begin{tabular}{lcc}
\toprule
Component & Ops/step & Memory (float64) \\
\midrule
ESN: $W_{\mathrm{res}}x + W_{\mathrm{in}}u$ + tanh & $\approx$1.1k MAC & $W_{\mathrm{res}}$ 8 kB + $W_{\mathrm{in}}$ 1 kB \\
Spline map & $O(DK)\approx10^{3}$ & $\phi$ buffer 8 kB \\
Block-RLS & $O(DK^{2})\approx10^{4}$ & $P$ 78 kB + $W_{\mathrm{fast}}$ 8 kB \\
Ctx.\ memory & $O(SF)\approx3\cdot10^{3}$ & $W_{\mathrm{slow}}$ 24 kB \\
\textbf{Total} & $\approx$130 \textmu s (Python, full update path) & \textbf{$\approx$0.13 MB} \\
\bottomrule
\end{tabular}""")

# ---------- Appendix A: per-seed ----------
kinds = ['block', 'naive', 'multihead', 'mh_alloc',
         'ctx_v0', 'ctx_v1', 'ctx_v2', 'ctx_v3']
v2r = f['onset_ctx_v2'] - f['onset_ctx_v1']  # >0 = regresyon
rows = []
for i in range(10):
    cells = [f"{f['onset_'+k][i]:.3f}" for k in kinds]
    if v2r[i] > 0:
        cells[6] += '$^{*}$'
    rows.append(f"{i} & " + " & ".join(cells) + r" \\")
mean_row = r"mean & " + " & ".join(
    f"{mu(k):.3f}" if k != 'ctx_v3' else rf"\textbf{{{mu(k):.3f}}}"
    for k in kinds) + r" \\"
w('tableA', "\\begin{tabular}{l" + "c" * 8 + "}\n\\toprule\n"
    "seed & Block & naive & MH & MH+alloc & v0 & v1 & v2 & v3 (final) \\\\\n"
    "\\midrule\n" + "\n".join(rows) + "\n\\midrule\n" + mean_row +
    "\n\\bottomrule\n\\end{tabular}")

rows2 = []
for i in range(10):
    rows2.append(f"{i} & {c['mh_soft']['per_seed'][i]:.3f} & "
                 f"{c['mh_alloc_soft']['per_seed'][i]:.3f} \\\\")
w('tableB', "\\begin{tabular}{lcc}\n\\toprule\n"
    "seed & MH+soft & MH+alloc+soft \\\\\n\\midrule\n" +
    "\n".join(rows2) + "\n\\midrule\n"
    f"mean & {c['mh_soft']['onset_mean']:.3f} & "
    f"{c['mh_alloc_soft']['onset_mean']:.3f} \\\\\n"
    "\\bottomrule\n\\end{tabular}")

# ---------- Makrolar: claim-bearing sayilar ----------
macros = {
    'ctxOnset': f"{mu('ctx_v3'):.3f}",
    'blockOnset': f"{mu('block'):.3f}",
    'headlineRatio': f"{mu('block')/mu('ctx_v3'):.2f}",
    'mhAllocSoft': f"{c['mh_alloc_soft']['onset_mean']:.3f}",
    'residualRatio': f"{c['mh_alloc_soft']['onset_mean']/mu('ctx_v3'):.1f}",
    'clampOnlyOnset': f"{c['block_clamp']['onset_mean']:.3f}",
    'ctxEnd': f"{pe_ctx:.3f}",
    'blockEnd': f"{pe_blk:.3f}",
    'clampEnd': f"{c['block_clamp']['f3_end_mean']:.3f}",
    'ctxEndPenalty': f"{pe_ctx/pe_blk:.1f}",
    'mhOnset': f"{mu('multihead'):.3f}",
    'mhAllocOnset': f"{mu('mh_alloc'):.3f}",
    'clampRatio': f"{mu('block')/c['block_clamp']['onset_mean']:.2f}",
    'memBeyondClamp': f"{c['block_clamp']['onset_mean']/mu('ctx_v3'):.2f}",
    'naiveOnset': f"{mu('naive'):.3f}",
}
with open('numbers_macros.tex', 'w', encoding='utf-8') as fh:
    for k, val in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
print('gen/table*.tex + numbers_macros.tex generated')
