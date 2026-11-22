"""
build_tex.py — paper_draft.md -> paper.tex
  * turns 10 markdown tables into gen/table*.tex + caption + label
  * places 6 figures at their anchor points
  * binds claim-bearing numbers to numbers_macros.tex macros
  * converts the body to latex via pandoc and wraps the preamble
"""
import re
import pypandoc

md = open('paper_draft.md', encoding='utf-8').read()

# --- drop the draft footer line; lift H1 out (title goes to preamble) ---
md = re.sub(r'\*Draft v2[^\n]*\*\n?', '', md)
md = re.sub(r'^# (.+)\n', '', md)  # H1 removed (title set in preamble)
# header levels: ## -> section, ### -> subsection
md = re.sub(r'^### ', '##SUB## ', md, flags=re.M)
md = re.sub(r'^## ', '# ', md, flags=re.M)
md = md.replace('##SUB## ', '## ')
# strip hand-written section numbers (LaTeX numbers automatically)
md = re.sub(r'^(#{1,3}) \d[\d.]*\.?\s+', r'\1 ', md, flags=re.M)

# ---------- 1) macro bindings (claim-bearing numbers) ----------
# Longer matches first; tokens are word-bounded.
subs = [
    ('0.459', r'\\blockOnset'),
    ('0.096', r'\\ctxOnset'),
    ('0.422', r'\\mhOnset'),
    ('0.268', r'\\mhAllocSoft'),
    ('0.369', r'\\mhAllocOnset'),
    ('0.222', r'\\clampOnlyOnset'),
    ('0.405', r'\\clampEnd'),
    ('0.037', r'\\ctxEnd'),
    ('0.456', r'\\naiveOnset'),
]
for a, b in subs:
    md = re.sub(r'(?<![\d.])' + re.escape(a) + r'(?![\d.])', b, md)
# ratios and contextual numbers
md = md.replace('×4.76', r'$\times$\headlineRatio')
md = md.replace('×2.8', r'$\times$\residualRatio')
md = md.replace('×2.07', r'$\times$\clampRatio')
md = md.replace('×2.30', r'$\times$\memBeyondClamp')
md = md.replace('3.6× worse', r'\ctxEndPenalty$\times$ worse')
md = md.replace('3.6×', r'\ctxEndPenalty$\times$')
md = md.replace('$0.010\\to\\ctxEnd$', r'$\blockEnd\to\ctxEnd$')
md = md.replace('(3.7e-2 vs 1.0e-2)', r'(\ctxEnd{} vs \blockEnd)')

# ---------- 2) tables: raw \begin{table}..\input blocks live in the
# md (single source of truth gen/table*.tex; no orphan-caption risk)

# ---------- 3) figures ----------
def fig(fname, cap, label, w_=r'\columnwidth'):
    return ('\\begin{figure}[t]\n\\centering\n'
            f'\\includegraphics[width={w_}]{{{fname}}}\n'
            f'\\caption{{{cap}}}\n\\label{{{label}}}\n\\end{{figure}}\n')

FIGS = [
    (r'Figure 1 illustrates the pipeline\.',
     fig('fig1_architecture.png',
         'Architecture. The ungated fast path $\\phi^\\top '
         'W_{\\mathrm{fast}}$ (red) is always active; the gated path '
         '$\\beta\\phi^\\top\\sum_i w_i W_{\\mathrm{slow}}^{(i)}$ '
         '(green) is selected by the gate. Consolidation migrates a '
         '$(1-\\gamma)/\\beta$-scaled copy into a slot and decays '
         '$W_{\\mathrm{fast}}$ by $\\gamma$ --- a lossless drainage '
         'valve. Two decision channels: prediction gate $w_i$ and '
         'write trigger $E_{\\min}\\lessgtr\\tau\\hat m_2$. The dashed '
         'norm cap is the lossy control of Table~\\ref{tab:table6}.',
         'fig:arch')),
    (r'## Main Protocol',
     fig('fig2_windup.png',
         'Covariance windup at $\\lambda{=}0.98$ (representative '
         'seed). Top: unbounded $\\mathrm{tr}(P)$ reaches '
         '$\\sim10^{48}$; the trace bound '
         '$\\Pi_{\\max}{=}10^4\\mathrm{tr}(P_0)$ is a chosen '
         'reference, persistently active --- the effective algorithm '
         'approaches normalized RLS. Bottom: rolling NMSE. The '
         'chain: bound crossing ($t{\\approx}465$) $\\to$ first '
         '$\\min\\mathrm{diag}(P){<}0$ ($t{=}1685$) $\\to$ sustained '
         'divergence ($t{=}2234$, rolling NMSE $>10\\times$ bounded '
         'run). Divergence is loss of positive-definiteness, not '
         'overflow.',
         'fig:windup')),
    (r'reach the same level\. Figure 4 visualizes',
     fig('fig3_error_trajectory.png',
         'Regime return, seed 0: rolling NMSE ($w{=}250$, normalized '
         'by $\\sigma_0$). At the square$\\to$sine boundary CTX '
         're-engages the stored sine slot while the memoryless '
         'baseline re-learns from scratch.',
         'fig:traj') + '\n' +
     fig('fig4_slot_heatmap.png',
         'Gate weights $w_k$ across the protocol (seed 0). Slot 0 '
         'captures the sine regime; the square phase fragments '
         'writes across slots 1--2 (novelty/unpredictability '
         'conflation, Sec. 5). At the sine return the gate selects '
         'the correct slot.',
         'fig:gate')),
    (r'Tables 3a–3b and Fig\. 5 isolate',
     fig('fig5_ablation_bar.png',
         'Ablation and competitor ladder, 10 seeds (mean$\\pm$std). '
         'Three families: baselines (grey), competitor ladder '
         '(orange, hatched), CTX ladder (blue). Annotations give the '
         'ratio against the memoryless baseline.',
         'fig:ablation')),
    (r'## Cost Profile',
     fig('fig6_valve.png',
         'Consolidation as drainage valve (4-regime overload, seed '
         '0). Top: $\\|W_{\\mathrm{fast}}\\|$ --- refuse lets it wind '
         'to $\\sim$35, overwrite drains to $\\sim$0.3, the clamp '
         'caps it at $R{=}3.18$. Bottom: rolling NMSE --- refuse '
         'loses the onset but wins the tail (it never pays the '
         '$\\gamma$-decay cost).',
         'fig:valve')),
]
# pass to append figures AFTER their anchor paragraph:
lines = md.split('\n')
out, pending = [], []
for ln in lines:
    out.append(ln)
    for anchor, fbody in FIGS:
        if re.search(anchor, ln) and fbody not in pending:
            pending.append(fbody)
    # paragraph end (blank line) with a pending figure -> insert
    if not ln.strip() and pending:
        out.append(pending.pop(0))
md = '\n'.join(out)
for fbody in pending:  # safety: append uninserted figures at the end
    md += '\n' + fbody

# ---------- 3.5) pandoc math-dollars rule: '$x$5' does not parse ---
# raw unicode arrow -> math equivalent (no glyph in Latin Modern);
# must precede the digit rule so '$\leftrightarrow$2' ->
# '$\leftrightarrow 2$'
md = md.replace('↔', r'$\leftrightarrow$')
# if a digit follows the closing $, pull the digit into math
md = re.sub(r'\$([^$\n]+)\$(\d+(?:\.\d+)?)', r'$\1 \2$', md)
# '~N' -> '$\sim N$' (pandoc leaves \textasciitilde)
md = re.sub(r'(?<![\\~\w])~(\d+(?:\.\d+)?)', r'$\\sim \1$', md)

# ---------- 4) pandoc: markdown -> latex body ----------
body = pypandoc.convert_text(
    md, 'latex', format='markdown+raw_tex+smart',
    extra_args=['--wrap=preserve'])

# Abstract -> abstract env; title handled separately
body = body.replace('\r\n', '\n').replace('\r', '\n')
# cross references (AFTER pandoc): 'Table N'/'Fig. N' -> \ref
_TABMAP = {'1': 'table1', '2': 'table2', '3a': 'table3a',
           '3b': 'table3b', '4': 'table4', '5': 'table5',
           '6': 'table6', '7': 'table7'}
body = re.sub(r'Tables 3a-+3b',
              'Tables~\\\\ref{tab:table3a}--\\\\ref{tab:table3b}', body)
body = re.sub(r'Table (3a|3b|[1-7])\b',
              lambda m: 'Table~\\ref{tab:' + _TABMAP[m.group(1)] + '}',
              body)
_FIGMAP = {'1': 'arch', '2': 'windup', '3': 'traj', '4': 'gate',
           '5': 'ablation', '6': 'valve'}
body = re.sub(r'(?:Fig\.|Figure) (\d)\b',
              lambda m: ('Fig.~' if 'Fig' in m.group(0) else 'Figure~')
              + '\\ref{fig:' + _FIGMAP[m.group(1)] + '}', body)
# long \texttt file names do not break -> \url (breakable)
def _tt2url(m):
    return '\\url{' + m.group(1).replace('\\_', '_') + '}'
body = re.sub(
    r'\\texttt\{((?:[A-Za-z0-9\\_]+\.(?:py|npz|json|tex))[A-Za-z0-9\\_]*)\}',
    _tt2url, body)
# macro + space -> \macro{} + space (LaTeX control words swallow it)
_MACROS = ('ctxOnset|blockOnset|headlineRatio|mhAllocSoft|residualRatio|'
           'clampOnlyOnset|ctxEnd|blockEnd|clampEnd|ctxEndPenalty|mhOnset|'
           'mhAllocOnset|clampRatio|memBeyondClamp|naiveOnset')
body = re.sub(r'(\\(?:' + _MACROS + r'))(\s)', r'\1{}\2', body)
body = re.sub(r'\\section\{Abstract\}\\label\{abstract\}\s*',
              '\\\\begin{abstract}\n', body, count=1)
# close the abstract body at the first \section
_body_i = body.find('\\begin{abstract}')
_body_j = body.find('\\section', _body_i)
body = (body[:_body_j] + '\\end{abstract}\n\n' + body[_body_j:]
        if _body_i >= 0 and _body_j > _body_i else body)
body = body.replace('\\section{References (verified)}',
                    '\\section*{References}')
body = body.replace('\\section{Appendix A: Per-seed Results}',
                    '\\appendix\n\\section{Per-seed Results}')
body = body.replace('\\section{Code and Data Availability}',
                    '\\section*{Code and Data Availability}')

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}%
  \setlength{\parskip}{0pt}}
\emergencystretch=2em
\input{numbers_macros.tex}

\title{Write-Protected Contextual Consolidation for
Backpropagation-Free Continual Learning in Reservoir-Based
Prediction}
\author{}
\date{}

\begin{document}
\maketitle
"""

with open('paper.tex', 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(PREAMBLE + body + "\n\\end{document}\n")
print('paper.tex written:',
      body.count('\\section'), 'sections,',
      body.count('begin{table'), 'table-envs,',
      body.count('includegraphics'), 'figures')
