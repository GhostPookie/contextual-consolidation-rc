"""build_tex2.py — paper2_draft.md -> arxiv2_src/paper2.tex

Phase-2 variant of build_tex.py:
  * strips draft scaffolding (page budgets, working-note tail sections)
  * converts the four md pipe tables into gen/table*.tex (real table envs
    so \\ref works) and replaces them in-place with \\begin{table} wrappers
  * unicode math chars -> LaTeX equivalents BEFORE pandoc
  * figures anchored at their prose mention points
"""
import os
import re
import pypandoc

SRC = 'paper2_draft.md'
OUT_DIR = 'arxiv2_src'
GEN = os.path.join(OUT_DIR, 'gen')
os.makedirs(GEN, exist_ok=True)

md = open(SRC, encoding='utf-8').read()

# ---------- 0) strip draft scaffolding ----------
md = re.sub(r'^\*Phase-2 draft skeleton[^\n]*\*\n?', '', md, flags=re.M)
md = re.sub(r'^# (.+)\n', '', md)                       # H1 -> preamble title
md = re.sub(r'\(~[\d.]+ pp\)', '', md)                 # page budgets
md = re.sub(r'^## Figures & tables to generate.*$', '', md,
            flags=re.M | re.S)                         # drop working tail
# bullets that are notes-to-self or prose-as-bullet -> paragraphs
md = re.sub(r'^- Honest note: ', 'Honest note. ', md, flags=re.M)
md = re.sub(r'^- Figure 3 \(fig_vram_dual\)[^\n]*\n', '', md, flags=re.M)
md = re.sub(r'^- Timing caveat: ', 'Timing caveat. ', md, flags=re.M)
md = re.sub(r'^- Figure \(fig_vram_wall\)[^\n]*\n', '', md, flags=re.M)
# internal working notes -> drop entirely
md = re.sub(r'^- Strict constraints \(keep in prose[^\n]*\n', '', md,
            flags=re.M)
md = re.sub(r'^- NOTE \(read-tour fix\)[^\n]*\n', '', md, flags=re.M)
md = re.sub(r'^> \*Finding order[^\n]*\*\n?', '', md, flags=re.M)
md = re.sub(r'^> \*(claim|message|framing)[^\n]*\*\n?', '', md,
            flags=re.M | re.I)

# ---------- 1) tables -> gen/table*.tex + inline table env ----------
CAPS = {
    1: ("Return-onset error across scales and families (mean$\\pm$std, "
        "3 seeds; OPT-1.3B single-seed). Identical configuration "
        "throughout; the adapter has no lr to retune."),
    2: ("Learning-rate fragility of online LoRA (r=8; mean$\\pm$std, "
        "3 seeds; OPT-1.3B single-seed). The viable window is narrow "
        "and shifts with scale; the pattern repeats on a different "
        "model family."),
    3: ("Peak VRAM on a 6,144MB device (frozen gpt2-large, seq-128). "
        "APIA grows only via transient forward activations; the "
        "$\\sim$400KB learning state is CPU-resident."),
    4: ("Peak VRAM on a 15,636MB T4 (frozen OPT-1.3B fp16, seq-128, "
        "Kaggle). Same divergence at the higher ceiling: LoRA OOMs at "
        "B=64 while the adapter uses $\\sim$54\\% of the device."),
}

BODIES = {
    1: r"""\begin{tabular}{lllll}
\toprule
Model & Params & APIA onset & LoRA (lr=1e-3) & LoRA status \\
\midrule
GPT-2 & 124M & $0.32{\pm}0.04$ & $0.46{\pm}0.04$ & seed-dependent \\
GPT-2 Medium & 355M & $0.28{\pm}0.05$ & $0.36{\pm}0.17$ & unstable/late div. \\
GPT-2 Large & 774M & $0.32{\pm}0.05$ & $0.47{\pm}0.04$ & diverged \\
OPT-1.3B (fp16, T4) & 1.3B & 0.15 (n=1) & 0.46 (n=1) & diverged \\
\bottomrule
\end{tabular}""",
    2: r"""\begin{tabular}{lllll}
\toprule
Model & lr & Onset err & Phase-end & Observation \\
\midrule
GPT-2 Medium & 1e-3 & $0.36{\pm}0.17$ & $0.49{\pm}0.01$ & late divergence \\
GPT-2 Medium & 1e-4 & \textbf{0.15$\bm{\pm}$0.05} & $0.08{\pm}0.02$ & optimal \\
GPT-2 Medium & 1e-5 & $0.47{\pm}0.05$ & $0.40{\pm}0.05$ & no learning \\
GPT-2 Large & 1e-3 & $0.47{\pm}0.04$ & $0.54{\pm}0.04$ & divergence \\
GPT-2 Large & 1e-4 & \textbf{0.14$\bm{\pm}$0.04} & $0.08{\pm}0.01$ & optimal (a priori) \\
GPT-2 Large & 1e-5 & $0.34{\pm}0.07$ & $0.12{\pm}0.07$ & partial/slow \\
OPT-1.3B (T4) & 1e-3 & 0.46 (n=1) & 0.60 (n=1) & divergence (cross-family) \\
OPT-1.3B (T4) & 1e-4 & \textbf{0.05} (n=1) & 0.04 (n=1) & optimal \\
\bottomrule
\end{tabular}""",
    3: r"""\begin{tabular}{lllll}
\toprule
Batch & APIA peak & LoRA peak & LoRA step & Status \\
\midrule
1 & 3,243 & 3,989 & 268 ms & fits \\
4 & 3,425 & \textbf{6,191} & 575 ms & exceeds wall \\
8 & 3,670 & \textbf{9,159} & 1,267 ms & $\sim$4GB spill \\
32 & 5,132 & --- & --- & APIA still fits \\
\bottomrule
\end{tabular}""",
    4: r"""\begin{tabular}{lllll}
\toprule
Batch & APIA peak & LoRA peak & LoRA/APIA & Status \\
\midrule
1 & 2,731 & 2,961 & 1.08$\times$ & both fit \\
4 & 3,002 & 3,839 & 1.28$\times$ & both fit \\
8 & 3,361 & 5,008 & 1.49$\times$ & both fit \\
16 & 4,082 & 7,345 & 1.80$\times$ & both fit \\
32 & 5,523 & 12,023 & 2.18$\times$ & LoRA near wall \\
64 & 8,406 & \textbf{OOM} & --- & LoRA $>$16GB wall \\
\bottomrule
\end{tabular}""",
}

# replace each md table block (header sep .. caption line) with latex
for n in (1, 2, 3, 4):
    pat = re.compile(
        r'^\|[^\n]*\n\|[^\n]*\n(?:\|[^\n]*\n)+\n?\*Table ' + str(n) +
        r':[^*]*\*\n?', re.M)
    open(f'{GEN}/table{n}.tex', 'w', encoding='utf-8').write(BODIES[n] + '\n')
    env = ('\\begin{table}[t]\n\\centering\\small\n'
           f'\\input{{gen/table{n}.tex}}\n'
           f'\\caption{{{CAPS[n]}}}\n\\label{{tab:table{n}}}\n'
           '\\end{table}\n')
    md = pat.sub(lambda _m, e=env: e, md)

# '~N' -> '$\sim N$' must run BEFORE the '§'->'Sec.~' replacement
# (else the '~' we just inserted gets re-matched below)
md = re.sub(r'(?<![\\~\w])~(\d+(?:\.\d+)?)', r'$\\sim \1$', md)

# ---------- 2a) composed unicode formulas -> clean latex (pre-map) ----------
PRE = [
    ('E_i = ||y_t − β φ_tᵀ W_slow^(i)||²',
     r'$E_i = \|y_t - \beta\,\phi_t^\top W_{\mathrm{slow}}^{(i)}\|^2$'),
    ('(1−γ)/β·W_fast', r'$(1-\gamma)/\beta \cdot W_{\mathrm{fast}}$'),
]
for a, b in PRE:
    md = md.replace(a, b)

# ---------- 2) unicode -> latex ----------
U = [
    ('m\u0302\u2082', r'$\widehat{m_2}$'),          # m̂₂ first (composed)
    ('\u0302', ''),                                 # stray combining hat
    ('\u2082', '$_2$'),
    ('\u2032', "'"),
    ('\u1d40', r'$^\top$'),
    ('\u00b7', r'$\cdot$'),
    ('\u00d7', r'$\times$'),
    ('\u2192', r'$\to$'),
    ('\u2190', r'$\leftarrow$'),
    ('\u2208', r'$\in$'),
    ('\u03c6_t', r'$\phi_t$'),
    ('\u03c6', r'$\phi$'),
    ('\u00b2', '$^2$'),
    ('\u03b1_ema', r'$\alpha_{\mathrm{ema}}$'),
    ('\u03b1', r'$\alpha$'),
    ('\u03b3', r'$\gamma$'),
    ('\u03b2', r'$\beta$'),
    ('\u03b7', r'$\eta$'),
    ('\u03c4', r'$\tau$'),
    ('\u2264', r'$\leq$'),
    ('\u00b1', r'$\pm$'),
    ('\u2248', r'$\approx$'),
    ('\u2212', '-'),
    ('\u2013', '--'),
    ('\u00a7', 'Sec.~'),
]
for a, b in U:
    md = md.replace(a, b)
md = md.replace('E[y^2]', r'$\mathbb{E}[y^2]$')
md = md.replace('R^d', r'$\mathbb{R}^d$')
md = md.replace('R^{d$\cdot$K+1}', r'$\mathbb{R}^{d\cdot K+1}$')
md = md.replace('O(d$\cdot$K$^2$)', r'$O(d\cdot K^2)$')

# ---------- 2b) ASCII math tokens -> wrapped (post-map) ----------
POST = [
    ('$\\phi_t$$^\\top$', r'$\phi_t^\top$'),
    ('W_slow^(i)', r'$W_{\mathrm{slow}}^{(i)}$'),
    ('W_slow^(k)', r'$W_{\mathrm{slow}}^{(k)}$'),
    ('W_fast', r'$W_{\mathrm{fast}}$'),
    ('W_slow', r'$W_{\mathrm{slow}}$'),
    ('e_fast', r'$e_{\mathrm{fast}}$'),
    ('e_gate', r'$e_{\mathrm{gate}}$'),
    ('E_min', r'$E_{\min}$'),
    ('argmin_i c_i', r'$\arg\min_i c_i$'),
    ('argmin c_i', r'$\arg\min c_i$'),
    ('E[y$^2$]', r'$\mathbb{E}[y^2]$'),
]
for a, b in POST:
    md = md.replace(a, b)

# pandoc math-dollars rule: if a digit follows the closing $, pull it in
md = re.sub(r'\$([^$\n]+)\$(\d+(?:\.\d+)?)', r'$\1 \2$', md)

# ---------- 3) headers ----------
md = re.sub(r'^### ', '##SUB## ', md, flags=re.M)
md = re.sub(r'^## ', '# ', md, flags=re.M)
md = md.replace('##SUB## ', '## ')
md = re.sub(r'^(#{1,3}) \d[\d.]*\.?\s+', r'\1 ', md, flags=re.M)
md = re.sub(r'^# Abstract \(draft\)', '# Abstract', md, flags=re.M)

# ---------- 4) figures ----------
def fig(fname, cap, label):
    return ('\\begin{figure}[t]\n\\centering\n'
            f'\\includegraphics[width=\\columnwidth]{{{fname}}}\n'
            f'\\caption{{{cap}}}\n\\label{{{label}}}\n\\end{{figure}}\n')

FIGS = [
    (r'^# Method',
     fig('fig1_architecture_p2.png',
         'Architecture. A frozen backbone (GPT-2 or OPT, '
         '$\\texttt{requires\\_grad}=\\mathrm{False}$, '
         '$\\texttt{no\\_grad}$) emits a last-token feature $\\phi_t$; a '
         'fixed cubic B-spline map lifts it to $\\mathbb{R}^{d\\cdot '
         'K+1}$. The ungated fast channel $W_{\\mathrm{fast}}$ (red) is a '
         'block-diagonal RLS updated every step --- no learning rate, '
         'no backward graph. The gated slow slots '
         '$W_{\\mathrm{slow}}^{(i)}$ (green) receive a '
         '$(1{-}\\gamma)/\\beta$-scaled copy of $W_{\\mathrm{fast}}$ '
         'only when the write trigger fires and the target passes '
         'write protection. The whole learning state is CPU-resident.',
         'fig:arch')),
    (r'The contextual-memory arm tells a more nuanced story',
     fig('fig2_lr_sweep.png',
         'Learning-rate fragility (return-onset error vs lr, log '
         'scale). LoRA forms a U-shape that bottoms near lr=1e-4 on '
         'both model sizes but diverges at 1e-3 and fails to learn at '
         '1e-5; the APIA band is flat because RLS has no step size.',
         'fig:lrsweep')),
    (r'not remove it, because the retained-activation bill is '
     r'proportional to batch while the forward-only cost is not',
     fig('fig_vram_dual.png',
         'Peak VRAM vs batch at two hardware tiers. Left: 6GB device, '
         'frozen gpt2-large --- LoRA crosses the wall at B=4 and spills '
         '$\\sim$4GB at B=8. Right: 16GB T4, frozen OPT-1.3B --- same '
         'divergence delayed, not removed (LoRA OOM at B=64; adapter at '
         '8.4GB). X marks the measured OOM point.',
         'fig:vram')),
]
lines = md.split('\n')
out, pending = [], []
for ln in lines:
    out.append(ln)
    for anchor, fbody in FIGS:
        if re.search(anchor, ln) and fbody not in pending:
            pending.append(fbody)
    if not ln.strip() and pending:
        out.append(pending.pop(0))
md = '\n'.join(out)
for fbody in pending:
    md += '\n' + fbody

# ---------- 5) pandoc ----------
body = pypandoc.convert_text(
    md, 'latex', format='markdown+raw_tex+smart',
    extra_args=['--wrap=preserve'])
body = body.replace('\r\n', '\n').replace('\r', '\n')

# refs
body = re.sub(r'Tables 1--2',
              'Tables~\\\\ref{tab:table1}--\\\\ref{tab:table2}', body)
for n in (1, 2, 3, 4):
    body = re.sub(r'Table ' + str(n) + r'\b',
                  'Table~\\\\ref{tab:table%d}' % n, body)
_FIGMAP = {'1': 'arch', '2': 'lrsweep', '3': 'vram'}
body = re.sub(r'(?:Fig\.|Figure) ([123])\b',
              lambda m: ('Fig.~' if 'Fig' in m.group(0) else 'Figure~')
              + '\\ref{fig:' + _FIGMAP[m.group(1)] + '}', body)
body = body.replace('Sec.\\textasciitilde', 'Sec.~')

# abstract env
body = re.sub(r'\\section\{Abstract\}\\label\{abstract\}\s*',
              '\\\\begin{abstract}\n', body, count=1)
_i = body.find('\\begin{abstract}')
_j = body.find('\\section', _i)
if _i >= 0 and _j > _i:
    body = body[:_j] + '\\end{abstract}\n\n' + body[_j:]
# url for code-ish names
body = re.sub(r'\\texttt\{((?:[A-Za-z0-9\\_]+\.(?:py|json|tex|npz))'
              r'[A-Za-z0-9\\_]*)\}',
              lambda m: '\\url{' + m.group(1).replace('\\_', '_') + '}',
              body)

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}%
  \setlength{\parskip}{0pt}}
\emergencystretch=2em

\title{Tuning-Free Continual Adaptation of Frozen Language Models via
Write-Protected Contextual Memory}
\author{Anonymous}
\date{}

\begin{document}
\maketitle
"""

REFS = r"""
\section*{References}
\begin{itemize}\tightlist
\item E.~J. Hu et al., ``LoRA: Low-Rank Adaptation of Large Language
  Models,'' \emph{ICLR}, 2022. arXiv:2106.09685.
\item N. Houlsby et al., ``Parameter-Efficient Transfer Learning for
  NLP,'' \emph{ICML}, 2019.
\item J. Kirkpatrick et al., ``Overcoming catastrophic forgetting in
  neural networks,'' \emph{PNAS} 114(13), 2017.
\item F. Zenke, B. Poole, S. Ganguli, ``Continual learning through
  synaptic intelligence,'' \emph{ICML}, 2017.
\item H. Jaeger, ``The `echo state' approach to analysing and training
  recurrent neural networks,'' GMD Report 148, 2001.
\item S. Haykin, \emph{Adaptive Filter Theory}, 5th ed., Pearson, 2014
  (recursive least squares).
\item A. Radford et al., ``Language models are unsupervised multitask
  learners,'' OpenAI Tech. Report, 2019 (GPT-2).
\item S. Zhang et al., ``OPT: Open pre-trained transformer language
  models,'' arXiv:2205.01068, 2022.
\end{itemize}
"""

with open(f'{OUT_DIR}/paper2.tex', 'w', encoding='utf-8', newline='\n') as fh:
    fh.write(PREAMBLE + body + REFS + "\n\\end{document}\n")
print('paper2.tex written:', body.count('\\section'), 'sections,',
      body.count('begin{table'), 'tables,',
      body.count('includegraphics'), 'figures')
