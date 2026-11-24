# Write-Protected Contextual Consolidation

Backpropagation-free, fully local, online continual learning for
reservoir-based time-series prediction.

A fixed Echo State Network reservoir (ESP-constrained) feeds a
cubic B-spline feature map and a block-diagonal online RLS readout.
A dual-rate contextual memory — one ungated fast weight channel plus
S gated slow slots — consolidates regime-specific knowledge under
write protection, cutting regime-return onset error ~4.8x versus the
memoryless baseline (10 seeds, sine->square->sine protocol).

All learning is local and online: no backpropagation, no replay
buffer, no offline batch passes. ~130 us/step, ~0.13 MB (float64)
in the pure-NumPy prototype.

## Architecture

![System architecture](fig1_architecture.png)

A fixed ESP-constrained reservoir drives a per-dimension cubic
B-spline expansion. One ungated fast-weight channel always learns;
S gated slow slots consolidate under write protection.

## Headline result

![Regime-return error trajectory](fig3_error_trajectory.png)

When the stream returns to sine after a square-wave regime, the
contextual memory's onset NMSE is ~4.8x lower than the memoryless
block baseline (0.096 vs 0.459, mean over 10 seeds).

![Ablation summary](fig5_ablation_bar.png)

Factorial ablations separate the two mechanisms: stored context
content vs. implicit drainage of the ungated channel. The matched-norm
clamp control recovers only part of the gain — slot content matters.

## Contents

| File | Role |
|---|---|
| `reservoir.py` | ESP-constrained ESN + cubic B-spline feature map |
| `rls_engine.py` | block-diagonal RLS with trace bound |
| `contextual_memory.py` | fast/slow slots, gate, write protection, transfer |
| `memory.py`, `multihead.py` | naive-memory and Bereska-style competitors |
| `continual_protocol.py` | stream builder + main sine->square->sine protocol |
| `benchmark_block_rls.py` | readout characterization (paper Table 1) |
| `different_tasks.py` | cross-family generalization (NARMA/MG-30) |
| `saturation_test.py` | slot-overload policies (overwrite/refuse/clamp) |
| `valve_test.py` | drainage-valve analysis (paper Fig. 6) |
| `clamp_control.py` | matched-norm clamp control (paper Table 7) |
| `make_figures.py`, `fig1_architecture.py`, `fig2_windup.py` | figures + `metrics.json`/`fig_data.npz` |
| `make_tables.py` | generated tables (`gen/`) + `numbers_macros.tex` |
| `audit_numbers.py` | 24-assert data audit: every claim vs data files |
| `build_tex.py`, `lint_tex.py`, `lint_pdf.py` | md->tex build + render lint |
| `metrics.json`, `fig_data.npz`, `saturation_data.npz`, `valve_data.npz`, `clamp_control.json` | measured data (single source of truth) |
| `llm_spike.py`, `cls_spike.py`, `adapter_spike.py`, `scale_spike.py` | Phase-2 experiments: same mechanism on frozen GPT-2 features — time-series, domain-return classification, APIA-vs-LoRA head-to-head, 124M->774M scale sweep |
| `vram_spike.py` | Phase-2: gpt2-large batch sweep — APIA fwd-only vs LoRA fwd+bwd+AdamW peak VRAM (`vram_scaling` key) |
| `fig_phase2_scale.py`, `fig_vram_wall.py`, `fig_vram_dual.py` | README figure generators (data from `spike_results.json`) |
| `spike_results.json` | Phase-2 spike measurements (P3 errors, VRAM, ms/step, lr-sweep, vram_scaling) |
| `build_tex2.py`, `fig1_architecture_p2.py`, `fig2_lr_sweep.py` | Phase-2 paper build + figures (`arxiv2_src/`) |
| `native_apia_lm.py`, `_chunked.py` | Phase-3: native APIA char-LM — Kaggle kernels (gate-fix, hard-gate, forced-trigger, detect arms) |
| `mamba_apia_spike.py`, `mamba_apia_results.json` | Phase-2 Mamba probe: 2x2 + multi-layer grid (negative result, paper-2 §5.5) |
| `analyze_gate_E.py`, `normalize_gate_e.py` | gate-diagnostic tooling for the v3 mechanism-repair chain |
| `kaggle_*_out/` | per-run Kaggle outputs: `native_lm_v3_*.json` + kernel logs |

## Reproduce

```bash
pip install -r requirements.txt   # Python 3.13, NumPy 2.5.2
python make_figures.py            # main protocol, 10 seeds -> metrics.json
python different_tasks.py         # cross-family, 5 seeds
python saturation_test.py         # -> saturation_data.npz
python valve_test.py              # -> valve_data.npz
python clamp_control.py           # -> clamp_control.json
python audit_numbers.py           # 24 asserts vs data
python make_tables.py             # gen/table*.tex + numbers_macros.tex
python build_tex.py               # paper.tex (needs pandoc)
python lint_pdf.py                # render-level checks (needs tectonic+pdftotext)
```

Every run uses `np.random.default_rng(seed)` — seeds 0-9 (main),
0-4 (cross-family). Results are bit-reproducible.

## Phase-2: APIA on frozen LLM backbones

Same mechanism mounted on frozen LLM features (GPT-2 family, OPT-1.3B,
Mamba-130M/1.4B probed), tested against online LoRA on a domain-return
classification stream (`spike_results.json` for all numbers).
Phase-2 paper: `paper2_draft.md` -> `arxiv2_src/paper2.tex`.

![APIA on a frozen LLM](fig1_architecture_p2.png)

![Scale sweep: APIA vs online LoRA](fig_phase2_scale.png)

The honest finding is not "APIA is more accurate" — tuned LoRA wins
on accuracy (0.09 vs 0.18 onset). It is that APIA needs **no learning
rate at all** and stays flat (0.18-0.24) from 124M to 774M while the
same LoRA config collapses; updates cost ~1 ms regardless of model
size vs 61-196 ms. See `llm_spike.py`, `cls_spike.py`,
`adapter_spike.py`, `scale_spike.py`.

### The scaling wall (6GB desktop + 16GB T4)

Same protocol on frozen `gpt2-large` (6GB device) and OPT-1.3B fp16
(16GB T4), batch sweeps (`vram_spike.py`, `kaggle_1p3b_vram.py`):

![VRAM wall at two hardware tiers](fig_vram_dual.png)

| batch | APIA peak | LoRA peak |
|---|---|---|
| 1 | 3,243 MB | 3,989 MB |
| 4 | 3,425 MB | **6,191 MB** |
| 8 | 3,670 MB | **9,159 MB** |
| 32 | 5,132 MB | — (already past the wall) |

Batch 1→32: APIA stays under the wall the whole way (+1,889 MB of
transient forward activations only — `no_grad`, nothing retained for
backward). LoRA grows +5,170 MB by batch 8 alone (retained
activations + grads + AdamW state), exceeding the 6GB physical wall
at batch 4 — beyond it, Windows spills ~4 GB into shared RAM and step
time balloons (575→1,267 ms, spill overhead not compute). **APIA's
learning-state cost scales with the feature dimension (~400 KB), not
the model — the structural reason a frozen 100B backbone could adapt
in a fixed memory budget.** Timing scopes differ (APIA update is
feature-space RLS; LoRA is full fwd+bwd+opt), so the speed gap is a
lower bound, not a headline. On the 16GB tier the wall moves but is
not removed: LoRA OOMs at batch 64 while APIA peaks at 8.4 GB.

## Papers

- Phase-1 (reservoir): `paper_draft.md` -> `arxiv_src/` (19 pp).
- Phase-2 (frozen LLM): `paper2_draft.md` -> `arxiv2_src/paper2.tex`.
