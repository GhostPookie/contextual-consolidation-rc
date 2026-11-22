# Tuning-Free Continual Adaptation of Frozen Language Models via Write-Protected Contextual Memory

*Phase-2 draft skeleton — target: arXiv preprint / workshop short paper (4–6 pp). All numbers from `spike_results.json`; do not type numbers by hand into prose without checking against the file.*

---

## Abstract (draft)

Continual adaptation of large language models to shifting domains typically relies on gradient-based low-rank adapters (LoRA), which require per-model learning-rate tuning and incur VRAM costs that grow with batch size and model scale through retained activations and optimizer state. We attach a backpropagation-free adapter — a block-diagonal recursive least-squares readout over spline-expanded frozen features, coupled to a write-protected contextual memory — to GPT-2 family models (124M/355M/774M) on a single-pass domain-return classification stream. Across scales, the adapter's return-onset error stays flat (0.18–0.24) while identically-configured online LoRA collapses outside a narrow learning-rate window that shifts with model size (divergence at lr=1e-3 on 355M/774M, no learning at 1e-5, competitive accuracy only at 1e-4). On a 6GB device with frozen gpt2-large, LoRA's peak memory exceeds the physical wall at batch 4 and reaches 9.2GB at batch 8, while the adapter remains under the wall at batch 32 (5.1GB) with ~400KB of feature-space state independent of model size. Tuning-free, forward-only adaptation is feasible where gradient methods cannot be tuned or cannot fit. (Speed comparison is reported only as a scoped lower bound — update-path vs full training-step — not a headline claim.)

## 1. Introduction (~0.75 pp)

- Problem: continual/domain-shift adaptation of frozen LLMs in streaming settings.
- Why existing methods strain: gradient adapters need (i) learning rates tuned per model scale, (ii) backward passes retaining activations, (iii) optimizer state.
- Our claim: an adapter with *no learning rate* and *no backward pass* — RLS + write-protected contextual memory over frozen features. Strict constraints: single pass, predict-before-update, no replay buffer, no offline batch.
- Contributions: (1) same mechanism transfers from reservoirs to frozen GPT-2; (2) measured scale-stability vs LoRA lr-fragility; (3) measured VRAM divergence on real 6GB hardware.
- NOTE (read-tour fix): the two structural properties are INDEPENDENT — no-lr comes from RLS being second-order; VRAM-flat comes from forward-only (no retained activations). Do NOT chain them causally in prose ("because it needs no lr, it avoids the wall" is wrong). Common root: local forward-only learning.

## 2. Related Work (~0.5 pp)

- Parameter-efficient adaptation: LoRA/adapters, prompt tuning.
- Continual learning w/o replay: regularization (EWC), architectural (slots/memory).
- Gradient-free/local learning: reservoir computing, RLS readouts, forward-only methods.

## 3. Method (~1 pp)

### 3.1 Frozen backbone → feature stream

- Model weights frozen (`requires_grad=False`); single forward per example under `torch.no_grad()`. The learning/inference boundary: the backbone is pure inference — all learning happens feature-space-side, so no gradient graph is ever built through the model.
- Feature: last-token hidden state of final block, d ∈ {768, 1024, 1280} for GPT-2 s/m/l. Adapter sees only φ_t ∈ R^d — a fixed interface, model-size-agnostic by construction.
- Online standardization of features (note: current spikes use whole-stream statistics — flagged limitation, future runs must use streaming mean/var).
- Optional fixed random projection (768→100, tanh) shown to bottleneck probe accuracy (90.25%→81.25%); main experiments use standardized raw features.

### 3.2 Spline expansion + block-diagonal RLS

- Per-dimension cubic B-spline features; block-diagonal RLS readout, exact per-step update, no learning rate (second-order; step size implicit in covariance P).
- Learning state: W, P blocks over spline features — ~250–410KB, lives on CPU, independent of model size.

### 3.3 Write-protected contextual memory

- Fast ungated channel (W_fast, always written) + N gated slow slots (consolidated, protected).
- Consolidation trigger rescaled to task error baseline (e_gate=1.0 for 0/1 labels; the regression-calibrated e_fast ≤ e_gate·E[y²] nearly disables consolidation on classification).
- Gate time constant must be << regime-transition window: α_ema=0.01 (~100-step lag) poisons onset via stale slots; α_ema=0.05, η=5.0 reduces return-onset error 0.30→0.23.

### 3.4 Why no lr / why VRAM-flat

- RLS has no step-size hyperparameter: update magnitude set by Kalman gain from P.
- Forward-only: activations transient, freed per batch — peak = weights + O(B·d·L) transient, not weights + activations + grads + 2×optimizer state.

## 4. Experiments (~2 pp)

### 4.1 Setup
- 20 Newsgroups domain-return stream A→B→A′, single pass, predict-before-update.
- Baselines: same-adapter block-RLS (no memory), online LoRA r=8 (q,v) + AdamW.
- Metrics: phase-end / return-onset error, peak VRAM (torch.cuda.max_memory_allocated), per-step ms.
- Finding order rationale: 4.2 establishes the adapter is stable where LoRA needs tuning; 4.3 shows *why* (the tuning window itself shifts with scale); 4.4 shifts axis from accuracy to hardware cost. Findings 2–3 are presented as independent properties, not a causal chain.

### 4.2 Finding 1 — scale-stability

> Table 1 reports return-onset error for both arms across the GPT-2 family. With identical configuration — no learning rate exists to retune — the adapter's onset error stays within 0.18–0.24 from 124M to 774M parameters. Identically configured online LoRA (r=8, lr=1e-3, a setting that works well at 124M) degrades to 0.49–0.52 on the two larger models, i.e., near-chance. The adapter's learning dynamics live entirely in feature space (φ_t ∈ R^d, d ≤ 1280): the readout and memory see the same dimensionality regardless of how many parameters produced the features, so scale changes the backbone, not the learning problem.

- Data: APIA onset 0.23/0.18/0.20 at 124M/355M/774M; LoRA(lr=1e-3) 0.19/0.49/0.52.

### 4.3 Finding 2 — lr fragility

> Section 4.2's LoRA collapse is not a fundamental failure of low-rank adaptation — it is a tuning failure. Table 2 sweeps lr ∈ {1e-3, 1e-4, 1e-5} on gpt2-medium and gpt2-large. On medium, lr=1e-3 diverges (0.49 onset), lr=1e-4 attains 0.09, and lr=1e-5 never learns (0.51). On large the pattern repeats: lr=1e-3 diverges (0.52), lr=1e-4 attains the best onset of the whole study (0.06), and lr=1e-5 only partially learns within the stream (0.32 onset, still improving at 0.18 by phase end). Two observations matter for deployment: (i) the usable window is narrow — a factor of 10 in either direction destroys or cripples the run; (ii) the window's position is not predictable from the smaller model — 1e-3 is optimal at 124M and fatal at 355M+. In a streaming setting there is no held-out pass for a grid search; the stream arrives once. The adapter sidesteps the question entirely: RLS has no step-size parameter, and the identical configuration that runs at 124M runs at 774M.

- Data: medium lr∈{1e-3,1e-4,1e-5} → onset 0.49/0.09/0.51; large → 0.52/0.06/0.32.
- Honest note: tuned LoRA beats APIA decisively on accuracy (0.06–0.09 vs 0.18–0.20 onset) — the gap grows with scale, and this must be stated, not hidden. The claim is tunability + memory, not accuracy.

### 4.4 Finding 3 — VRAM wall

> Figure 3 changes the axis from accuracy to hardware cost. On a 6GB device running frozen gpt2-large (seq-128), peak VRAM for the forward-only adapter grows from 3,243MB at batch 1 to 5,132MB at batch 32 — transient activations freed under `no_grad`, plus ~400KB of feature-space state on CPU. The LoRA arm (fwd+bwd+AdamW) reaches 6,191MB at batch 4 — already past the device's physical 6,144MB — and requests 9,159MB at batch 8, forcing ~4GB of spill into shared RAM where step time inflates 2.2× (575→1,267ms) from memory pressure, not compute. Note this is the independent property, not a corollary of §4.3: even a perfectly tuned LoRA pays this memory bill, because retained activations and optimizer state are structural to gradient descent, not to any particular lr.

- Figure (fig_vram_wall): APIA 3,243→5,132MB at B=1→32; LoRA 3,989→9,159MB at B=1→8, over 6GB wall at B≥4.
- Timing caveat: scopes differ (feature-space RLS vs full fwd+bwd+opt) — speed gap is a lower bound, not a headline; memory divergence is the structural claim.

## 5. Discussion (~0.75 pp)

- Unified message: two INDEPENDENT structural properties of local forward-only learning — (a) second-order updates ⇒ no tunable step size ⇒ scale-stable accuracy; (b) no backward graph ⇒ adaptation memory scales with features, not parameters ⇒ stays under hardware walls where fwd+bwd+optimizer cannot fit.
- Limitations: (a) accuracy gap vs tuned LoRA is real and grows with scale (0.06–0.09 vs 0.18–0.20 onset at 355M/774M) — we trade accuracy for tunability + memory; (b) stream-wide feature normalization (online deployment needs streaming stats); (c) GPT-2 family only, d ≤ 1280; (d) single binary-classification stream; (e) LoRA arm is a single r=8 config — rank/schedule variants unexplored.
- Future work: quantized RLS, larger backbones (6GB wall on 1.3B+), hardware mapping.

## 6. Conclusion (~0.25 pp)

- Tuning-free forward-only adaptation works under walls where gradient adapters cannot be tuned or cannot fit.

---

## Figures & tables to generate
- Fig 1: architecture diagram (reuse fig1 style + LLM branch).
- Fig 2: lr-sweep — LoRA onset vs lr (3 pts, log x) + APIA flat line.
- Fig 3: fig_vram_wall.png (extend APIA arm to B=32).
- Tab 1: scale sweep; Tab 2: lr sweep; Tab 3: VRAM/timing by batch.

## Data sources (single truth)
`spike_results.json` — keys: per-model `apia`/`lora`/`lr_*`, `vram_scaling`.
Scripts: `cls_spike.py`, `adapter_spike.py`, `scale_spike.py`, `vram_spike.py`.
