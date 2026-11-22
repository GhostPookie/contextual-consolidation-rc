# Tuning-Free Continual Adaptation of Frozen Language Models via Write-Protected Contextual Memory

*Phase-2 draft skeleton — target: arXiv preprint / workshop short paper (4–6 pp). All numbers from `spike_results.json`; do not type numbers by hand into prose without checking against the file.*

---

## Abstract (draft)

Continual adaptation of large language models to shifting domains typically relies on gradient-based low-rank adapters (LoRA), which require per-model learning-rate tuning and incur VRAM costs that grow with batch size and model scale through retained activations and optimizer state. We attach a backpropagation-free adapter — a block-diagonal recursive least-squares readout over spline-expanded frozen features, coupled to a write-protected contextual memory — to GPT-2 family models (124M/355M/774M) on a single-pass domain-return classification stream. Across scales, the adapter's return-onset error stays flat (0.18–0.24) while identically-configured online LoRA collapses outside a narrow learning-rate window that shifts with model size (divergence at lr=1e-3 on 355M/774M, no learning at 1e-5, competitive accuracy only at 1e-4). On a 6GB device with frozen gpt2-large, LoRA's peak memory exceeds the physical wall at batch 4 and reaches 9.2GB at batch 8, while the adapter remains under the wall at batch 32 (5.1GB) with ~400KB of feature-space state independent of model size. Tuning-free, forward-only adaptation is feasible where gradient methods cannot be tuned or cannot fit. (Speed comparison is reported only as a scoped lower bound — update-path vs full training-step — not a headline claim.)

## 1. Introduction (~0.75 pp)

> Continual adaptation of large language models (LLMs) to shifting data distributions is typically addressed via gradient-based parameter-efficient fine-tuning (PEFT) methods, such as LoRA. While effective in offline settings, these methods face two fundamental bottlenecks in streaming, resource-constrained environments: (1) they require careful, model-specific learning rate tuning, where a suboptimal choice leads to catastrophic divergence or failure to learn; and (2) they incur substantial VRAM overhead due to the retention of backward-pass activations and optimizer states, which scales unfavorably with batch size and model parameters.
>
> We propose a fundamentally different approach: a backpropagation-free, tuning-free adapter based on block-diagonal recursive least squares (RLS) coupled with write-protected contextual memory, operating entirely in the frozen feature space of the LLM. Because RLS is a second-order method, it possesses no step-size hyperparameter, eliminating the need for learning rate tuning. Furthermore, by strictly enforcing a forward-only, predict-before-update regime, our adapter avoids building a gradient graph, keeping peak VRAM dependent only on transient feature activations, not on model scale or optimizer state.
>
> In this work, we demonstrate this mechanism on a single-pass domain-return classification stream using the GPT-2 family (124M to 774M parameters). We show that while identically configured online LoRA collapses outside a narrow, scale-dependent learning rate window, our adapter maintains stable return-onset error (0.18–0.24) across all scales. Finally, we empirically map the VRAM divergence on a 6GB device, showing that the adapter remains well within hardware limits even at large batch sizes, whereas the gradient-based baseline exceeds physical memory and incurs severe spill-overhead.

- Strict constraints (keep in prose or setup): single pass, predict-before-update, no replay buffer, no offline batch.
- NOTE (read-tour fix): the two structural properties are INDEPENDENT — no-lr comes from RLS being second-order; VRAM-flat comes from forward-only (no retained activations). Do NOT chain them causally ("because it needs no lr, it avoids the wall" is wrong). Common root: local forward-only learning.

## 2. Related Work (~0.5 pp)

> **Parameter-Efficient Fine-Tuning (PEFT).** Methods like LoRA [Hu et al., 2021] and adapters [Houlsby et al., 2019] reduce trainable parameters but remain fundamentally gradient-based. They require backward passes, retaining activations in memory, and optimizer states (e.g., Adam), making their VRAM footprint structurally tied to batch size and sequence length, regardless of the low rank.
>
> **Continual Learning without Replay.** Catastrophic forgetting is often mitigated via regularization (EWC, SI) or architectural isolation. However, these typically assume access to gradient signals or stored data buffers. Our work aligns with memory-based continual learning but operates strictly under single-pass, no-replay, and no-gradient constraints.
>
> **Gradient-Free and Local Learning.** Reservoir computing and online RLS have long provided efficient, forward-only adaptation for time-series prediction. We extend this paradigm to frozen LLM feature streams, demonstrating that second-order, local update rules can provide robust, tuning-free adaptation where first-order gradient descent is either too fragile to tune or too expensive to fit.

## 3. Method (~1 pp)

> Our approach attaches a backpropagation-free, stateful adapter to the frozen representations of a pre-trained language model. The architecture strictly enforces a predict-before-update, single-pass constraint, ensuring that no gradient graph is ever constructed through the backbone.
>
> **3.1 Frozen backbone and feature stream.** The base model weights are frozen (`requires_grad=False`), and inference is executed under `torch.no_grad()`. For each incoming sequence we extract the last-token hidden state of the final transformer block, yielding φ_t ∈ R^d, d ∈ {768, 1024, 1280} for the GPT-2 family. The adapter operates exclusively in this feature space: because the interface dimension d is bounded and independent of backbone parameter count, the learning problem is structurally identical whether features come from a 124M or a 774M model. In the reported experiments features are standardized with stream-wide statistics; online deployment requires a streaming mean/variance estimator (Sec. 5). An optional fixed random projection (768→100, tanh) was found to bottleneck probe accuracy (90.25%→81.25%) and is not used in the main experiments.
>
> **3.2 Spline expansion and block-diagonal RLS.** To capture non-linear boundaries without backpropagation, φ_t is expanded by a fixed per-dimension cubic B-spline basis (R^d → R^{d·K+1}, K=10, plus bias). The expanded representation feeds a block-diagonal recursive least-squares readout. RLS is second-order: the update magnitude is set implicitly by the inverse-covariance blocks P, so there is no step-size hyperparameter to tune. The block-diagonal approximation keeps per-step cost at O(d·K²) and the full learning state (W_fast, P) at ~250–410KB on CPU — independent of backbone size.
>
> **3.3 Write-protected contextual memory.** The readout splits into a fast, ungated channel (W_fast, updated every step on the gated residual) and N gated slow slots (W_slow^(i)). Consolidation runs periodically (every `consolidate_every` steps) but only fires when the short-term fast error is stable, e_fast ≤ e_gate·m̂₂, where m̂₂ is an online estimate of E[y²] — a trigger that must be rescaled to the task's error baseline (e_gate=1.0 for 0/1 labels; the regression-calibrated 0.5 nearly disables consolidation on classification). Write protection then selects the target: if the best-matching slot's context-match error E_i = ||y_t − β φ_tᵀ W_slow^(i)||² — computed *excluding* W_fast to avoid self-masking — satisfies E_min < τ·m̂₂, that slot is reinforced; otherwise the write goes to the least-consolidated slot (argmin_i c_i), so heavily consolidated memories are not overwritten by transient noise. At saturation the 'refuse' policy skips the write entirely. The transfer preserves prediction continuity: W_slow^(k) ← W_slow^(k) + (1−γ)/β·W_fast, then W_fast ← γ·W_fast (γ=0.1).
>
> **3.4 Gate time constant.** The slot-match EMA must be much faster than the regime-transition window: the default α_ema=0.01 (~100-step lag) leaves a stale slot active at domain return and poisons the onset; α_ema=0.05 with sharper gating (η=5.0) reduced return-onset error 0.30→0.23 on raw-768 features.
>
> **3.5 Why no lr / why VRAM-flat (independent properties).** No-lr follows from RLS being second-order (update magnitude set by P). VRAM-flat follows from forward-only execution: activations are transient and freed per batch — peak ≈ weights + O(B·d·L) transient, not weights + activations + grads + 2×optimizer state. Both share the root of local forward-only learning but are NOT causally chained.

## 4. Experiments (~2 pp)

### 4.1 Experimental Setup

> **Tasks and protocol.** We evaluate continual adaptation on a single-pass, domain-return classification stream built from 20 Newsgroups: phase 1 learns domain A (sci.space vs rec.sport.baseball, 600 samples), phase 2 domain B (comp.graphics vs talk.politics.misc, 600 samples), phase 3 returns to fresh samples of domain A′ (400 samples). Strict online constraints throughout: single pass, predict-before-update, no replay buffer, no offline batch training.
>
> **Models and baselines.** The adapter attaches to the frozen GPT-2 backbone (124M/355M/774M; `requires_grad=False`, `torch.no_grad()` inference). The learning boundary is strict: the backbone provides pure inference features (last-token hidden state, d ∈ {768, 1024, 1280}) and all adaptation happens feature-space-side. Two arms: (i) **APIA** — block-diagonal RLS over spline-expanded features + write-protected contextual memory (S=3 slots), learning state ≈250–410KB on CPU, independent of model scale; (ii) **online LoRA** — r=8 on `c_attn`/`c_proj`, AdamW, updated per sample on the same stream.
>
> *Feature normalization caveat: reported experiments standardize with stream-wide statistics; edge deployment requires a streaming estimator (Sec. 5).*
>
> **Metrics.** Phase-end and return-onset error (first 100 samples of A′ — adaptation speed and forgetting); peak VRAM via `torch.cuda.max_memory_allocated()`; per-step latency.
>
> **Hardware.** GPT-2-scale runs on an RTX 3060 Laptop (6GB) deliberately below the wall; the 1.3B extension (OPT-1.3B) targets a 16GB Kaggle T4 to re-test divergence at a higher ceiling.
>
> *Finding order: 4.2 establishes stability where LoRA needs tuning; 4.3 shows why (the window shifts with scale); 4.4 shifts the axis from accuracy to hardware cost. Findings 2–3 are independent properties, not a causal chain.*

### 4.2 Finding 1 — scale-stability

> Table 1 reports return-onset error for both arms across the GPT-2 family. With identical configuration — no learning rate exists to retune — the adapter's onset error stays within 0.18–0.24 from 124M to 774M parameters. Identically configured online LoRA (r=8, lr=1e-3, a setting that works well at 124M) degrades to 0.49–0.52 on the two larger models, i.e., near-chance. The adapter's learning dynamics live entirely in feature space (φ_t ∈ R^d, d ≤ 1280): the readout and memory see the same dimensionality regardless of how many parameters produced the features, so scale changes the backbone, not the learning problem.

| Model Scale | Parameters | APIA Onset | LoRA (lr=1e-3) Onset | LoRA Status |
| :--- | :--- | :--- | :--- | :--- |
| GPT-2 | 124M | 0.23 | 0.19 | Optimal |
| GPT-2 Medium | 355M | 0.18 | 0.49 | Diverged |
| GPT-2 Large | 774M | 0.20 | 0.52 | Diverged |

*Table 1: APIA remains stable across scales without tuning, while a fixed LoRA configuration (lr=1e-3) collapses as model size increases.*

### 4.3 Finding 2 — lr fragility

> Section 4.2's LoRA collapse is not a fundamental failure of low-rank adaptation — it is a tuning failure. Table 2 sweeps lr ∈ {1e-3, 1e-4, 1e-5} on gpt2-medium and gpt2-large. On medium, lr=1e-3 diverges (0.49 onset), lr=1e-4 attains 0.09, and lr=1e-5 never learns (0.51). On large the pattern repeats: lr=1e-3 diverges (0.52), lr=1e-4 attains the best onset of the whole study (0.06), and lr=1e-5 only partially learns within the stream (0.32 onset, still improving at 0.18 by phase end). Two observations matter for deployment: (i) the usable window is narrow — a factor of 10 in either direction destroys or cripples the run; (ii) the window's position is not predictable from the smaller model — 1e-3 is optimal at 124M and fatal at 355M+. In a streaming setting there is no held-out pass for a grid search; the stream arrives once. The adapter sidesteps the question entirely: RLS has no step-size parameter, and the identical configuration that runs at 124M runs at 774M.

| Model | Learning Rate | Return-Onset Error | Phase-End Error | Observation |
| :--- | :--- | :--- | :--- | :--- |
| GPT-2 Medium | 1e-3 | 0.49 | 0.53 | Divergence |
| GPT-2 Medium | 1e-4 | **0.09** | 0.06 | Optimal |
| GPT-2 Medium | 1e-5 | 0.51 | 0.51 | No learning |
| GPT-2 Large | 1e-3 | 0.52 | 0.51 | Divergence |
| GPT-2 Large | 1e-4 | **0.06** | 0.02 | Optimal (but unpredictable) |
| GPT-2 Large | 1e-5 | 0.32 | 0.18 | Partial/slow learning |

*Table 2: The viable learning-rate window is narrow and shifts with scale. Streaming deployment cannot afford this grid-search.*

- Honest note: tuned LoRA beats APIA decisively on accuracy (0.06–0.09 vs 0.18–0.20 onset) — the gap grows with scale, and this must be stated, not hidden. The claim is tunability + memory, not accuracy.

### 4.4 Finding 3 — VRAM wall

> Figure 3 changes the axis from accuracy to hardware cost. On a 6GB device running frozen gpt2-large (seq-128), peak VRAM for the forward-only adapter grows from 3,243MB at batch 1 to 5,132MB at batch 32 — transient activations freed under `no_grad`, plus ~400KB of feature-space state on CPU. The LoRA arm (fwd+bwd+AdamW) reaches 6,191MB at batch 4 — already past the device's physical 6,144MB — and requests 9,159MB at batch 8, forcing ~4GB of spill into shared RAM where step time inflates 2.2× (575→1,267ms) from memory pressure, not compute. Note this is the independent property, not a corollary of §4.3: even a perfectly tuned LoRA pays this memory bill, because retained activations and optimizer state are structural to gradient descent, not to any particular lr.

| Batch Size | APIA Peak VRAM | LoRA Peak VRAM | LoRA Step Time | Status |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 3,243 MB | 3,989 MB | 268 ms | Fits |
| 4 | 3,425 MB | **6,191 MB** | 575 ms | **Exceeds 6GB wall** |
| 8 | 3,670 MB | **9,159 MB** | 1,267 ms | ~4GB spill to shared RAM |
| 32 | 5,132 MB | — | — | APIA still fits |

*Table 3: Peak VRAM on the 6,144MB device (frozen gpt2-large, seq-128). APIA grows only by transient forward activations; the ~400KB learning state is CPU-side. LoRA blows past the wall at B=4 — retained backward activations + grads + AdamW state are structural to gradient descent.*

> The wall moves, but does not disappear. Repeating the same scan on a frozen OPT-1.3B backbone (fp16, weights 2,632MB) on a 16GB T4, the adapter grows 2,731→8,406MB from B=1 to B=64 — under half the device at every point. The LoRA arm reaches 12,023MB at B=32 and OOMs at B=64; the LoRA/APIA peak ratio grows monotonically with batch (1.08× → 2.18×). More headroom delays the divergence; it does not remove it, because the retained-activation bill is proportional to batch while the forward-only cost is not.

| Batch Size | APIA Peak VRAM | LoRA Peak VRAM | LoRA/APIA Ratio | Status |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 2,731 MB | 2,961 MB | 1.08× | Both fit |
| 4 | 3,002 MB | 3,839 MB | 1.28× | Both fit |
| 8 | 3,361 MB | 5,008 MB | 1.49× | Both fit |
| 16 | 4,082 MB | 7,345 MB | 1.80× | Both fit |
| 32 | 5,523 MB | 12,023 MB | 2.18× | LoRA near wall |
| 64 | 8,406 MB | **OOM** | — | **LoRA exceeds 16GB wall** |

*Table 4: Peak VRAM on a 15,636MB T4 (frozen OPT-1.3B fp16, seq-128, Kaggle). The same divergence pattern repeats at the higher ceiling; the LoRA arm reaches OOM at B=64 while the adapter still uses ~54% of the device.*

- Figure 3 (fig_vram_dual): two panels, one wall each — 6GB gpt2-large and 16GB OPT-1.3B.
- Timing caveat: scopes differ (feature-space RLS vs full fwd+bwd+opt) — speed gap is a lower bound, not a headline; memory divergence is the structural claim. On the T4 the scope gap is explicit: LoRA runs 26–32ms/sample fwd+bwd+opt at all measured batch sizes while APIA amortizes to 0.3ms/sample at B=64.

## 5. Discussion (~0.75 pp)

> The empirical results establish a unified message: local, forward-only learning possesses two independent structural advantages over gradient-based adaptation in streaming environments. First, because RLS is second-order, it has no step-size hyperparameter, yielding scale-stable accuracy without per-model tuning. Second, because it requires no backward pass, its adaptation memory scales strictly with the feature dimension d, not the backbone size, allowing it to operate beneath hardware VRAM walls where forward+backward+optimizer pipelines fail. These are independent properties sharing a common root (forward-only execution), not a causal chain.
>
> **Limitations.** We state the boundaries of this proof-of-concept transparently:
> 1. **Accuracy vs. tunability trade-off:** when perfectly tuned (lr=1e-4), LoRA reaches 0.09/0.06 onset on medium/large vs the adapter's 0.18/0.20 — and the gap grows with scale. We explicitly trade peak accuracy for elimination of tuning fragility and guaranteed memory bounds.
> 2. **Feature normalization:** current experiments use stream-wide statistics; true online deployment requires a streaming mean/variance tracker.
> 3. **Scope:** accuracy claims cover the GPT-2 family only (d ≤ 1280) on a single binary domain-return stream; the VRAM scan extends to OPT-1.3B fp16 but without an accuracy arm on that backbone. The LoRA baseline is one r=8 config (rank/schedule variants unexplored).
> 4. **Gate latency:** the slot-match EMA imposes a lag of order 1/α_ema steps after each regime shift; at the default 0.01 the stale slot poisons the onset (measured 0.30→0.23 with α_ema=0.05).
>
> **Future work.** The 1.3B wall is now measured (Table 4): next steps are an accuracy arm on OPT-scale backbones, still larger models (2.7B/6.7B) where fp16 weights alone approach the 16GB ceiling, quantized (int8/int4) RLS to shrink adapter state, dynamic slot allocation on severe context mismatch, and direct mapping of the O(d·K²) update to FPGA/analog hardware.

## 6. Conclusion (~0.25 pp)

> We have demonstrated that tuning-free, forward-only adaptation is a viable and structurally superior alternative to gradient-based adapters in streaming, hardware-constrained settings. By operating exclusively in the frozen feature space with block-diagonal RLS and write-protected contextual memory, our method eliminates the learning rate tuning burden and decouples adaptation memory from model scale. Empirical results on the GPT-2 family confirm that while online LoRA is highly fragile to learning rate choices that shift with model size, our adapter remains stable. Furthermore, on a 6GB device, the adapter avoids the VRAM wall that forces gradient-based methods into severe memory-spill overhead. This establishes a clear path toward deployable, lifelong learning for large models at the edge.

---

## Figures & tables to generate
- Fig 1: architecture diagram (reuse fig1 style + LLM branch).
- Fig 2: lr-sweep — LoRA onset vs lr (3 pts, log x) + APIA flat line.
- Fig 3: fig_vram_dual.png (two hardware tiers: 6GB gpt2-large, 16GB OPT-1.3B).
- Tab 1: scale sweep; Tab 2: lr sweep; Tab 3: VRAM/timing by batch (6GB); Tab 4: VRAM by batch (16GB T4).

## Data sources (single truth)
`spike_results.json` — keys: per-model `apia`/`lora`/`lr_*`, `vram_scaling` (6GB gpt2-large), `vram_scaling_opt13b` (16GB T4 OPT-1.3B fp16).
Scripts: `cls_spike.py`, `adapter_spike.py`, `scale_spike.py`, `vram_spike.py`, `kaggle_1p3b_vram.py` (Kaggle T4 notebook).
