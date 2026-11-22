"""Phase-2 spike: is the mechanism reservoir-specific, or feature-agnostic?

Same protocol as continual_protocol.py (sine->square->sine, same seeds,
same onset metric), but the feature extractor is a FROZEN LLM (GPT-2,
last hidden state of a text-encoded input window) instead of the ESP
reservoir. Readout pipeline unchanged: SplineFeatureMap + BlockDiagRLS +
ContextualMemory.

Pipeline per step:
  last L values of u -> text "0.52 0.48 ..." -> GPT-2 last hidden state
  h_t (768) -> per-dim standardization -> random projection R -> x_t (D=100)
  -> spline features -> RLS / ContextualMemory (identical code)

If contextual consolidation still cuts phase-3 onset error vs the
memoryless block baseline, the mechanism transfers to pretrained
features - i.e. it is an adapter for frozen models, not an RC trick.
"""

import numpy as np
import time, os, sys

import torch
from transformers import AutoModel, AutoTokenizer

from reservoir import ReservoirEngine          # not used; kept for parity
from rls_engine import SplineFeatureMap
from continual_protocol import (build_stream, run_method, window_nmse,
                                waveform, WASHOUT, PHASE_LEN, PHASES,
                                D, K, SEEDS, WINDOW, PERIOD)


def build_stream_jit(seed, phases=None, phase_len=None):
    """build_stream + per-seed phase/frequency jitter.

    build_stream(seed) is deterministic apart from the reservoir init,
    so on LLM features every seed would run the identical stream. Jitter
    restores seed variance: phase offset + +-10% period perturbation.
    """
    phases = PHASES if phases is None else phases
    phase_len = PHASE_LEN if phase_len is None else phase_len
    rng = np.random.default_rng(1000 + seed)
    off = rng.uniform(0, PERIOD)
    per = PERIOD * rng.uniform(0.9, 1.1)
    total = WASHOUT + phase_len * len(phases)
    u = np.zeros(total)
    for t in range(total):
        tt = t + off
        s = np.sin(2.0 * np.pi * tt / per)
        if t >= WASHOUT and phases[(t - WASHOUT) // phase_len] == "square":
            u[t] = np.sign(s)
        else:
            u[t] = s
    mu, sd = u.mean(), u.std()
    u = (u - mu) / sd
    nxt = (waveform(phases[-1], total - WASHOUT) - mu) / sd
    y = np.empty(total)
    y[:-1] = u[1:]
    y[-1] = nxt
    return u, y

MODEL_ID = "openai-community/gpt2"
L = 16                     # input window encoded as text
D_PROJ = 100               # projected feature dim (matches reservoir D)
CACHE = "llm_feat_j_s{seed}.npz"


def encode_window(u, t):
    lo = max(0, t - L + 1)
    w = u[lo:t + 1]
    if lo == 0 and len(w) < L:
        w = np.concatenate([np.zeros(L - len(w)), w])
    return " ".join(f"{v:.2f}" for v in w)


@torch.no_grad()
def llm_features(u_all, seed, tok, mdl, dev, batch=64):
    """h_t for every t, batched; per-dim standardize; project to D_PROJ."""
    cache = CACHE.format(seed=seed)
    if os.path.exists(cache):
        z = np.load(cache)
        return z["X"]
    texts = [encode_window(u_all, t) for t in range(len(u_all))]
    H = np.empty((len(texts), mdl.config.n_embd), dtype=np.float32)
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt",
                  padding=True).to(dev)
        out = mdl(**enc).last_hidden_state          # (B, T, 768)
        idx = enc["attention_mask"].sum(1) - 1       # last real token
        H[i:i + batch] = out[torch.arange(len(idx)), idx].cpu().numpy()
    # per-dim standardization on the whole stream (same protocol-level
    # treatment as u's own mu/sd normalization in build_stream)
    H = (H - H.mean(0)) / (H.std(0) + 1e-6)
    rng = np.random.default_rng(12345)               # fixed across seeds
    R = rng.standard_normal((H.shape[1], D_PROJ)) / np.sqrt(H.shape[1])
    X = np.tanh(H @ R)                               # squash to reservoir range
    np.savez_compressed(cache, X=X)
    return X


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    mdl = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    fmap = SplineFeatureMap(D_PROJ, K)
    res = {"block": [], "ctxmem": []}
    t_feat = t_run = 0.0
    for seed in range(seeds):
        u, y = build_stream_jit(seed)
        sig_var = float(np.var(y))

        t0 = time.perf_counter()
        X = llm_features(u, seed, tok, mdl, dev)
        Phi = np.stack([fmap.transform(X[t])
                        for t in range(WASHOUT, len(u))])
        t_feat += time.perf_counter() - t0

        Y = y[WASHOUT:].reshape(-1, 1)
        for kind in res:
            t0 = time.perf_counter()
            errs, _ = run_method(kind, Phi, Y, sig_var)
            t_run += time.perf_counter() - t0
            # phase-3 onset NMSE (first WINDOW steps of final phase)
            lo = 2 * PHASE_LEN
            res[kind].append(window_nmse(errs, y, lo, lo + WINDOW))
        print(f"seed {seed}: block={res['block'][-1]:.3f} "
              f"ctxmem={res['ctxmem'][-1]:.3f}", flush=True)

    b, c = np.array(res["block"]), np.array(res["ctxmem"])
    print("\n=== LLM-feature spike (sine->square->sine) ===")
    print(f"model={MODEL_ID} dev={dev} proj={D_PROJ} window={L} "
          f"seeds={seeds}")
    print(f"phase-3 onset NMSE  block : {b.mean():.3f} +- {b.std():.3f}")
    print(f"phase-3 onset NMSE  ctxmem: {c.mean():.3f} +- {c.std():.3f}")
    print(f"ratio block/ctxmem        : {b.mean()/max(c.mean(),1e-9):.2f}x")
    print(f"feature time/seed={t_feat/seeds:.1f}s "
          f"train time/seed={t_run/seeds:.1f}s")
    if dev == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")


if __name__ == "__main__":
    main()
