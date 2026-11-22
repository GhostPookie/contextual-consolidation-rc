"""Phase-2b spike: does contextual memory rescue a frozen-LLM classifier
on a regime-return stream? (direct analog of the reservoir protocol)

Protocol (mirrors sine->square->sine):
  phase 1: domain A task (sci.space vs rec.sport)      600 samples
  phase 2: domain B task (comp.graphics vs politics) 600 samples
  phase 3: domain A again (held-out half)            400 samples

Every step: predict with current readout -> observe label -> local
update. Baseline = BlockDiagRLS (memoryless). Candidate = same RLS +
ContextualMemory. If memory works, phase-3 onset error drops sharply:
the A-slot is recalled instead of re-learned.

Features: frozen GPT-2 last-token hidden state -> per-dim standardize
-> seeded projection to D=100 -> cubic B-spline map -> RLS regression
to {0,1} (predict 1 iff score > 0.5).
"""

import numpy as np
import time, os, sys

import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.datasets import fetch_20newsgroups

from rls_engine import SplineFeatureMap, BlockDiagRLS
from contextual_memory import ContextualMemory
from continual_protocol import LAM, INIT_VAR, WINDUP_MULT

MODEL_ID = "openai-community/gpt2"
D_PROJ = 100
K = 10
ONSET = 100                     # onset window (samples) at phase-3 start
PA, PB = 600, 600
PA_RET = 400
MAX_TOK = 128
CACHE = "clsfeat_{name}.npz"


def load_domains():
    """Two binary tasks on disjoint category pairs (regime switch)."""
    domA = fetch_20newsgroups(subset="all",
                              categories=["sci.space", "rec.sport.baseball"],
                              remove=("headers", "footers", "quotes"),
                              shuffle=True, random_state=0)
    domB = fetch_20newsgroups(subset="all",
                              categories=["comp.graphics",
                                          "talk.politics.misc"],
                              remove=("headers", "footers", "quotes"),
                              shuffle=True, random_state=1)
    # label remap: first category -> 0, second -> 1 (sklearn targets are
    # category indices, already 0/1 within the subset)
    A = list(zip(domA.data, domA.target))
    B = list(zip(domB.data, domB.target))
    rng = np.random.default_rng(7)
    rng.shuffle(A); rng.shuffle(B)
    # phase-3 uses a disjoint half of A (no train/test leakage)
    return A[:PA], B[:PB], A[PA:PA + PA_RET]


@torch.no_grad()
def embed(texts, tok, mdl, dev, batch=32):
    """GPT-2 last-token hidden state per text."""
    H = np.empty((len(texts), mdl.config.n_embd), dtype=np.float32)
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_TOK).to(dev)
        out = mdl(**enc).last_hidden_state
        idx = enc["attention_mask"].sum(1) - 1
        H[i:i + batch] = out[torch.arange(len(idx)), idx].cpu().numpy()
    return H


def get_features(name, texts, tok, mdl, dev):
    cf = CACHE.format(name=name)
    if os.path.exists(cf):
        return np.load(cf)["X"]
    H = embed(texts, tok, mdl, dev)
    np.savez_compressed(cf, X=H)
    return H


def run_stream(Phi, Y, kind, e_gate=0.5, eta=2.0, alpha_ema=0.01):
    """predict-then-update; returns 0/1 decisions before update."""
    d = Phi.shape[1] // K
    base = BlockDiagRLS(d, K, 1, forgetting=LAM,
                        init_var=INIT_VAR, windup_mult=WINDUP_MULT)
    if kind == "ctxmem":
        model = ContextualMemory(base, n_slots=3, beta=1.0, gamma=0.1,
                                 eta=eta, alpha_ema=alpha_ema,
                                 consolidate_every=250,
                                 e_gate=e_gate, p_inflate=1.5)
    else:
        model = base
    preds = np.empty(len(Phi))
    for i, phi in enumerate(Phi):
        preds[i] = float(model.predict(phi)[0])
        model.update(phi, Y[i])
    return (preds > 0.5).astype(float), model


def phase_err(preds, y, lo, hi):
    return float(np.mean(preds[lo:hi] != y[lo:hi]))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    mdl = AutoModel.from_pretrained(MODEL_ID).to(dev).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    A1, B, A2 = load_domains()
    stream = A1 + B + A2
    texts = [t for t, _ in stream]
    Y = np.array([l for _, l in stream], dtype=float).reshape(-1, 1)

    t0 = time.perf_counter()
    H = np.concatenate([get_features("A1", [t for t, _ in A1], tok, mdl, dev),
                        get_features("B", [t for t, _ in B], tok, mdl, dev),
                        get_features("A2", [t for t, _ in A2], tok, mdl, dev)])
    t_feat = time.perf_counter() - t0

    # per-dim standardization over the whole stream (protocol-consistent)
    H = (H - H.mean(0)) / (H.std(0) + 1e-6)
    rng = np.random.default_rng(12345)
    R = rng.standard_normal((H.shape[1], D_PROJ)) / np.sqrt(H.shape[1])
    X_proj = np.tanh(H @ R)
    X_raw = np.clip(H, -3.0, 3.0) / 3.0   # spline map needs bounded input

    bounds = [0, PA, PA + PB, PA + PB + PA_RET]
    print(f"stream: A[{bounds[0]}:{bounds[1]}] B[{bounds[1]}:{bounds[2]}] "
          f"A'[{bounds[2]}:{bounds[3]}]")
    for tag, X in (("proj-100", X_proj), ("raw-768", X_raw)):
        fmap = SplineFeatureMap(X.shape[1], K)
        Phi = np.stack([fmap.transform(x) for x in X])
        print(f"--- features: {tag} (F={Phi.shape[1]})")
        for lab, kw in (("block", {}),
                        ("ctxmem eg=.5", dict(e_gate=0.5)),
                        ("ctxmem eg=1", dict(e_gate=1.0)),
                        ("ctxmem fast-gate",
                         dict(e_gate=1.0, eta=5.0, alpha_ema=0.05))):
            t0 = time.perf_counter()
            preds, model = run_stream(Phi, Y,
                                      "ctxmem" if "ctx" in lab else "block",
                                      **kw)
            dt = time.perf_counter() - t0
            p3_onset = phase_err(preds, Y[:, 0], bounds[2],
                                 bounds[2] + ONSET)
            p3_end = phase_err(preds, Y[:, 0], bounds[3] - ONSET,
                               bounds[3])
            p1_end = phase_err(preds, Y[:, 0], bounds[1] - ONSET,
                               bounds[1])
            p2_end = phase_err(preds, Y[:, 0], bounds[2] - ONSET,
                               bounds[2])
            nc = getattr(model, "n_consolidations", "-")
            print(f"{lab:>16}: P1-end={p1_end:.3f} P2-end={p2_end:.3f} "
                  f"P3-onset={p3_onset:.3f} P3-end={p3_end:.3f} "
                  f"cons={nc} ({dt:.0f}s)", flush=True)

    print(f"\nfeature extraction: {t_feat:.0f}s total | "
          f"peak VRAM: {torch.cuda.max_memory_allocated()/1e6:.0f} MB"
          if dev == "cuda" else f"features: {t_feat:.0f}s")


if __name__ == "__main__":
    main()
