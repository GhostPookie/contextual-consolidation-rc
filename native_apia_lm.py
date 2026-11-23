# native_apia_lm.py — native APIA substrate as a language model.
#
# Phase-3 spike: the model IS the memory. No frozen backbone, no
# attention, no backprop through the substrate. A fixed recurrent
# reservoir emits features; a block-diagonal RLS readout (optionally
# write-protected contextual memory) learns next-char prediction
# strictly online, predict-before-update, single pass.
#
# Baselines (torch): same-param GRU and decoder-only transformer
# trained online single-pass (loss recorded BEFORE optimizer.step —
# prequential evaluation, same honesty as the APIA arms).
#
# Substrate swap point: `Substrate` protocol. A stacked local-target
# variant (predictive-coding-style layers) can implement the same
# interface later without touching the readout/eval code.
#
# Usage:
#   python native_apia_lm.py --arm all --chars 300000 --seed 0
#   python native_apia_lm.py --arm apia_ctxmem --smoke   # quick check

import argparse
import json
import os
import time
import urllib.request
from abc import ABC, abstractmethod

import numpy as np

from reservoir import ReservoirEngine
from rls_engine import SplineFeatureMap, BlockDiagRLS
from contextual_memory import ContextualMemory

SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/"
                   "char-rnn/master/data/tinyshakespeare/input.txt")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "tinyshakespeare.txt")

# ---------- data ----------

def load_chars(path=DATA_PATH, url=SHAKESPEARE_URL):
    if not os.path.exists(path):
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return np.array([stoi[c] for c in text], dtype=np.int64), len(chars)

# ---------- substrate interface ----------

class Substrate(ABC):
    """Maps a stream of discrete tokens to a real-valued feature stream.

    Swap point for the v2 stacked local-target architecture: anything
    implementing step()->ndarray plugs into the same readout + eval.
    """
    n_features: int
    param_floats: int      # fixed substrate parameters (untrained)
    state_bytes: int       # bytes of learning-relevant state

    @abstractmethod
    def reset(self): ...
    @abstractmethod
    def step(self, tok: int) -> np.ndarray: ...


class ReservoirSubstrate(Substrate):
    """ESP reservoir + fixed spline expansion. The recurrent core is
    NEVER trained (no credit-assignment problem); all learning lives
    in the readout. Phase-1 engine, reused."""

    def __init__(self, vocab: int, n_res: int = 512, n_basis: int = 10,
                 leak: float = 0.4, inp_scale: float = 1.0,
                 sparsity: float = 0.1, seed: int = 0):
        self.vocab, self.n_res = vocab, n_res
        self.res = ReservoirEngine(
            input_dim=vocab, reservoir_size=n_res, sparsity=sparsity,
            leakage_rate=leak, input_scaling=inp_scale, seed=seed)
        self.spline = SplineFeatureMap(input_dim=n_res, n_basis=n_basis,
                                       lo=-1.0, hi=1.0)
        self.n_features = self.spline.n_features
        self._oh = np.zeros(vocab)

    def reset(self):
        self.res.reset()

    def step(self, tok: int) -> np.ndarray:
        self._oh[:] = 0.0
        self._oh[tok] = 1.0
        h = self.res.step(self._oh)
        return self.spline.transform(np.tanh(h))

    @property
    def param_floats(self):
        return (self.res.W_res.nnz + self.res.W_in.size)

    @property
    def state_bytes(self):
        return (self.res.W_res.data.nbytes + self.res.W_in.nbytes
                + self.res.state.nbytes)


# ---------- apia arm ----------

def run_apia(stream, vocab, arm, seed, n_res=512, n_basis=10,
             eval_every=2000, warm=500, verbose=True):
    """Predict-before-update char LM. y = one-hot next char."""
    sub = ReservoirSubstrate(vocab, n_res=n_res, n_basis=n_basis,
                             seed=seed)
    rls = BlockDiagRLS(input_dim=n_res, n_basis=n_basis,
                       n_outputs=vocab, forgetting=0.9995,
                       init_var=1e2, windup_mult=1e4)
    model = rls
    if arm == "apia_ctxmem":
        model = ContextualMemory(
            rls, n_slots=3, beta=1.0, gamma=0.1, eta=5.0,
            alpha_ema=0.05, alpha_fast=0.1, consolidate_every=250,
            e_gate=1.0, tau=0.5, p_inflate=1.5)
    elif arm != "apia_block":
        raise ValueError(arm)

    rng = np.random.default_rng(seed + 999)
    eye = np.eye(vocab)
    ln2 = np.log(2.0)
    losses, steps = [], []
    buf = []
    t0 = time.perf_counter()
    sub.reset()
    for t in range(len(stream) - 1):
        phi = sub.step(int(stream[t]))
        scores = model.predict(phi)                 # (vocab,)
        p = np.exp(scores - scores.max())
        p /= p.sum() + 1e-12
        nxt = int(stream[t + 1])
        buf.append(-np.log(max(p[nxt], 1e-12)) / ln2)
        model.update(phi, eye[nxt])
        if (t + 1) % eval_every == 0:
            losses.append(float(np.mean(buf)))
            steps.append(t + 1)
            buf = []
            if verbose:
                print(f"  {arm} t={t+1} bpc={losses[-1]:.3f}",
                      flush=True)
    dt = time.perf_counter() - t0
    n = len(stream) - 1
    tail = losses[max(0, len(losses) - 5):]
    W = rls.get_weights()
    state_bytes = (sub.state_bytes + W.nbytes
                   + rls.P_blocks.nbytes + rls.P_bias.nbytes)
    n_cons = (model.n_consolidations if arm == "apia_ctxmem" else 0)
    if arm == "apia_ctxmem":
        state_bytes += model.W_slow.nbytes
    return dict(
        arm=arm, seed=seed, n_chars=n,
        final_bpc=float(np.mean(tail)),
        bpc_curve=losses, curve_steps=steps,
        chars_per_s=float(n / dt), wall_s=float(dt),
        param_floats=int(sub.param_floats + W.size),
        learn_state_bytes=int(state_bytes),
        peak_vram_mb=0,  # CPU-resident by construction
        n_consolidations=n_cons)


# ---------- torch baselines ----------

def run_torch(stream, vocab, kind, seed, chars_ctx=128, batch=32,
              lr=1e-3, eval_every=2000, verbose=True):
    """Online single-pass LM: loss recorded before optimizer.step."""
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    if kind == "gru":
        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Embedding(vocab, 64)
                s.rnn = nn.GRU(64, 256, 1, batch_first=True)
                s.head = nn.Linear(256, vocab)
            def forward(s, x):
                h, _ = s.rnn(s.emb(x))
                return s.head(h)
    else:
        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Embedding(vocab, 128)
                s.pos = nn.Embedding(chars_ctx, 128)
                layer = nn.TransformerEncoderLayer(
                    128, 4, 512, dropout=0.0, batch_first=True,
                    norm_first=True)
                s.tr = nn.TransformerEncoder(layer, 2)
                s.head = nn.Linear(128, vocab)
                s.register_buffer(
                    "mask", torch.triu(torch.ones(
                        chars_ctx, chars_ctx, dtype=torch.bool), 1))
            def forward(s, x):
                e = s.emb(x) + s.pos.weight[:x.size(1)]
                return s.head(s.tr(e, mask=s.mask[:x.size(1),
                                                    :x.size(1)],
                                   is_causal=True))
    mdl = M().to(dev)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n_par = sum(p.numel() for p in mdl.parameters())

    x = torch.from_numpy(stream)
    T = chars_ctx
    losses, steps = [], []
    buf_x, buf_y = [], []
    i = 0
    t0 = time.perf_counter()
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    while i + T + 1 <= len(stream):
        xs = x[i:i + T]
        buf_x.append(xs)
        buf_y.append(x[i + 1:i + T + 1])
        i += T
        if len(buf_x) < batch and i + T + 1 <= len(stream):
            continue
        X = torch.stack(buf_x).to(dev)
        Y = torch.stack(buf_y).to(dev)
        logits = mdl(X)
        loss = lossf(logits.reshape(-1, vocab), Y.reshape(-1))
        lv = float(loss.item())                     # pre-update
        opt.zero_grad()
        loss.backward()
        opt.step()
        buf_x, buf_y = [], []
        losses.append(lv / np.log(2.0))
        steps.append(i)
        if verbose and len(losses) % 20 == 0:
            print(f"  {kind} tok={i} bpc={losses[-1]:.3f}", flush=True)
    dt = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() // 2**20
            if dev == "cuda" else 0)
    tail = losses[max(0, len(losses) - 20):]
    return dict(
        arm=kind, seed=seed, n_chars=i, ctx=chars_ctx, batch=batch,
        final_bpc=float(np.mean(tail)),
        bpc_curve=[float(v) for v in losses], curve_steps=steps,
        chars_per_s=float(i / dt), wall_s=float(dt),
        param_floats=int(n_par),
        learn_state_bytes=int(n_par * 4 * 3),  # W + grad + 2x Adam
        peak_vram_mb=int(peak), device=dev)


# ---------- driver ----------

def existing(results, arm, seed):
    return any(r["arm"] == arm and r["seed"] == seed
               for r in results.get("runs", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all",
                    choices=["all", "apia_block", "apia_ctxmem",
                             "gru", "tf"])
    ap.add_argument("--chars", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="native_lm_results.json")
    a = ap.parse_args()

    stream, vocab = load_chars()
    if a.smoke:
        a.chars = min(a.chars, 6_000)
    stream = stream[:a.chars + 1]
    print(f"stream: {len(stream)} chars, vocab={vocab}")

    results = {}
    if os.path.exists(a.out):
        results = json.load(open(a.out))
    results.setdefault("runs", [])
    results["config"] = dict(chars=a.chars, seed=a.seed,
                             ctx=a.ctx, batch=a.batch, lr=a.lr)

    arms = (["apia_block", "apia_ctxmem", "gru", "tf"]
            if a.arm == "all" else [a.arm])
    for arm in arms:
        if existing(results, arm, a.seed):
            print(f"skip {arm} s{a.seed} (exists)")
            continue
        print(f"== {arm} s{a.seed}")
        if arm.startswith("apia"):
            r = run_apia(stream, vocab, arm, a.seed)
        else:
            r = run_torch(stream, vocab, arm, a.seed,
                          chars_ctx=a.ctx, batch=a.batch, lr=a.lr)
        results["runs"].append(r)
        json.dump(results, open(a.out, "w"), indent=1)
        print(f"   -> bpc={r['final_bpc']:.3f} "
              f"{r['chars_per_s']:.0f} tok/s "
              f"state={r['learn_state_bytes']/1e6:.2f}MB "
              f"vram={r['peak_vram_mb']}MB")

    print("done ->", a.out)


if __name__ == "__main__":
    main()
