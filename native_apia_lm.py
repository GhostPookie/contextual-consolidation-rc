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


# ---------- numba fast path (fused per-char loop) ----------
# Same math, same predict-before-update order; removes interpreter
# overhead and exploits phi's local-support sparsity (degree+1 nnz
# per reservoir dim). Falls back to the numpy loop if numba absent.

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

if _HAVE_NUMBA:

    @njit(cache=True)
    def _basis_nz(x, knots, degree, K, Nbuf, ok, ov):
        """Scalar Cox-de Boor -> nonzero (basis-idx, val) pairs."""
        t = knots
        nt = t.shape[0]
        xc = x
        if xc < t[0]:
            xc = t[0]
        if xc > t[nt - 1]:
            xc = t[nt - 1]
        idx = 0
        for i in range(nt):
            if t[i] > xc:
                break
            idx = i
        if idx > nt - 2:
            idx = nt - 2
        for i in range(nt - 1):
            Nbuf[i] = 0.0
        Nbuf[idx] = 1.0
        for d in range(1, degree + 1):
            nb = nt - d - 1
            for i in range(nb):
                left = 0.0
                dl = t[i + d] - t[i]
                if dl > 0.0:
                    left = (xc - t[i]) / dl * Nbuf[i]
                right = 0.0
                dr = t[i + d + 1] - t[i + 1]
                if dr > 0.0:
                    right = (t[i + d + 1] - xc) / dr * Nbuf[i + 1]
                Nbuf[i] = left + right
        if xc >= t[nt - 1]:
            for i in range(nt - 1):
                Nbuf[i] = 0.0
            Nbuf[K - 1] = 1.0
        cnt = 0
        for k in range(K):
            v = Nbuf[k]
            if v != 0.0:
                ok[cnt] = k
                ov[cnt] = v
                cnt += 1
        return cnt

    @njit(cache=True)
    def _res_tanh(h, tok, W_in, Wd, Wi, Wp, alpha, tmp):
        """h <- (1-a)h + a*tanh(W_in[:,tok] + W_res@h); tmp=tanh(h)."""
        n = h.shape[0]
        for i in range(n):
            acc = W_in[i, tok]
            for p in range(Wp[i], Wp[i + 1]):
                acc += Wd[p] * h[Wi[p]]
            tmp[i] = acc
        for i in range(n):
            h[i] = (1.0 - alpha) * h[i] + alpha * np.tanh(tmp[i])
            tmp[i] = np.tanh(h[i])

    @njit(cache=True)
    def _nb_run_block(tokens, W_in, Wd, Wi, Wp, alpha, knots, degree, K,
                      W, Pb, Pbias, lam, trace_bound, bias_bound,
                      eval_every, vocab):
        """apia_block, fully fused. tokens[t]->predict->update(tokens[t+1])."""
        D = W_in.shape[0]
        F = D * K + 1
        C = vocab
        n = tokens.shape[0] - 1
        h = np.zeros(D)
        tmp = np.zeros(D)
        Nbuf = np.zeros(knots.shape[0] - 1)
        ok = np.empty(degree + 3, np.int64)
        ov = np.empty(degree + 3)
        jk = np.empty((D, degree + 2), np.int64)
        jv = np.empty((D, degree + 2))
        jc = np.empty(D, np.int64)
        Q = np.empty((D, K))
        scores = np.empty(C)
        e = np.empty(C)
        n_eval = n // eval_every
        curve = np.empty(n_eval)
        steps = np.empty(n_eval, np.int64)
        nll, cnt, nci = 0.0, 0, 0
        ln2 = np.log(2.0)
        for t in range(n):
            _res_tanh(h, tokens[t], W_in, Wd, Wi, Wp, alpha, tmp)
            for m in range(C):
                scores[m] = W[F - 1, m]
            for j in range(D):
                c = _basis_nz(tmp[j], knots, degree, K, Nbuf, ok, ov)
                jc[j] = c
                for z in range(c):
                    jk[j, z] = ok[z]
                    jv[j, z] = ov[z]
                    r = j * K + ok[z]
                    v = ov[z]
                    for m in range(C):
                        scores[m] += v * W[r, m]
            mx = scores[0]
            for m in range(1, C):
                if scores[m] > mx:
                    mx = scores[m]
            se = 0.0
            for m in range(C):
                se += np.exp(scores[m] - mx)
            tgt = tokens[t + 1]
            nll += -np.log(np.exp(scores[tgt] - mx) / se) / ln2
            cnt += 1
            for m in range(C):
                e[m] = -scores[m]
            e[tgt] += 1.0
            s = lam + Pbias[0]
            for j in range(D):
                for i in range(K):
                    acc = 0.0
                    for z in range(jc[j]):
                        acc += Pb[j, i, jk[j, z]] * jv[j, z]
                    Q[j, i] = acc
                for z in range(jc[j]):
                    s += jv[j, z] * Q[j, jk[j, z]]
            for j in range(D):
                for i in range(K):
                    g = Q[j, i] / s
                    for m in range(C):
                        W[j * K + i, m] += g * e[m]
                    for k in range(K):
                        Pb[j, i, k] = (Pb[j, i, k] - g * Q[j, k]) / lam
            gb = Pbias[0] / s
            for m in range(C):
                W[F - 1, m] += gb * e[m]
            Pbias[0] = (Pbias[0] - gb * Pbias[0]) / lam
            if trace_bound > 0.0:
                for j in range(D):
                    tr = 0.0
                    for i in range(K):
                        tr += Pb[j, i, i]
                    if tr > trace_bound:
                        sc = trace_bound / tr
                        for i in range(K):
                            for k in range(K):
                                Pb[j, i, k] *= sc
                if bias_bound > 0.0 and Pbias[0] > bias_bound:
                    Pbias[0] = bias_bound
            if (t + 1) % eval_every == 0:
                curve[nci] = nll / cnt
                steps[nci] = t + 1
                nci += 1
                nll, cnt = 0.0, 0
        return curve[:nci], steps[:nci]

    @njit(cache=True)
    def _nb_run_ctxmem(tokens, W_in, Wd, Wi, Wp, alpha, knots, degree, K,
                       W, Pb, Pbias, lam, trace_bound, bias_bound,
                       W_slow, E, beta, gamma, eta, a_ema, a_fast,
                       cons_every, e_gate, tau, p_inflate,
                       cons_counts, eval_every, vocab):
        """apia_ctxmem fused: 3-slot gated memory, same update order."""
        D = W_in.shape[0]
        F = D * K + 1
        C = vocab
        S = W_slow.shape[0]
        n = tokens.shape[0] - 1
        h = np.zeros(D)
        tmp = np.zeros(D)
        Nbuf = np.zeros(knots.shape[0] - 1)
        ok = np.empty(degree + 3, np.int64)
        ov = np.empty(degree + 3)
        jk = np.empty((D, degree + 2), np.int64)
        jv = np.empty((D, degree + 2))
        jc = np.empty(D, np.int64)
        Q = np.empty((D, K))
        scores = np.empty(C)
        ss = np.empty((S, C))
        w = np.empty(S)
        e = np.empty(C)
        n_eval = n // eval_every
        curve = np.empty(n_eval)
        steps = np.empty(n_eval, np.int64)
        nll, cnt, nci = 0.0, 0, 0
        e_fast, m2 = 0.0, 1.0
        a_m2 = 0.02
        step_c, n_cons = 0, 0
        ln2 = np.log(2.0)
        for t in range(n):
            _res_tanh(h, tokens[t], W_in, Wd, Wi, Wp, alpha, tmp)
            for m in range(C):
                scores[m] = W[F - 1, m]
            for s_ in range(S):
                for m in range(C):
                    ss[s_, m] = 0.0
            for j in range(D):
                c = _basis_nz(tmp[j], knots, degree, K, Nbuf, ok, ov)
                jc[j] = c
                for z in range(c):
                    jk[j, z] = ok[z]
                    jv[j, z] = ov[z]
                    r = j * K + ok[z]
                    v = ov[z]
                    for m in range(C):
                        scores[m] += v * W[r, m]
                    for s_ in range(S):
                        for m in range(C):
                            ss[s_, m] += v * W_slow[s_, r, m]
            tgt = tokens[t + 1]
            # slot errors (slow-only): ||y - beta*ss_s||^2
            for s_ in range(S):
                acc = 0.0
                for m in range(C):
                    acc += (beta * ss[s_, m]) ** 2
                acc += (1.0 - beta * ss[s_, tgt]) ** 2 - (
                    beta * ss[s_, tgt]) ** 2
                E[s_] = (1.0 - a_ema) * E[s_] + a_ema * acc
            # gate
            emin = E[0]
            for s_ in range(1, S):
                if E[s_] < emin:
                    emin = E[s_]
            if emin < 1e-8:
                emin = 1e-8
            zmax = -1e30
            for s_ in range(S):
                z = -eta * E[s_] / emin
                w[s_] = z
                if z > zmax:
                    zmax = z
            wsum = 0.0
            for s_ in range(S):
                w[s_] = np.exp(w[s_] - zmax)
                wsum += w[s_]
            # full scores = fast + beta*mix(ss)
            for m in range(C):
                acc = 0.0
                for s_ in range(S):
                    acc += w[s_] * ss[s_, m]
                scores[m] += beta * acc / wsum
            mx = scores[0]
            for m in range(1, C):
                if scores[m] > mx:
                    mx = scores[m]
            se = 0.0
            for m in range(C):
                se += np.exp(scores[m] - mx)
            nll += -np.log(np.exp(scores[tgt] - mx) / se) / ln2
            cnt += 1
            # residual update: e = y - scores (gated residual == total)
            for m in range(C):
                e[m] = -scores[m]
            e[tgt] += 1.0
            s = lam + Pbias[0]
            for j in range(D):
                for i in range(K):
                    acc = 0.0
                    for z in range(jc[j]):
                        acc += Pb[j, i, jk[j, z]] * jv[j, z]
                    Q[j, i] = acc
                for z in range(jc[j]):
                    s += jv[j, z] * Q[j, jk[j, z]]
            for j in range(D):
                for i in range(K):
                    g = Q[j, i] / s
                    for m in range(C):
                        W[j * K + i, m] += g * e[m]
                    for k in range(K):
                        Pb[j, i, k] = (Pb[j, i, k] - g * Q[j, k]) / lam
            gb = Pbias[0] / s
            for m in range(C):
                W[F - 1, m] += gb * e[m]
            Pbias[0] = (Pbias[0] - gb * Pbias[0]) / lam
            if trace_bound > 0.0:
                for j in range(D):
                    tr = 0.0
                    for i in range(K):
                        tr += Pb[j, i, i]
                    if tr > trace_bound:
                        sc = trace_bound / tr
                        for i in range(K):
                            for k in range(K):
                                Pb[j, i, k] *= sc
                if bias_bound > 0.0 and Pbias[0] > bias_bound:
                    Pbias[0] = bias_bound
            en = 0.0
            for m in range(C):
                en += e[m] * e[m]
            e_fast = (1.0 - a_fast) * e_fast + a_fast * en
            m2 = (1.0 - a_m2) * m2 + a_m2 * 1.0
            step_c += 1
            if step_c % cons_every == 0 and e_fast <= e_gate * m2:
                k = 0
                for s_ in range(1, S):
                    if E[s_] < E[k]:
                        k = s_
                target = -1
                if E[k] < tau * m2:
                    target = k
                else:
                    cmin = cons_counts[0]
                    for s_ in range(1, S):
                        if cons_counts[s_] < cmin:
                            cmin = cons_counts[s_]
                    if cmin > 0:
                        target = -1          # 'refuse' at saturation
                    else:
                        target = 0
                        for s_ in range(1, S):
                            if cons_counts[s_] < cons_counts[target]:
                                target = s_
                if target >= 0:
                    cf = (1.0 - gamma) / beta
                    for r in range(F):
                        for m in range(C):
                            W_slow[target, r, m] += cf * W[r, m]
                            W[r, m] *= gamma
                    for j in range(D):
                        for i in range(K):
                            for kk in range(K):
                                Pb[j, i, kk] *= p_inflate
                    Pbias[0] *= p_inflate
                    cons_counts[target] += 1
                    n_cons += 1
            if (t + 1) % eval_every == 0:
                curve[nci] = nll / cnt
                steps[nci] = t + 1
                nci += 1
                nll, cnt = 0.0, 0
        return curve[:nci], steps[:nci], n_cons


def _apia_init(vocab, n_res, n_basis, seed):
    sub = ReservoirSubstrate(vocab, n_res=n_res, n_basis=n_basis,
                             seed=seed)
    rls = BlockDiagRLS(input_dim=n_res, n_basis=n_basis,
                       n_outputs=vocab, forgetting=0.9995,
                       init_var=1e2, windup_mult=1e4)
    csr = sub.res.W_res.tocsr()
    args = dict(
        W_in=np.ascontiguousarray(sub.res.W_in),
        Wd=csr.data.astype(np.float64),
        Wi=csr.indices.astype(np.int64),
        Wp=csr.indptr.astype(np.int64),
        alpha=sub.res.alpha,
        knots=sub.spline.knots,
        degree=sub.spline.degree,
        K=n_basis,
        W=rls.W,
        Pb=rls.P_blocks,
        Pbias=rls.P_bias.reshape(-1),
        lam=rls.lam,
        trace_bound=rls.trace_bound or 0.0,
        bias_bound=rls._bias_bound or 0.0)
    return sub, rls, args


def run_apia_fast(stream, vocab, arm, seed, n_res=512, n_basis=10,
                  eval_every=2000, verbose=True):
    """Numba-fused run; identical update order to run_apia."""
    if not _HAVE_NUMBA:
        raise RuntimeError("numba unavailable")
    sub, rls, a = _apia_init(vocab, n_res, n_basis, seed)
    tok = np.ascontiguousarray(stream.astype(np.int64))
    t0 = time.perf_counter()
    if arm == "apia_block":
        curve, steps = _nb_run_block(
            tok, a["W_in"], a["Wd"], a["Wi"], a["Wp"], a["alpha"],
            a["knots"], a["degree"], a["K"], a["W"], a["Pb"],
            a["Pbias"], a["lam"], a["trace_bound"], a["bias_bound"],
            eval_every, vocab)
        n_cons = 0
    elif arm == "apia_ctxmem":
        S = 3
        W_slow = np.zeros((S,) + a["W"].shape)
        E = np.zeros(S)
        curve, steps, n_cons = _nb_run_ctxmem(
            tok, a["W_in"], a["Wd"], a["Wi"], a["Wp"], a["alpha"],
            a["knots"], a["degree"], a["K"], a["W"], a["Pb"],
            a["Pbias"], a["lam"], a["trace_bound"], a["bias_bound"],
            W_slow, E, 1.0, 0.1, 5.0, 0.05, 0.1, 250, 1.0, 0.5, 1.5,
            np.zeros(S, np.int64), eval_every, vocab)
        a["W_slow"] = W_slow
    else:
        raise ValueError(arm)
    dt = time.perf_counter() - t0
    n = len(stream) - 1
    tail = list(curve)[max(0, len(curve) - 5):]
    W = rls.get_weights()
    state_bytes = (sub.state_bytes + W.nbytes
                   + rls.P_blocks.nbytes + rls.P_bias.nbytes
                   + a.get("W_slow", np.empty(0)).nbytes)
    if verbose:
        print(f"  {arm} done {dt:.0f}s", flush=True)
    return dict(
        arm=arm, seed=seed, n_chars=n,
        final_bpc=float(np.mean(tail)),
        bpc_curve=[float(v) for v in curve],
        curve_steps=[int(v) for v in steps],
        chars_per_s=float(n / dt), wall_s=float(dt),
        param_floats=int(sub.param_floats + W.size),
        learn_state_bytes=int(state_bytes),
        peak_vram_mb=0, n_consolidations=int(n_cons))


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


def run_torch_online(stream, vocab, kind, seed, chars_ctx=128, lr=1e-3,
                     eval_every=2000, verbose=True):
    """B=1 online regime — same cadence as the APIA arms: every step
    sees ONE sample, loss logged BEFORE optimizer.step, single pass.
    Backprop stays (it IS torch's learning rule); what we equalize is
    cadence (batch=1, per-step update), not the rule.
    gru_b1: stateful GRU, hidden carried + detached (TBPTT-1).
    tf_b1 : sliding ctx-window, forward+backward per step."""
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    if kind == "gru_b1":
        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Embedding(vocab, 64)
                s.rnn = nn.GRU(64, 256, 1, batch_first=True)
                s.head = nn.Linear(256, vocab)
            def forward(s, x, h):
                h, _ = s.rnn(s.emb(x), h)
                return s.head(h[:, -1]), h.detach()
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
                o = s.tr(e, mask=s.mask[:x.size(1), :x.size(1)],
                         is_causal=True)
                return s.head(o[:, -1])
    mdl = M().to(dev)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n_par = sum(p.numel() for p in mdl.parameters())
    x = torch.from_numpy(stream).to(dev)

    T = len(stream) - 1
    losses, steps = [], []
    nll, cnt = 0.0, 0
    hid = None
    ln2 = float(np.log(2.0))
    t0 = time.perf_counter()
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for t in range(T):
        if kind == "gru_b1":
            xin = x[t:t + 1].view(1, 1)
            logits, hid = mdl(xin, hid)
        else:
            lo = max(0, t + 1 - chars_ctx)
            xin = x[lo:t + 1].view(1, -1)
            logits = mdl(xin)
        y = x[t + 1:t + 2]
        loss = lossf(logits, y)
        nll += float(loss.detach()) / ln2
        cnt += 1
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (t + 1) % eval_every == 0:
            losses.append(nll / cnt)
            steps.append(t + 1)
            nll, cnt = 0.0, 0
            if verbose and len(losses) % 5 == 0:
                print(f"  {kind} t={t+1} bpc={losses[-1]:.3f}",
                      flush=True)
    dt = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() // 2**20
            if dev == "cuda" else 0)
    tail = losses[max(0, len(losses) - 5):]
    return dict(
        arm=kind, seed=seed, n_chars=T, ctx=chars_ctx, batch=1,
        final_bpc=float(np.mean(tail)),
        bpc_curve=[float(v) for v in losses], curve_steps=steps,
        chars_per_s=float(T / dt), wall_s=float(dt),
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
                             "gru", "tf", "gru_b1", "tf_b1"])
    ap.add_argument("--chars", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fast", type=int, default=1,
                    help="use numba fused loop for apia arms (default on)")
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
            fn = (run_apia_fast if (a.fast and _HAVE_NUMBA)
                  else run_apia)
            r = fn(stream, vocab, arm, a.seed)
        elif arm.endswith("_b1"):
            r = run_torch_online(stream, vocab, arm, a.seed,
                                 chars_ctx=a.ctx, lr=a.lr)
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
