"""Kaggle: multi-seed OPT-1.3B sweep (Phase-2 robustness).

Decides whether the single-seed ctxmem advantage (0.22->0.15 onset) is
real or a lucky draw — same question multi-seed already answered (as
'seed-dependent') for the GPT-2 family.

Self-contained: inlines SplineFeatureMap / BlockDiagRLS /
ContextualMemory identical to kaggle_multiseed.py (validated: GPT-2
multi-seed results matched local engine bit-for-bit).

Arms per seed: apia_block | apia_ctxmem | lora r=8 @ lr=1e-4
(lr=1e-3 already measured diverged, n=1 — not re-run).

Kaggle notes:
  * Cell 0:  pip uninstall -y torchao     (peft compat)
  * Internet ON for HF downloads; GPU T4; OPT loaded fp16.
Output: /kaggle/working/multiseed_opt13b.json  (incremental)
"""

import json
import os
import time

import numpy as np
import torch
from transformers import (AutoModel, AutoTokenizer,
                          AutoModelForSequenceClassification)
from peft import LoraConfig, get_peft_model
from sklearn.datasets import fetch_20newsgroups

# ---------------- protocol constants ----------------
LAM = 0.98
INIT_VAR = 1e2
WINDUP_MULT = 1e4

PA, PB, PA_RET = 600, 600, 400
ONSET = 100
MAX_TOK = 128
K = 10

MODEL_ID = "facebook/opt-1.3b"
TAG = "opt-1.3b"
LORA_LRS = [1e-4]          # 1e-3 already measured diverged (n=1)
SEEDS = [0, 1, 2]
OUT = "/kaggle/working/multiseed_opt13b.json"


# ================= inlined: rls_engine.py =================
class SplineFeatureMap:
    def __init__(self, input_dim, n_basis=10, degree=3, lo=-1.0, hi=1.0):
        if n_basis < degree + 1:
            raise ValueError("n_basis must be >= degree+1")
        self.input_dim = input_dim
        self.n_basis = n_basis
        self.degree = degree
        self.lo, self.hi = lo, hi
        n_interior = n_basis - degree - 1
        interior = (np.linspace(lo, hi, n_interior + 2)[1:-1]
                    if n_interior > 0 else np.empty(0))
        self.knots = np.concatenate([
            np.full(degree + 1, lo), interior, np.full(degree + 1, hi)])
        assert len(self.knots) == n_basis + degree + 1
        self.n_features = input_dim * n_basis + 1

    def _basis_matrix(self, x):
        t = self.knots
        x = np.clip(np.asarray(x, dtype=float), t[0], t[-1])
        n = x.size
        B = np.zeros((n, len(t) - 1))
        idx = np.clip(np.searchsorted(t, x, side='right') - 1,
                      0, len(t) - 2)
        B[np.arange(n), idx] = 1.0
        for d in range(1, self.degree + 1):
            nb = len(t) - d - 1
            dl = t[d:d + nb] - t[:nb]
            left = np.divide(
                x[:, None] - t[:nb][None, :],
                np.broadcast_to(dl[None, :], (n, nb)),
                out=np.zeros((n, nb)), where=(dl[None, :] > 0)
            ) * B[:, :nb]
            dr = t[d + 1:d + 1 + nb] - t[1:nb + 1]
            right = np.divide(
                t[d + 1:d + 1 + nb][None, :] - x[:, None],
                np.broadcast_to(dr[None, :], (n, nb)),
                out=np.zeros((n, nb)), where=(dr[None, :] > 0)
            ) * B[:, 1:nb + 1]
            B = left + right
        edge = x >= t[-1]
        if np.any(edge):
            B[edge] = 0.0
            B[edge, -1] = 1.0
        return B

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.input_dim:
            raise ValueError("dim mismatch")
        phi = self._basis_matrix(x).reshape(-1)
        return np.append(phi, 1.0)


class BlockDiagRLS:
    def __init__(self, input_dim, n_basis, n_outputs,
                 forgetting=1.0, init_var=1e2, windup_mult=None):
        self.D, self.K = input_dim, n_basis
        self.n_features = input_dim * n_basis + 1
        self.W = np.zeros((self.n_features, n_outputs))
        self.P_blocks = np.tile(
            np.eye(n_basis) * init_var, (input_dim, 1, 1))
        self.P_bias = np.array([[init_var]])
        self.lam = forgetting
        self.trace_bound = (init_var * n_basis * windup_mult
                            if windup_mult is not None else None)
        self._bias_bound = (init_var * windup_mult
                            if windup_mult is not None else None)

    def update(self, phi, y):
        phi_b = phi[:self.D * self.K].reshape(self.D, self.K)
        pb = phi[-1]
        Q = np.einsum('dij,dj->di', self.P_blocks, phi_b)
        s = self.lam + np.sum(phi_b * Q) + self.P_bias[0, 0] * pb * pb
        e = y - phi @ self.W
        G = Q / s
        self.W[:self.D * self.K] += (G.reshape(-1, 1)) * e[None, :]
        self.P_blocks -= np.einsum('di,dj->dij', G, Q)
        self.P_blocks /= self.lam
        if self.trace_bound is not None:
            tr = np.einsum('dii->d', self.P_blocks)
            over = tr > self.trace_bound
            if np.any(over):
                self.P_blocks[over] *= (self.trace_bound
                                        / tr[over])[:, None, None]
        g_b = self.P_bias[0, 0] * pb / s
        self.W[-1] += g_b * e
        self.P_bias = (self.P_bias
                       - np.array([[g_b * self.P_bias[0, 0] * pb]])) / self.lam
        if (self._bias_bound is not None
                and self.P_bias[0, 0] > self._bias_bound):
            self.P_bias[0, 0] = self._bias_bound
        return e

    def predict(self, phi):
        return phi @ self.W

    def get_weights(self):
        return self.W.copy()

    def set_weights(self, W):
        self.W = np.asarray(W, dtype=float).copy()

    def inflate_P(self, factor):
        self.P_blocks *= factor
        self.P_bias *= factor


# ================= inlined: contextual_memory.py =================
class ContextualMemory:
    def __init__(self, model, n_slots=3, beta=1.0, gamma=0.1, eta=2.0,
                 alpha_ema=0.01, alpha_fast=0.1, consolidate_every=250,
                 e_gate=0.5, signal_var=1.0, tau=0.5, p_inflate=1.5,
                 eps=1e-8, alpha_m2=0.02, err_mode='slow',
                 write_protect=True, empty_rule='count',
                 saturation='overwrite', w_fast_cap=None):
        self.model = model
        self.n_slots = n_slots
        self.beta, self.gamma = beta, gamma
        self.eta = eta
        self.alpha_ema, self.alpha_fast = alpha_ema, alpha_fast
        self.consolidate_every = consolidate_every
        self.e_gate = e_gate
        self.alpha_m2 = alpha_m2
        self._m2 = float(signal_var)
        self.tau = tau
        self.p_inflate = p_inflate
        self.eps = eps
        self.err_mode = err_mode
        self.write_protect = write_protect
        self.empty_rule = empty_rule
        self.saturation = saturation
        self.w_fast_cap = w_fast_cap

        W0 = model.get_weights()
        self.W_slow = np.zeros((n_slots,) + W0.shape)
        self.E = np.zeros(n_slots)
        self._e_fast = 0.0
        self._steps = 0
        self.n_consolidations = 0
        self.max_discontinuity = 0.0
        self.slot_cons_counts = np.zeros(n_slots, dtype=int)

    def _slot_errors(self, phi, y):
        slow = np.einsum('f,sfm->sm', phi, self.W_slow)
        if self.err_mode == 'total':
            slow = slow + (phi @ self.model.get_weights())[None, :] / self.beta
        return np.sum((y[None, :] - self.beta * slow) ** 2, axis=1)

    def _gate(self):
        e_min = max(self.E.min(), self.eps)
        z = -self.eta * self.E / e_min
        z -= z.max()
        w = np.exp(z)
        return w / w.sum()

    def _slow_mix(self, phi, w):
        W_mix = np.einsum('s,sfm->fm', w, self.W_slow)
        return self.beta * (phi @ W_mix)

    def predict(self, phi):
        return self.model.predict(phi) + self._slow_mix(phi, self._gate())

    def update(self, phi, y):
        self.E = ((1 - self.alpha_ema) * self.E
                  + self.alpha_ema * self._slot_errors(phi, y))
        w = self._gate()
        residual = y - self._slow_mix(phi, w)
        e_total = self.model.update(phi, residual)
        e_now = float(np.sum(e_total ** 2))
        self._e_fast = ((1 - self.alpha_fast) * self._e_fast
                        + self.alpha_fast * e_now)
        self._m2 = ((1 - self.alpha_m2) * self._m2
                    + self.alpha_m2 * float(np.sum(y * y)))
        if self.w_fast_cap is not None:
            wf = self.model.get_weights()
            n = float(np.linalg.norm(wf))
            if n > self.w_fast_cap:
                self.model.set_weights(wf * (self.w_fast_cap / n))
        self._steps += 1
        if (self._steps % self.consolidate_every == 0
                and self._e_fast <= self.e_gate * self._m2):
            k = int(np.argmin(self.E))
            if not self.write_protect or self.E[k] < self.tau * self._m2:
                target = k
            else:
                if (self.slot_cons_counts.min() > 0
                        and self.saturation == 'refuse'):
                    target = -1
                else:
                    target = int(np.argmin(self.slot_cons_counts))
            if target >= 0:
                self.consolidate(phi, target)
        return e_total

    def consolidate(self, phi_probe, k):
        W_fast = self.model.get_weights()
        before = self.predict(phi_probe)
        self.W_slow[k] += (1.0 - self.gamma) / self.beta * W_fast
        self.model.set_weights(self.gamma * W_fast)
        if self.p_inflate != 1.0:
            self.model.inflate_P(self.p_inflate)
        d = float(np.max(np.abs(before - self.predict(phi_probe))))
        self.max_discontinuity = max(self.max_discontinuity, d)
        self.n_consolidations += 1
        self.slot_cons_counts[k] += 1


# ================= protocol =================
def load_domains(seed=7):
    domA = fetch_20newsgroups(subset="all",
                              categories=["sci.space", "rec.sport.baseball"],
                              remove=("headers", "footers", "quotes"),
                              shuffle=True, random_state=0)
    domB = fetch_20newsgroups(subset="all",
                              categories=["comp.graphics",
                                          "talk.politics.misc"],
                              remove=("headers", "footers", "quotes"),
                              shuffle=True, random_state=1)
    A = list(zip(domA.data, domA.target))
    B = list(zip(domB.data, domB.target))
    rng = np.random.default_rng(seed)
    rng.shuffle(A)
    rng.shuffle(B)
    return A[:PA], B[:PB], A[PA:PA + PA_RET]


@torch.no_grad()
def embed(texts, tok, mdl, dev, batch=32):
    H = np.empty((len(texts), mdl.config.hidden_size), dtype=np.float32)
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_TOK).to(dev)
        out = mdl(**enc).last_hidden_state
        idx = enc["attention_mask"].sum(1) - 1
        H[i:i + batch] = out[torch.arange(len(idx)), idx].cpu().numpy()
    return H


def get_features(name, texts, tok, mdl, dev):
    cf = f"/kaggle/working/clsfeat_{name}.npz"
    if os.path.exists(cf):
        return np.load(cf)["X"]
    H = embed(texts, tok, mdl, dev)
    np.savez_compressed(cf, X=H)
    return H


def run_stream(Phi, Y, kind, e_gate=0.5, eta=2.0, alpha_ema=0.01):
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


def run_lora(texts, Y, tok, dev, lr, seed):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, dtype=torch.float16)
    base.config.pad_token_id = tok.pad_token_id
    cfg = LoraConfig(r=8, lora_alpha=16,
                     target_modules=["q_proj", "v_proj"],
                     lora_dropout=0.0, bias="none")
    mdl = get_peft_model(base, cfg).to(dev)
    opt = torch.optim.AdamW(
        [p for p in mdl.parameters() if p.requires_grad], lr=lr)
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=MAX_TOK)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    preds = np.empty(len(texts))
    t0 = time.perf_counter()
    mdl.train()
    for i in range(len(texts)):
        inp = {"input_ids": ids[i:i + 1].to(dev),
               "attention_mask": mask[i:i + 1].to(dev)}
        out = mdl(**inp, labels=torch.tensor(
            [int(Y[i, 0])], device=dev))
        preds[i] = float(out.logits.argmax(-1).item())
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (i + 1) % 400 == 0:
            print(f"    lora lr={lr:g}: {i+1}/{len(texts)}", flush=True)
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e6
    del mdl, base, opt
    torch.cuda.empty_cache()
    return preds, dt, peak


def phase_err(preds, y, lo, hi):
    return float(np.mean(preds[lo:hi] != y[lo:hi]))


def save(results):
    json.dump(results, open(OUT, "w"), indent=2)


def summarize(vals):
    return {"seeds": vals, "mean": float(np.mean(vals)),
            "std": float(np.std(vals))}


def existing(results, arm, seed):
    ex = results.get(TAG, {}).get(arm, [])
    if isinstance(ex, dict):
        ex = ex["per_seed"]
    return len(ex) > SEEDS.index(seed)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    results.setdefault(TAG, {})

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token

    # ---- APIA arms: frozen OPT fp16 features -> spline -> RLS ----
    mdl_f = AutoModel.from_pretrained(
        MODEL_ID, dtype=torch.float16).to(dev).eval()
    for p in mdl_f.parameters():
        p.requires_grad_(False)
    for seed in SEEDS:
        if (existing(results, "apia_block", seed)
                and existing(results, "apia_ctxmem", seed)):
            continue
        A1, B, A2 = load_domains(seed=seed)
        Y = np.array([l for _, l in A1 + B + A2],
                     dtype=float).reshape(-1, 1)
        bounds = [0, PA, PA + PB, PA + PB + PA_RET]
        H = np.concatenate([
            get_features(f"opt_s{seed}_{n}",
                         [t for t, _ in s], tok, mdl_f, dev)
            for n, s in (("A1", A1), ("B", B), ("A2", A2))])
        H = (H - H.mean(0)) / (H.std(0) + 1e-6)
        X = np.clip(H, -3.0, 3.0) / 3.0
        fmap = SplineFeatureMap(X.shape[1], K)
        Phi = np.stack([fmap.transform(x) for x in X])
        for arm, (kind, kws) in (
                ("apia_block", ("block", {})),
                ("apia_ctxmem",
                 ("ctxmem", dict(e_gate=1.0, eta=5.0,
                                 alpha_ema=0.05)))):
            if existing(results, arm, seed):
                continue
            preds, model = run_stream(Phi, Y, kind, **kws)
            e = lambda lo, hi: phase_err(preds, Y[:, 0], lo, hi)
            rec = {"p1_end": e(bounds[1] - ONSET, bounds[1]),
                   "p2_end": e(bounds[2] - ONSET, bounds[2]),
                   "p3_onset": e(bounds[2], bounds[2] + ONSET),
                   "p3_end": e(bounds[3] - ONSET, bounds[3]),
                   "cons": getattr(model, "n_consolidations", 0)}
            results[TAG].setdefault(arm, []).append(rec)
            save(results)
            print(f"opt s{seed} {arm}: onset={rec['p3_onset']:.3f} "
                  f"p3end={rec['p3_end']:.3f} cons={rec['cons']}",
                  flush=True)
    del mdl_f
    torch.cuda.empty_cache()

    # ---- LoRA arm: fresh init per seed, lr=1e-4 only ----
    for seed in SEEDS:
        need = [lr for lr in LORA_LRS
                if not existing(results, f"lora_lr{lr:g}", seed)]
        if not need:
            continue
        A1, B, A2 = load_domains(seed=seed)
        texts = [t for t, _ in A1 + B + A2]
        Y = np.array([l for _, l in A1 + B + A2],
                     dtype=float).reshape(-1, 1)
        bounds = [0, PA, PA + PB, PA + PB + PA_RET]
        for lr in need:
            arm = f"lora_lr{lr:g}"
            preds, dt, peak = run_lora(texts, Y, tok, dev,
                                       lr=lr, seed=seed)
            e = lambda lo, hi: phase_err(preds, Y[:, 0], lo, hi)
            rec = {"p1_end": e(bounds[1] - ONSET, bounds[1]),
                   "p2_end": e(bounds[2] - ONSET, bounds[2]),
                   "p3_onset": e(bounds[2], bounds[2] + ONSET),
                   "p3_end": e(bounds[3] - ONSET, bounds[3]),
                   "ms_per_step": dt / len(texts) * 1000,
                   "peak_mb": peak}
            results[TAG].setdefault(arm, []).append(rec)
            save(results)
            print(f"opt s{seed} {arm}: onset={rec['p3_onset']:.3f} "
                  f"p3end={rec['p3_end']:.3f} ({dt:.0f}s)",
                  flush=True)
            del preds
            torch.cuda.empty_cache()

    # ---- summary ----
    for tag, arms in results.items():
        for arm, recs in list(arms.items()):
            if isinstance(recs, dict):
                recs = recs["per_seed"]
            arms[arm] = {
                "per_seed": recs,
                "p3_onset": summarize([r["p3_onset"] for r in recs]),
                "p3_end": summarize([r["p3_end"] for r in recs])}
    save(results)
    print("done -> " + OUT)
    print(json.dumps(results, indent=2)[:4000])


main()
