# kaggle_cls_opt13b.py — OPT-1.3B accuracy arm on Kaggle T4
# A->B->A' single-pass classification stream, predict-before-update.
# Arms: apia-block | apia-ctxmem | lora r=8 @ lr in {1e-4, 1e-3}
# Mirrors cls_spike.py/adapter_spike.py exactly; engine inlined
# (self-contained: Kaggle notebook does not clone the repo).
#
# Kaggle prerequisites:
#   pip uninstall -y torchao          # peft compatibility (0.10.0 too old)
#   internet ON (HF model + sklearn 20newsgroups download)
import numpy as np
import torch, time, json
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.datasets import fetch_20newsgroups

MODEL_ID = "facebook/opt-1.3b"
K = 10
ONSET = 100
PA, PB, PA_RET = 600, 600, 400
MAX_TOK = 128
LAM, INIT_VAR, WINDUP_MULT = 0.98, 1e2, 1e4
OUT = "/kaggle/working/cls_opt13b.json"


# ================= engine (verbatim ports) =================
class SplineFeatureMap:
    def __init__(self, input_dim, n_basis=10, degree=3, lo=-1.0, hi=1.0):
        self.input_dim, self.n_basis, self.degree = input_dim, n_basis, degree
        n_interior = n_basis - degree - 1
        interior = (np.linspace(lo, hi, n_interior + 2)[1:-1]
                    if n_interior > 0 else np.empty(0))
        self.knots = np.concatenate(
            [np.full(degree + 1, lo), interior, np.full(degree + 1, hi)])
        self.n_features = input_dim * n_basis + 1

    def _basis_matrix(self, x):
        t = self.knots
        x = np.clip(np.asarray(x, dtype=float), t[0], t[-1])
        n = x.size
        B = np.zeros((n, len(t) - 1))
        idx = np.clip(np.searchsorted(t, x, side='right') - 1, 0, len(t) - 2)
        B[np.arange(n), idx] = 1.0
        for d in range(1, self.degree + 1):
            nb = len(t) - d - 1
            dl = t[d:d + nb] - t[:nb]
            left = np.divide(x[:, None] - t[:nb][None, :],
                             np.broadcast_to(dl[None, :], (n, nb)),
                             out=np.zeros((n, nb)), where=(dl[None, :] > 0)) * B[:, :nb]
            dr = t[d + 1:d + 1 + nb] - t[1:nb + 1]
            right = np.divide(t[d + 1:d + 1 + nb][None, :] - x[:, None],
                              np.broadcast_to(dr[None, :], (n, nb)),
                              out=np.zeros((n, nb)), where=(dr[None, :] > 0)) * B[:, 1:nb + 1]
            B = left + right
        edge = x >= t[-1]
        if np.any(edge):
            B[edge] = 0.0
            B[edge, -1] = 1.0
        return B

    def transform(self, x):
        return np.append(self._basis_matrix(np.asarray(x, dtype=float)).reshape(-1), 1.0)


class BlockDiagRLS:
    def __init__(self, input_dim, n_basis, n_outputs,
                 forgetting=1.0, init_var=1e2, windup_mult=None):
        self.D, self.K = input_dim, n_basis
        self.n_features = input_dim * n_basis + 1
        self.W = np.zeros((self.n_features, n_outputs))
        self.P_blocks = np.tile(np.eye(n_basis) * init_var, (input_dim, 1, 1))
        self.P_bias = np.array([[init_var]])
        self.lam = forgetting
        self.trace_bound = (init_var * n_basis * windup_mult
                            if windup_mult is not None else None)
        self._bias_bound = init_var * windup_mult if windup_mult is not None else None

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
                self.P_blocks[over] *= (self.trace_bound / tr[over])[:, None, None]
        g_b = self.P_bias[0, 0] * pb / s
        self.W[-1] += g_b * e
        self.P_bias = (self.P_bias - np.array([[g_b * self.P_bias[0, 0] * pb]])) / self.lam
        if self._bias_bound is not None and self.P_bias[0, 0] > self._bias_bound:
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


class ContextualMemory:
    def __init__(self, model, n_slots=3, beta=1.0, gamma=0.1, eta=2.0,
                 alpha_ema=0.01, alpha_fast=0.1, consolidate_every=250,
                 e_gate=0.5, signal_var=1.0, tau=0.5, p_inflate=1.5,
                 eps=1e-8, alpha_m2=0.02):
        self.model = model
        self.n_slots, self.beta, self.gamma, self.eta = n_slots, beta, gamma, eta
        self.alpha_ema, self.alpha_fast = alpha_ema, alpha_fast
        self.consolidate_every, self.e_gate = consolidate_every, e_gate
        self.tau, self.p_inflate, self.eps = tau, p_inflate, eps
        self.alpha_m2 = alpha_m2
        self._m2 = float(signal_var)
        W0 = model.get_weights()
        self.W_slow = np.zeros((n_slots,) + W0.shape)
        self.E = np.zeros(n_slots)
        self._e_fast = 0.0
        self._steps = 0
        self.n_consolidations = 0
        self.slot_cons_counts = np.zeros(n_slots, dtype=int)

    def _slot_errors(self, phi, y):
        slow = np.einsum('f,sfm->sm', phi, self.W_slow)
        return np.sum((y[None, :] - self.beta * slow) ** 2, axis=1)

    def _gate(self):
        e_min = max(self.E.min(), self.eps)
        z = -self.eta * self.E / e_min
        z -= z.max()
        w = np.exp(z)
        return w / w.sum()

    def _slow_mix(self, phi, w):
        return self.beta * (phi @ np.einsum('s,sfm->fm', w, self.W_slow))

    def predict(self, phi):
        return self.model.predict(phi) + self._slow_mix(phi, self._gate())

    def update(self, phi, y):
        self.E = ((1 - self.alpha_ema) * self.E
                  + self.alpha_ema * self._slot_errors(phi, y))
        w = self._gate()
        e_total = self.model.update(phi, y - self._slow_mix(phi, w))
        e_now = float(np.sum(e_total ** 2))
        self._e_fast = ((1 - self.alpha_fast) * self._e_fast
                        + self.alpha_fast * e_now)
        self._m2 = ((1 - self.alpha_m2) * self._m2
                    + self.alpha_m2 * float(np.sum(y * y)))
        self._steps += 1
        if (self._steps % self.consolidate_every == 0
                and self._e_fast <= self.e_gate * self._m2):
            k = int(np.argmin(self.E))
            if self.E[k] < self.tau * self._m2:
                target = k
            elif self.slot_cons_counts.min() > 0:
                target = -1                      # refuse at saturation
            else:
                target = int(np.argmin(self.slot_cons_counts))
            if target >= 0:
                self._consolidate(target)
        return e_total

    def _consolidate(self, k):
        W_fast = self.model.get_weights()
        self.W_slow[k] += (1.0 - self.gamma) / self.beta * W_fast
        self.model.set_weights(self.gamma * W_fast)
        if self.p_inflate != 1.0:
            self.model.inflate_P(self.p_inflate)
        self.n_consolidations += 1
        self.slot_cons_counts[k] += 1


# ================= protocol =================
def load_domains():
    domA = fetch_20newsgroups(subset="all",
                              categories=["sci.space", "rec.sport.baseball"],
                              remove=("headers", "footers", "quotes"),
                              shuffle=True, random_state=0)
    domB = fetch_20newsgroups(subset="all",
                              categories=["comp.graphics", "talk.politics.misc"],
                              remove=("headers", "footers", "quotes"),
                              shuffle=True, random_state=1)
    A = list(zip(domA.data, domA.target))
    B = list(zip(domB.data, domB.target))
    rng = np.random.default_rng(7)
    rng.shuffle(A); rng.shuffle(B)
    return A[:PA], B[:PB], A[PA:PA + PA_RET]


@torch.no_grad()
def embed(texts, tok, mdl, dev, batch=32):
    d = mdl.config.hidden_size
    H = np.empty((len(texts), d), dtype=np.float32)
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_TOK).to(dev)
        out = mdl(**enc).last_hidden_state
        idx = enc["attention_mask"].sum(1) - 1
        H[i:i + batch] = out[torch.arange(len(idx)), idx].cpu().numpy()
    return H


def run_stream(Phi, Y, kind):
    d = (Phi.shape[1] - 1) // K
    base = BlockDiagRLS(d, K, 1, forgetting=LAM,
                        init_var=INIT_VAR, windup_mult=WINDUP_MULT)
    model = (ContextualMemory(base, n_slots=3, beta=1.0, gamma=0.1,
                              eta=5.0, alpha_ema=0.05,
                              consolidate_every=250, e_gate=1.0,
                              p_inflate=1.5)
             if kind == "ctxmem" else base)
    preds = np.empty(len(Phi))
    for i, phi in enumerate(Phi):
        preds[i] = float(model.predict(phi)[0])
        model.update(phi, Y[i])
    return (preds > 0.5).astype(float), model


def run_lora(texts, Y, tok, dev, lr):
    from peft import LoraConfig, get_peft_model
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, dtype=torch.float16)
    base.config.pad_token_id = tok.pad_token_id
    cfg = LoraConfig(r=8, lora_alpha=16,
                     target_modules=["q_proj", "v_proj"],
                     lora_dropout=0.0, bias="none")
    mdl = get_peft_model(base, cfg).to(dev)
    opt = torch.optim.AdamW(
        [p for p in mdl.parameters() if p.requires_grad], lr=lr)
    enc = tok(texts, return_tensors="pt", padding=True,
              truncation=True, max_length=MAX_TOK)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    preds = np.empty(len(texts))
    t0 = time.perf_counter()
    mdl.train()
    for i in range(len(texts)):
        inp = {"input_ids": ids[i:i + 1].to(dev),
               "attention_mask": mask[i:i + 1].to(dev)}
        out = mdl(**inp, labels=torch.tensor([int(Y[i, 0])], device=dev))
        preds[i] = float(out.logits.argmax(-1).item())
        out.loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if (i + 1) % 200 == 0:
            print(f"    lora lr={lr:g}: {i+1}/{len(texts)}", flush=True)
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e6
    del mdl, opt, base
    torch.cuda.empty_cache()
    return preds, dt, peak


def phase_err(preds, y, lo, hi):
    return float(np.mean(preds[lo:hi] != y[lo:hi]))


def report(res, lab, preds, Y, bounds, extra=""):
    p = lambda lo, hi: phase_err(preds, Y[:, 0], lo, hi)
    r = {"p1_end": p(bounds[1] - ONSET, bounds[1]),
         "p2_end": p(bounds[2] - ONSET, bounds[2]),
         "p3_onset": p(bounds[2], bounds[2] + ONSET),
         "p3_end": p(bounds[3] - ONSET, bounds[3])}
    res[lab] = r
    print(f"{lab:>14}: P1-end={r['p1_end']:.3f} P2-end={r['p2_end']:.3f} "
          f"P3-onset={r['p3_onset']:.3f} P3-end={r['p3_end']:.3f} {extra}",
          flush=True)


def main():
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token

    A1, B, A2 = load_domains()
    stream = A1 + B + A2
    texts = [t for t, _ in stream]
    Y = np.array([l for _, l in stream], dtype=float).reshape(-1, 1)
    bounds = [0, PA, PA + PB, PA + PB + PA_RET]
    res = {"model": MODEL_ID, "seq": MAX_TOK, "onset": ONSET,
           "phases": [PA, PB, PA_RET],
           "domains": {"A": ["sci.space", "rec.sport.baseball"],
                        "B": ["comp.graphics", "talk.politics.misc"]}}

    # ---- APIA arms: frozen OPT features -> spline -> RLS ----
    mdl_f = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(dev).eval()
    for p in mdl_f.parameters():
        p.requires_grad_(False)
    print("extracting features (batch=32)...", flush=True)
    H = np.concatenate([embed([t for t, _ in s], tok, mdl_f, dev)
                        for s in (A1, B, A2)])
    del mdl_f
    torch.cuda.empty_cache()

    H = (H - H.mean(0)) / (H.std(0) + 1e-6)      # stream-wide stats (known limitation)
    X = np.clip(H, -3.0, 3.0) / 3.0
    fmap = SplineFeatureMap(X.shape[1], K)
    print(f"spline map d={X.shape[1]} F={fmap.n_features} ...", flush=True)
    Phi = np.stack([fmap.transform(x) for x in X])

    t0 = time.perf_counter()
    preds, _ = run_stream(Phi, Y, "block")
    report(res, "apia_block", preds, Y, bounds, f"({time.perf_counter()-t0:.0f}s)")

    t0 = time.perf_counter()
    preds, mem = run_stream(Phi, Y, "ctxmem")
    report(res, "apia_ctxmem", preds, Y, bounds,
           f"cons={mem.n_consolidations} ({time.perf_counter()-t0:.0f}s)")

    # ---- LoRA arms ----
    for lr in (1e-4, 1e-3):
        preds, dt, peak = run_lora(texts, Y, tok, dev, lr)
        report(res, f"lora_lr{lr:g}", preds, Y, bounds,
               f"({dt:.0f}s, peak={peak:.0f}MB)")

    json.dump({"cls_opt13b": res}, open(OUT, "w"), indent=2)
    print("saved ->", OUT)


main()
