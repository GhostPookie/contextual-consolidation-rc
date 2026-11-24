"""mamba_apia_spike.py - Mamba(SSM)+APIA hybrid on Kaggle GPU.
Frozen mamba-130m last-token hidden -> B-spline -> BlockDiagRLS (+CtxMem).
A->B->A' 20NG domain-return; predict-then-update, online, no backprop.
Prints __MAMBAARM__ JSON records for offline analysis.
"""
import numpy as np, time, os, sys, json

import subprocess, sys
def _ensure_tf():
    import transformers, pkg_resources
    v = pkg_resources.parse_version(transformers.__version__)
    if v < pkg_resources.parse_version("4.40"):
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "-q", "-U", "transformers", "accelerate"])
_ensure_tf()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.datasets import fetch_20newsgroups

LAM, INIT_VAR, WINDUP_MULT = 0.98, 1e2, 1e4
MODEL_ID = "state-spaces/mamba-130m-hf"
D_PROJ, K, ONSET = 100, 10, 100
PA, PB, PA_RET, MAX_TOK = 600, 600, 400, 256

"""
APIA 2.0 Research Prototype
Module: rls_engine.py
Purpose: map the reservoir state (x_t) to a GAM-style spline feature
      map (phi) and train a LINEAR readout on those features.
      Three methods are compared:
        (a) Batch Ridge Regression  -> O(N*F^2) training cost, offline
        (b) Full RLS                -> O(F^2) = O(D^2 K^2) per step
        (c) Block-Diagonal RLS      -> O(D*K^2) per step (~100x savings)

      Key point: a single-layer spline expansion + linear readout is
      linear in the weights, so it can be solved in CLOSED FORM by
      RLS/ridge. No "Hebbian" update is needed; RLS is the proven
      online algorithm that converges to the ridge solution.
"""

import numpy as np
import time


# ============================================================
# 1. SPLINE FEATURE MAP (GAM-style, fixed basis)
# ============================================================
class SplineFeatureMap:
    """
    Computes K B-spline basis functions per input component x_j over an
    open-uniform knot vector.

    phi(x) = [B(x_1), B(x_2), ..., B(x_D), 1.0]  in  R^{D*K + 1}

    The basis functions are FIXED (knots are not learned); only the
    readout coefficients are learned. This is the "KAN-inspired basis
    expansion + linear solve" approach in the spirit of KARC/HKAN.
    """

    def __init__(self, input_dim: int, n_basis: int = 10, degree: int = 3,
                 lo: float = -1.0, hi: float = 1.0):
        if n_basis < degree + 1:
            raise ValueError(
                f"n_basis ({n_basis}) must be >= degree+1 ({degree+1})")
        self.input_dim = input_dim
        self.n_basis = n_basis
        self.degree = degree
        self.lo, self.hi = lo, hi

        # Open-uniform knot vector: endpoints repeated degree+1 times.
        n_interior = n_basis - degree - 1
        interior = (np.linspace(lo, hi, n_interior + 2)[1:-1]
                    if n_interior > 0 else np.empty(0))
        self.knots = np.concatenate([
            np.full(degree + 1, lo), interior, np.full(degree + 1, hi)])
        # len(knots) = n_basis + degree + 1  (check)
        assert len(self.knots) == n_basis + degree + 1

        self.n_features = input_dim * n_basis + 1  # +1: bias

    def _basis_matrix(self, x: np.ndarray) -> np.ndarray:
        """Cox-de Boor: x (n,) -> basis values (n, n_basis)."""
        t = self.knots
        x = np.clip(np.asarray(x, dtype=float), t[0], t[-1])
        n = x.size

        # Degree 0: write 1 into the supporting interval of each x
        B = np.zeros((n, len(t) - 1))
        idx = np.clip(np.searchsorted(t, x, side='right') - 1,
                      0, len(t) - 2)
        B[np.arange(n), idx] = 1.0

        for d in range(1, self.degree + 1):
            nb = len(t) - d - 1
            # Left term: (x - t_i)/(t_{i+d} - t_i) * B_{i,d-1}
            dl = t[d:d + nb] - t[:nb]
            left = np.divide(
                x[:, None] - t[:nb][None, :],
                np.broadcast_to(dl[None, :], (n, nb)),
                out=np.zeros((n, nb)), where=(dl[None, :] > 0)
            ) * B[:, :nb]
            # Right term: (t_{i+d+1} - x)/(t_{i+d+1} - t_{i+1}) * B_{i+1,d-1}
            dr = t[d + 1:d + 1 + nb] - t[1:nb + 1]
            right = np.divide(
                t[d + 1:d + 1 + nb][None, :] - x[:, None],
                np.broadcast_to(dr[None, :], (n, nb)),
                out=np.zeros((n, nb)), where=(dr[None, :] > 0)
            ) * B[:, 1:nb + 1]
            B = left + right

        # Right endpoint (x == hi): the last basis function equals 1
        edge = x >= t[-1]
        if np.any(edge):
            B[edge] = 0.0
            B[edge, -1] = 1.0
        return B

    def transform(self, x: np.ndarray) -> np.ndarray:
        """x (D,) -> phi (D*K + 1,). Block layout: phi[j*K:(j+1)*K] = B(x_j)."""
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.input_dim:
            raise ValueError(f"Expected dim {self.input_dim}, "
                             f"got {x.shape[0]}")
        phi = self._basis_matrix(x).reshape(-1)
        return np.append(phi, 1.0)


# ============================================================
# 2. READOUT METHODS
# ============================================================
class RidgeReadout:
    """Batch ridge regression: W = (Phi^T Phi + lam I)^{-1} Phi^T Y."""

    def __init__(self, n_features: int, n_outputs: int):
        self.W = np.zeros((n_features, n_outputs))

    def fit(self, Phi: np.ndarray, Y: np.ndarray, lam: float = 1e-4):
        A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
        self.W = np.linalg.solve(A, Phi.T @ Y)
        return self

    def predict(self, phi: np.ndarray) -> np.ndarray:
        return phi @ self.W


class FullRLS:
    """
    Standard RLS with forgetting factor.
    Per-step cost: O(F^2) = O(D^2 K^2). Memory: F^2.
    P0 = init_var * I  (large init_var = weak prior confidence)
    Choosing lam < 1 gives exponential forgetting for continual learning.
    """

    def __init__(self, n_features: int, n_outputs: int,
                 forgetting: float = 1.0, init_var: float = 1e2,
                 windup_mult: float = None):
        self.W = np.zeros((n_features, n_outputs))
        self.P = np.eye(n_features) * init_var
        self.lam = forgetting
        self._step = 0
        # Covariance windup bound: if trace(P) exceeds this threshold,
        # P is softly scaled down (eigenvalue structure preserved).
        # None = unbounded.
        self.trace_bound = (init_var * n_features * windup_mult
                            if windup_mult is not None else None)

    def update(self, phi: np.ndarray, y: np.ndarray):
        phiP = phi @ self.P                      # (F,) - the only O(F^2)
        # matvec; P is symmetric so P@phi == phiP.T (no second matvec).
        s = self.lam + phiP @ phi
        g = phiP / s
        e = y - phi @ self.W                     # prediction error (predict first)
        self.W += np.outer(g, e)
        self.P -= np.outer(g, phiP)              # O(F^2)
        self.P /= self.lam
        # P-BOUNDING: with lam<1 the trace in dead directions grows like
        # lam^-t; if the threshold is crossed, softly rescale (not reset).
        if self.trace_bound is not None:
            tr = np.trace(self.P)
            if tr > self.trace_bound:
                self.P *= self.trace_bound / tr
        # Numerical symmetry drift accumulates slowly; maintenance every
        # 64 steps is sufficient.
        self._step += 1
        if self._step % 64 == 0:
            self.P = 0.5 * (self.P + self.P.T)
        return e

    def predict(self, phi: np.ndarray) -> np.ndarray:
        return phi @ self.W


class BlockDiagRLS:
    """
    Block-diagonal RLS approximation: instead of a full P (F x F), a
    separate P_j (K x K) is kept per input component j (+ scalar bias
    block).

      s = lam + sum_j phi_j^T P_j phi_j  (+ P_bias)
      g_j = P_j phi_j / s
      W_j += g_j e^T ;  P_j = (P_j - g_j (P_j phi_j)^T) / lam

    Per-step cost: O(D*K^2). Memory: D*K^2.
    D=100, K=10 -> ~100x fewer operations and memory than full RLS.
    """

    def __init__(self, input_dim: int, n_basis: int, n_outputs: int,
                 forgetting: float = 1.0, init_var: float = 1e2,
                 windup_mult: float = None):
        self.D, self.K = input_dim, n_basis
        self.n_features = input_dim * n_basis + 1
        self.W = np.zeros((self.n_features, n_outputs))
        self.P_blocks = np.tile(
            np.eye(n_basis) * init_var, (input_dim, 1, 1))
        self.P_bias = np.array([[init_var]])
        self.lam = forgetting
        # Per-block trace bound: stops dead-direction windup with an
        # O(K) check.
        self.trace_bound = (init_var * n_basis * windup_mult
                            if windup_mult is not None else None)
        self._bias_bound = (init_var * windup_mult
                            if windup_mult is not None else None)

    def update(self, phi: np.ndarray, y: np.ndarray):
        phi_b = phi[:self.D * self.K].reshape(self.D, self.K)
        pb = phi[-1]  # bias feature (=1.0)

        Q = np.einsum('dij,dj->di', self.P_blocks, phi_b)  # (D,K) O(DK^2)
        s = self.lam + np.sum(phi_b * Q) + self.P_bias[0, 0] * pb * pb

        e = y - phi @ self.W

        G = Q / s
        self.W[:self.D * self.K] += (G.reshape(-1, 1)) * e[None, :]
        self.P_blocks -= np.einsum('di,dj->dij', G, Q)
        self.P_blocks /= self.lam

        # P-BOUNDING (soft): blocks exceeding the trace threshold are
        # rescaled; eigenvalue structure preserved, O(K) check cost.
        if self.trace_bound is not None:
            tr = np.einsum('dii->d', self.P_blocks)
            over = tr > self.trace_bound
            if np.any(over):
                self.P_blocks[over] *= (self.trace_bound
                                        / tr[over])[:, None, None]

        g_b = self.P_bias[0, 0] * pb / s
        self.W[-1] += g_b * e
        self.P_bias = (self.P_bias - np.array([[g_b * self.P_bias[0, 0] * pb]])) / self.lam
        if self._bias_bound is not None and self.P_bias[0, 0] > self._bias_bound:
            self.P_bias[0, 0] = self._bias_bound
        return e

    def predict(self, phi: np.ndarray) -> np.ndarray:
        return phi @ self.W

    # ---- Memory-transfer interface (for memory.py) ----
    def get_weights(self) -> np.ndarray:
        return self.W.copy()

    def set_weights(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float).copy()

    def inflate_P(self, factor: float):
        """Reopens plasticity by inflating P after consolidation."""
        self.P_blocks *= factor
        self.P_bias *= factor


# ============================================================
# 3. METRICS AND DATA GENERATION
# ============================================================
def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized MSE: ||y - yhat||^2 / ||y - mean(y)||^2  (1.0 = mean predictor)"""
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return float(num / den)


def mackey_glass(n: int, tau: int = 17, beta: float = 0.2,
                 gamma: float = 0.1, seed: int = 0) -> np.ndarray:
    """Standard RC benchmark: Mackey-Glass tau=17 series (Euler, dt=1)."""
    rng = np.random.default_rng(seed)
    x = np.full(n + tau, 1.2) + 0.01 * rng.standard_normal(n + tau)
    for t in range(tau, n + tau - 1):
        x[t + 1] = (x[t] - gamma * x[t]
                    + beta * x[t - tau] / (1.0 + x[t - tau] ** 10))
    return x[tau:]


# ============================================================
# 4. COMPARISON PROTOCOL (Mackey-Glass 1-step prediction)
# ============================================================
if __name__ == "__main__":
    print("=== APIA 2.0 RLS Engine Comparison (Mackey-Glass, 1-step) ===\n")

    # --- Protocol parameters ---
    D, K = 100, 10                 # reservoir size, basis count
    N_TOTAL, WASHOUT = 4500, 500   # series length, transient discarded
    N_TRAIN, N_TEST = 2000, 1000
    SEED = 42

    from reservoir import ReservoirEngine

    # --- 1. Data: input and target standardized ---
    series = mackey_glass(N_TOTAL, seed=SEED)
    mu, sd = series.mean(), series.std()
    u_all = ((series - mu) / sd).astype(float)
    y_all = np.roll(u_all, -1)     # target: next step

    # --- 2. Extract the reservoir state sequence ---
    res = ReservoirEngine(input_dim=1, reservoir_size=D, sparsity=0.1,
                          leakage_rate=0.3, input_scaling=0.8, seed=SEED)
    fmap = SplineFeatureMap(input_dim=D, n_basis=K, degree=3)
    F = fmap.n_features
    print(f"Feature dim F = D*K+1 = {F}  "
          f"(Full RLS P: {F}x{F} = {F*F*8/1e6:.1f} MB, "
          f"Block RLS P: {D}x{K}x{K} = {D*K*K*8/1e3:.0f} KB)\n")

    states = np.zeros((N_TOTAL, D))
    for t in range(N_TOTAL):
        states[t] = res.step(np.array([u_all[t]]))

    tr = slice(WASHOUT, WASHOUT + N_TRAIN)
    te = slice(WASHOUT + N_TRAIN, WASHOUT + N_TRAIN + N_TEST)

    # Precompute phis (isolates readout timing)
    Phi_tr = np.array([fmap.transform(states[t]) for t in range(*tr.indices(N_TOTAL))])
    Y_tr = y_all[tr].reshape(-1, 1)
    Phi_te = np.array([fmap.transform(states[t]) for t in range(*te.indices(N_TOTAL))])
    Y_te = y_all[te].reshape(-1, 1)

    # --- 3a. RIDGE (batch): pick lam via held-out train split ---
    lam_grid = [1e-8, 1e-6, 1e-4, 1e-2]
    split = int(0.8 * N_TRAIN)
    best = min(lam_grid, key=lambda l: nmse(
        Y_tr[split:], Phi_tr[split:] @
        RidgeReadout(F, 1).fit(Phi_tr[:split], Y_tr[:split], l).W))
    ridge = RidgeReadout(F, 1).fit(Phi_tr, Y_tr, best)
    print(f"[RIDGE] lam={best:g}  NMSE_test = "
          f"{nmse(Y_te, Phi_te @ ridge.W):.6f}")

    # --- 3b. FULL RLS (online) ---
    rls = FullRLS(F, 1, forgetting=1.0, init_var=1e2)
    t0 = time.perf_counter()
    for i in range(N_TRAIN):
        rls.update(Phi_tr[i], Y_tr[i])
    dt_full = (time.perf_counter() - t0) / N_TRAIN * 1e6
    print(f"[FULL RLS] NMSE_test = {nmse(Y_te, Phi_te @ rls.W):.6f}   "
          f"avg update = {dt_full:.0f} us/step")

    # --- 3c. BLOCK-DIAG RLS (online) ---
    brls = BlockDiagRLS(D, K, 1, forgetting=1.0, init_var=1e2)
    t0 = time.perf_counter()
    for i in range(N_TRAIN):
        brls.update(Phi_tr[i], Y_tr[i])
    dt_blk = (time.perf_counter() - t0) / N_TRAIN * 1e6
    print(f"[BLK-RLS ] NMSE_test = {nmse(Y_te, Phi_te @ brls.W):.6f}   "
          f"avg update = {dt_blk:.0f} us/step")

    print(f"\nSpeedup Full->Block: {dt_full/dt_blk:.1f}x")
    print(f"Memory Full->Block:  {(F*F)/(D*K*K):.0f}x less")

"""
APIA 2.0 Research Prototype
Module: contextual_memory.py
Purpose: N-slot contextual memory (context-gated memory).

Motivation (from the failure of memory.py):
    Unconditional consolidation moved forgetting into W_slow because
    the transfer was regime-agnostic. Fix: each slot represents a
    context, and consolidation writes ONLY to the active slot the
    gate is confident about.

Prediction model (soft gate):
      y_hat = phi'W_fast + beta * phi' * (sum_i w_i * W_slow_i)
      w_i   = softmax(-eta * E_i / (E_min + eps))

      E_i : slow EMA error of slot i (context-match measure).
      Ratioing by E_min -> scale-free hard discrimination.
      An empty slot (W_slow=0) returns the residual error -> new
      regimes naturally flow toward empty slots.

Consolidation (correct slot semantics, bounded jump):
      W_slow_k <- W_slow_k + (1-gamma)/beta * W_fast
      W_fast   <- gamma * W_fast

      Each slot represents its context INDEPENDENTLY (when w_k=1 its
      contribution has the correct magnitude). For w_k<1 the jump is
      bounded by (1-w_k)(1-gamma)|phi'W_fast|; when the gate is
      confident (w_k~1) the jump goes to ~0. The (1-gamma)/(beta*w_k)
      rule was deliberately NOT used: it preserves instantaneous
      continuity but overcharges the slot at w<1 and induces gate
      oscillation.

Contamination protection:
      Consolidation fires only when the period arrives AND the
      short-term total error e_fast < e_gate*signal_var. Transition
      window = high error = writing stops -> gives the gate time to
      identify the true context. Target = argmin E; empty slots tie
      on error, lowest index wins (bootstrap starts at slot 0).
"""

import numpy as np


class ContextualMemory:
    def __init__(self, model, n_slots: int = 3, beta: float = 1.0,
                 gamma: float = 0.1, eta: float = 2.0,
                 alpha_ema: float = 0.01, alpha_fast: float = 0.1,
                 consolidate_every: int = 250, e_gate: float = 0.5,
                 signal_var: float = 1.0, tau: float = 0.5,
                 p_inflate: float = 1.5, eps: float = 1e-8,
                 alpha_m2: float = 0.02,
                 err_mode: str = 'slow', write_protect: bool = True,
                 empty_rule: str = 'count',
                 saturation: str = 'overwrite',
                 w_fast_cap: float = None):
        """
        :param model: RLS engine with get_weights/set_weights/inflate_P
        :param n_slots: number of context slots (fixed)
        :param beta: gain of W_slow's contribution to prediction
        :param gamma: fraction of W_fast retained at consolidation
        :param eta: gate sharpness (applied to the E/E_min ratio)
        :param alpha_ema: slot-error EMA coefficient (for gate, slow)
        :param alpha_fast: total-error EMA coefficient (for trigger, fast)
        :param consolidate_every: consolidation period (steps)
        :param e_gate: max fast error allowing consolidation
                       (as a multiple of signal_var)
        :param signal_var: initial prior estimate of E[y^2] (in place
               of a non-causal global variance; overwritten by the
               online estimate within ~1/alpha_m2 steps)
        :param alpha_m2: E[y^2] EMA coefficient (signal-energy measure)
        :param p_inflate: P inflation factor at consolidation
        """
        self.model = model
        self.n_slots = n_slots
        self.beta, self.gamma = beta, gamma
        self.eta = eta
        self.alpha_ema, self.alpha_fast = alpha_ema, alpha_fast
        self.consolidate_every = consolidate_every
        self.e_gate = e_gate
        # CAUSALITY: thresholds use the online E[y^2] estimate instead
        # of var(y). The empty-slot floor is E[y^2] anyway (M4); for
        # zero-mean signals E[y^2]=var(y), so behavior matches earlier
        # results. The initial value is a non-informative prior
        # estimate; no future data is used.
        self.alpha_m2 = alpha_m2
        self._m2 = float(signal_var)
        # Context-aware target threshold: E_k < tau*var => reinforce;
        # otherwise write to the emptiest slot (contamination
        # protection).
        self.tau = tau
        self.p_inflate = p_inflate
        self.eps = eps
        # Ablation flags (for Fig5 / Table 3):
        #   err_mode='total' -> v1 self-masking: slot error includes W_fast
        #   write_protect=False -> blind argmin target without tau*var gate
        #   empty_rule='norm' -> v2: ||W_slow|| argmin (faulty occupancy proxy)
        self.err_mode = err_mode
        self.write_protect = write_protect
        self.empty_rule = empty_rule
        # Saturation policy (when ALL slots are populated and NEW_CTX
        # arrives):
        #   'overwrite' -> write over argmin c_i (least-consolidated);
        #                  occupancy-weighted eviction, tends to
        #                  protect the oldest/strongest context but
        #                  gives NO guarantee.
        #   'refuse'    -> refuse the write; all stored contexts are
        #                  preserved, the new regime lives only in
        #                  W_fast (memoryless).
        self.saturation = saturation
        # Valve-isolation ablation: refuse + norm-clamp. None = off.
        # Non-consolidation drainage of W_fast — controls the bloat on
        # its own; does NOT touch transfer/consolidation behavior.
        self.w_fast_cap = w_fast_cap

        W0 = model.get_weights()                       # (F, m)
        self.W_slow = np.zeros((n_slots,) + W0.shape)  # (S, F, m)
        self.E = np.zeros(n_slots)                     # slot EMA errors
        self._e_fast = 0.0                             # total fast error
        self._steps = 0
        self.n_consolidations = 0
        self.max_discontinuity = 0.0
        self.slot_log = []                             # gate argmax history
        self.gate_log = []                             # gate weight vector (S,) history
        self.cons_log = []                             # (step, slot, mode) history
        # Occupancy proxy: count, not norm. Regime weight norms are
        # incomparable; how many consolidations a slot received is
        # neutral and comparable.
        self.slot_cons_counts = np.zeros(n_slots, dtype=int)

    # ---------- internal helpers ----------
    def _slot_errors(self, phi, y):
        """
        Context-match error: e_i = ||y - beta*phi'W_slow_i||^2 -> (S,)

        Deliberately EXCLUDES W_fast: W_fast learns to compensate for
        the active slot's contribution (self-masking) and masks slot
        quality -> the gate locks up. The pure slow-memory error is
        both a context detector and an empty-slot detector (empty:
        e = var(y)).
        """
        slow = np.einsum('f,sfm->sm', phi, self.W_slow)   # (S,m)
        if self.err_mode == 'total':
            # v1 ablation: include the W_fast prediction (self-masking).
            slow = slow + (phi @ self.model.get_weights())[None, :] / self.beta
        return np.sum((y[None, :] - self.beta * slow) ** 2, axis=1)

    def _gate(self):
        """w_i ∝ exp(-eta * E_i/(E_min+eps)); (S,) weights."""
        e_min = max(self.E.min(), self.eps)
        z = -self.eta * self.E / e_min
        z -= z.max()  # numerical safety
        w = np.exp(z)
        return w / w.sum()

    def _slow_mix(self, phi, w):
        """beta * phi' * (sum_i w_i W_slow_i) -> (m,)"""
        W_mix = np.einsum('s,sfm->fm', w, self.W_slow)
        return self.beta * (phi @ W_mix)

    # ---------- main interface ----------
    def predict(self, phi):
        return self.model.predict(phi) + self._slow_mix(phi, self._gate())

    def update(self, phi, y):
        # 1) Context-match errors (pure W_slow, no W_fast)
        self.E = ((1 - self.alpha_ema) * self.E
                  + self.alpha_ema * self._slot_errors(phi, y))

        # 2) Update W_fast on the GATED residual target
        w = self._gate()
        residual = y - self._slow_mix(phi, w)
        e_total = self.model.update(phi, residual)

        # 3) Fast EMA for the trigger + online E[y^2] for signal energy
        e_now = float(np.sum(e_total ** 2))
        self._e_fast = ((1 - self.alpha_fast) * self._e_fast
                        + self.alpha_fast * e_now)
        self._m2 = ((1 - self.alpha_m2) * self._m2
                    + self.alpha_m2 * float(np.sum(y * y)))
        # Single-variable intervention: norm-clamp (independent of
        # consolidation)
        if self.w_fast_cap is not None:
            wf = self.model.get_weights()
            n = float(np.linalg.norm(wf))
            if n > self.w_fast_cap:
                self.model.set_weights(wf * (self.w_fast_cap / n))

        # 4) Periodic + stable + CONTEXT-AWARE consolidation.
        #    E_min < tau*var  -> REINFORCE: write to argmin slot
        #    otherwise        -> NEW CONTEXT: write to the emptiest
        #                        slot. Never write to a populated
        #                        high-error slot -> contamination
        #                        stops (slot-poisoning fix).
        self._steps += 1
        self.slot_log.append(int(np.argmax(w)))
        self.gate_log.append(w.copy())
        if (self._steps % self.consolidate_every == 0
                and self._e_fast <= self.e_gate * self._m2):
            k = int(np.argmin(self.E))
            if not self.write_protect or self.E[k] < self.tau * self._m2:
                target, mode = k, 0            # REINFORCE (v1: blind argmin)
            elif self.empty_rule == 'count':
                # Emptiest slot = slot with fewest consolidations.
                # The invariant "never write to a populated slot" is
                # only defined while an empty slot exists: at
                # saturation, 'refuse' skips the write, 'overwrite'
                # evicts the least-consolidated context (mode=3).
                if (self.slot_cons_counts.min() > 0
                        and self.saturation == 'refuse'):
                    target, mode = -1, 2
                else:
                    target = int(np.argmin(self.slot_cons_counts))
                    mode = 1 if self.slot_cons_counts[target] == 0 else 3
            else:
                # v2 ablation: norm-based occupancy (faulty proxy)
                norms = np.linalg.norm(
                    self.W_slow.reshape(self.n_slots, -1), axis=1)
                target, mode = int(np.argmin(norms)), 1
            if target >= 0:
                self.consolidate(phi, target)
            self.cons_log.append((self._steps, target, mode))
        return e_total

    def consolidate(self, phi_probe, k):
        """Inject (1-gamma)/beta * W_fast into slot k, decay W_fast."""
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


def load_domains(seed=7):
    domA = fetch_20newsgroups(subset="all",
        categories=["sci.space", "rec.sport.baseball"],
        remove=("headers","footers","quotes"), shuffle=True, random_state=0)
    domB = fetch_20newsgroups(subset="all",
        categories=["comp.graphics","talk.politics.misc"],
        remove=("headers","footers","quotes"), shuffle=True, random_state=1)
    A = list(zip(domA.data, domA.target))
    B = list(zip(domB.data, domB.target))
    rng = np.random.default_rng(seed)
    rng.shuffle(A); rng.shuffle(B)
    return A[:PA], B[:PB], A[PA:PA+PA_RET]


@torch.no_grad()
def embed(texts, tok, mdl, dev, batch=16):
    d = mdl.config.hidden_size
    H = np.empty((len(texts), d), dtype=np.float32)
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i+batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_TOK).to(dev)
        hs = mdl(**enc, output_hidden_states=True).hidden_states[-1]
        idx = enc["attention_mask"].sum(1) - 1
        H[i:i+batch] = hs[torch.arange(len(idx)), idx].float().cpu().numpy()
    return H


def run_stream(Phi, Y, kind, e_gate=0.5, eta=2.0, alpha_ema=0.01,
               hard_gate=False):
    d = Phi.shape[1] // K
    base = BlockDiagRLS(d, K, 1, forgetting=LAM, init_var=INIT_VAR,
                        windup_mult=WINDUP_MULT)
    if kind == "ctxmem":
        model = ContextualMemory(base, n_slots=3, beta=1.0, gamma=0.1,
                                 eta=eta, alpha_ema=alpha_ema,
                                 consolidate_every=250, e_gate=e_gate,
                                 p_inflate=1.5)
    else:
        model = base
    preds = np.empty(len(Phi))
    for i, phi in enumerate(Phi):
        preds[i] = float(model.predict(phi)[0])
        model.update(phi, Y[i])
    return (preds > 0.5).astype(float), model


def perr(p, y, lo, hi):
    return float(np.mean(p[lo:hi] != y[lo:hi]))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # last token = true sequence end
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32).to(dev).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print("params MB: %.0f" % (sum(p.numel() for p in mdl.parameters())*4/1e6))

    A1, B, A2 = load_domains()
    stream = A1 + B + A2
    Y = np.array([l for _, l in stream], float).reshape(-1, 1)

    t0 = time.perf_counter()
    H = embed([t for t, _ in stream], tok, mdl, dev)
    t_feat = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated()/1e6 if dev == "cuda" else 0
    print("features: %dx%d in %.0fs, peak VRAM %.0f MB"
          % (H.shape[0], H.shape[1], t_feat, vram), flush=True)

    H = (H - H.mean(0)) / (H.std(0) + 1e-6)
    rng = np.random.default_rng(12345)
    R = rng.standard_normal((H.shape[1], D_PROJ)) / np.sqrt(H.shape[1])
    feats = {"proj-100": np.tanh(H @ R),
             "raw-d": np.clip(H, -3.0, 3.0) / 3.0}

    bounds = [0, PA, PA+PB, PA+PB+PA_RET]
    res = {"model": MODEL_ID, "bounds": bounds, "feat_s": t_feat,
           "vram_mb": vram, "runs": []}
    for tag, X in feats.items():
        fmap = SplineFeatureMap(X.shape[1], K)
        Phi = np.stack([fmap.transform(x) for x in X])
        print("--- features:", tag, "F=", Phi.shape[1], flush=True)
        for lab, kw in (("block", {}),
                        ("ctxmem_soft", dict(e_gate=1.0, eta=5.0,
                                             alpha_ema=0.05))):
            t0 = time.perf_counter()
            preds, model = run_stream(Phi, Y,
                "ctxmem" if "ctx" in lab else "block", **kw)
            dt = time.perf_counter() - t0
            r = dict(arm=lab, feat=tag,
                     p1_end=perr(preds, Y[:,0], bounds[1]-ONSET, bounds[1]),
                     p2_end=perr(preds, Y[:,0], bounds[2]-ONSET, bounds[2]),
                     p3_onset=perr(preds, Y[:,0], bounds[2], bounds[2]+ONSET),
                     p3_end=perr(preds, Y[:,0], bounds[3]-ONSET, bounds[3]),
                     cons=getattr(model, "n_consolidations", 0),
                     wall_s=round(dt, 1))
            res["runs"].append(r)
            print("__MAMBAARM__" + json.dumps(r), flush=True)
    json.dump(res, open("/kaggle/working/mamba_apia_results.json", "w"),
              indent=1)
    print("MAMBA_SPIKE_DONE", flush=True)


main()
