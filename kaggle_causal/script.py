# kaggle_native_lm.py - self-contained native APIA LM spike.
# Auto-generated: ReservoirEngine + SplineFeatureMap + BlockDiagRLS +
# ContextualMemory inlined from repo modules; runner from
# native_apia_lm.py. Run in a Kaggle cell (internet ON for
# tiny-shakespeare download).
import argparse
import json
import os
import time
import urllib.request
from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm as sparse_norm

DENSE_SVD_LIMIT = 2048

SHAKESPEARE_URL = ('https://raw.githubusercontent.com/karpathy/'
                   'char-rnn/master/data/tinyshakespeare/input.txt')
DATA_PATH = '/kaggle/working/tinyshakespeare.txt'




class ReservoirEngine:
    def __init__(self, input_dim: int, reservoir_size: int,
                 sparsity: float = 0.1, leakage_rate: float = 0.3,
                 input_scaling: float = 1.0, seed: int = 42):
        """
        :param input_dim: size of the input vector (e.g. lagged history L)
        :param reservoir_size: number of reservoir neurons (D)
        :param sparsity: nonzero fraction in the W_res matrix
        :param leakage_rate: alpha (leak rate). 0 < alpha <= 1
        :param input_scaling: amplitude scale of the W_in matrix
        :param seed: random seed for reproducibility
        """
        if not (0.0 < leakage_rate <= 1.0):
            raise ValueError(f"leakage_rate must be in (0, 1], got {leakage_rate}")
        if not (0.0 < sparsity <= 1.0):
            raise ValueError(f"sparsity must be in (0, 1], got {sparsity}")

        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.sparsity = sparsity
        self.alpha = leakage_rate
        self.input_scaling = input_scaling
        self.seed = seed

        # Module-local RNG: does not pollute global np.random state.
        self.rng = np.random.default_rng(self.seed)

        # 1. W_in: input weights, drawn uniform on [-1, 1] and scaled.
        self.W_in = self.rng.uniform(
            -1.0, 1.0, (reservoir_size, input_dim)) * self.input_scaling

        # 2. W_res: reservoir weights (sparse matrix)
        self.W_res = self._generate_sparse_reservoir_matrix()

        # 3. ESP guarantee: (1 - alpha) + alpha * ||W_res||_2 < 1
        self._enforce_esp_condition()

        # Initial state (zero vector)
        self.state = np.zeros(reservoir_size)

    def _generate_sparse_reservoir_matrix(self) -> sp.csr_matrix:
        """
        Builds a random reservoir weight matrix with the given sparsity;
        entries lie in [-1, 1].

        IMPORTANT: replace=False samples unique (row, col) pairs.
        Generating indices with np.random.randint leads to duplicate
        indices that csr_matrix SUMS together, which both reduces the
        effective sparsity and pushes entries outside [-1, 1].
        """
        total_elements = self.reservoir_size * self.reservoir_size
        num_nonzero = int(total_elements * self.sparsity)

        flat_idx = self.rng.choice(total_elements, size=num_nonzero,
                                   replace=False)
        rows, cols = np.divmod(flat_idx, self.reservoir_size)
        data = self.rng.uniform(-1.0, 1.0, num_nonzero)

        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.reservoir_size, self.reservoir_size))

    def _spectral_norm(self) -> float:
        """||W_res||_2: exact SVD for small matrices, svds for large."""
        if self.reservoir_size <= DENSE_SVD_LIMIT:
            return float(np.linalg.svd(
                self.W_res.toarray(), compute_uv=False)[0])
        return float(sparse_norm(self.W_res, ord=2))

    def _enforce_esp_condition(self):
        """
        Sufficient condition for ESP (contraction map):
            ||dF/dx|| <= (1 - alpha) + alpha * ||W_res||_2 < 1
        This bound is valid because tanh has Lipschitz constant 1.
        Scales W_res until the condition holds.
        """
        current_norm = self._spectral_norm()

        # Target: (1 - alpha) + alpha * target_norm <= 0.99
        #         (0.99 = safety margin for the strict inequality)
        # => target_norm <= (alpha - 0.01) / alpha
        target_norm = (self.alpha - 0.01) / self.alpha

        if target_norm <= 0:
            raise ValueError(
                f"Leakage rate (alpha={self.alpha}) too small to "
                "guarantee ESP with margin.")

        if current_norm > target_norm:
            if current_norm == 0.0:
                raise ValueError("W_res is degenerate (zero norm).")
            scaling_factor = target_norm / current_norm
            self.W_res = self.W_res * scaling_factor
            print(f"[INFO] W_res scaled by {scaling_factor:.4f} "
                  "to enforce ESP condition.")
            print(f"       New ||W_res||_2 = {self._spectral_norm():.4f}")
        else:
            print(f"[INFO] ESP condition naturally satisfied. "
                  f"||W_res||_2 = {current_norm:.4f}")

    def step(self, u: np.ndarray) -> np.ndarray:
        """
        Advances the reservoir state by one time step.
        x_t = (1 - alpha) * x_{t-1} + alpha * tanh(W_res @ x_{t-1} + W_in @ u_t)

        :param u: input vector (input_dim,)
        :return: updated reservoir state (reservoir_size,)
        """
        u = np.asarray(u, dtype=float)
        if u.shape[0] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.input_dim}, "
                f"got {u.shape[0]}")

        activated = np.tanh(self.W_res.dot(self.state) + self.W_in.dot(u))
        self.state = (1.0 - self.alpha) * self.state + self.alpha * activated

        return self.state.copy()

    def get_state(self) -> np.ndarray:
        """Returns the current reservoir state."""
        return self.state.copy()

    def reset(self):
        """Resets the reservoir state to zero."""
        self.state = np.zeros(self.reservoir_size)

    def validate_esp_empirical(self, num_steps: int = 500,
                               num_trials: int = 5,
                               tol: float = 1e-4,
                               seed: int = 99) -> bool:
        """
        Empirical ESP validation: starts num_trials pairs of different
        initial states, feeds the same input sequence, and checks whether
        the states converge. Does NOT touch the engine's own state or the
        global RNG; works entirely with local variables.

        :return: True if max distance < tol across all trials
        """
        rng = np.random.default_rng(seed)
        inputs = rng.uniform(-1.0, 1.0, (num_steps, self.input_dim))

        max_dist = 0.0
        for _ in range(num_trials):
            state1 = rng.uniform(-1.0, 1.0, self.reservoir_size)
            state2 = rng.uniform(-1.0, 1.0, self.reservoir_size)

            for u in inputs:
                state1 = ((1.0 - self.alpha) * state1
                          + self.alpha * np.tanh(
                              self.W_res.dot(state1) + self.W_in.dot(u)))
                state2 = ((1.0 - self.alpha) * state2
                          + self.alpha * np.tanh(
                              self.W_res.dot(state2) + self.W_in.dot(u)))

            max_dist = max(max_dist, float(np.linalg.norm(state1 - state2)))

        print(f"[VALIDATION] Empirical ESP max distance over {num_trials} "
              f"trials after {num_steps} steps: {max_dist:.6e}")
        return max_dist < tol


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



@njit(cache=True)
def _nb_run_ctxmem_v3(tokens, W_in, Wd, Wi, Wp, alpha, knots,
                      degree, K, W, Pb, Pbias, lam, trace_bound,
                      bias_bound, W_slow, E, beta, gamma, eta,
                      a_ema, a_fast, cons_every, e_gate, tau,
                      p_inflate, cons_counts, eval_every, vocab,
                      sat_mode):
    """_nb_run_ctxmem + sat_mode (0=refuse, 1=overwrite) + logs.

    cons_log: (step, target, mode) per consolidation ATTEMPT:
      mode 0 = REINFORCE write (E_k < tau*m2)
      mode 1 = NEW-CONTEXT write to emptiest slot
      mode 3 = OVERWRITE evict least-consolidated (sat_mode=1)
      mode -1 = REFUSED (sat_mode=0, saturated)
    gate_log: per eval boundary -> argmax slot, E vector snapshot.
    """
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
    gate_am = np.empty(n_eval, np.int64)
    gate_E = np.empty((n_eval, S))
    max_att = n // cons_every + 2
    cl_step = np.empty(max_att, np.int64)
    cl_tgt = np.empty(max_att, np.int64)
    cl_mode = np.empty(max_att, np.int64)
    nll, cnt, nci = 0.0, 0, 0
    e_fast, m2 = 0.0, 1.0
    a_m2 = 0.02
    step_c, n_cons, n_cl = 0, 0, 0
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
        for s_ in range(S):
            acc = 0.0
            for m in range(C):
                acc += (beta * ss[s_, m]) ** 2
            acc += (1.0 - beta * ss[s_, tgt]) ** 2 - (
                beta * ss[s_, tgt]) ** 2
            E[s_] = (1.0 - a_ema) * E[s_] + a_ema * acc
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
            target, mode = -1, -1
            if E[k] < tau * m2:
                target, mode = k, 0          # REINFORCE
            else:
                cmin = cons_counts[0]
                for s_ in range(1, S):
                    if cons_counts[s_] < cmin:
                        cmin = cons_counts[s_]
                if cmin > 0:
                    if sat_mode == 1:
                        target = 0           # OVERWRITE: evict
                        for s_ in range(1, S):
                            if cons_counts[s_] < cons_counts[target]:
                                target = s_
                        mode = 3
                    # else refuse: target=-1, mode=-1
                else:
                    target = 0               # NEW-CONTEXT
                    for s_ in range(1, S):
                        if cons_counts[s_] < cons_counts[target]:
                            target = s_
                    mode = 1
            cl_step[n_cl] = step_c
            cl_tgt[n_cl] = target
            cl_mode[n_cl] = mode
            n_cl += 1
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
            am = 0
            for s_ in range(1, S):
                if w[s_] > w[am]:
                    am = s_
            gate_am[nci] = am
            for s_ in range(S):
                gate_E[nci, s_] = E[s_]
            nci += 1
            nll, cnt = 0.0, 0
    return (curve[:nci], steps[:nci], n_cons,
            cl_step[:n_cl], cl_tgt[:n_cl], cl_mode[:n_cl],
            gate_am[:nci], gate_E[:nci])

def run_apia_v3(stream, vocab, arm, seed, n_res=512, n_basis=10,
                eval_every=2000, cons_every=5000, verbose=True):
    """v3: ctxmem with explicit saturation policy + cons/gate logs.

    arm='apia_ctxmem_refuse'    -> sat_mode 0 (write-protected)
    arm='apia_ctxmem_overwrite' -> sat_mode 1 (evict least-consolidated)
    arm='apia_block'            -> delegates to _nb_run_block.
    cons_every defaults 5000 so slot writes use a MATURE W_fast
    (v2's 250 wrote immature weights within the first ~750 steps).
    """
    if not _HAVE_NUMBA:
        raise RuntimeError("numba unavailable")
    sub, rls, a = _apia_init(vocab, n_res, n_basis, seed)
    tok = np.ascontiguousarray(stream.astype(np.int64))
    t0 = time.perf_counter()
    cons_log, gate_log, gate_E = None, None, None
    if arm == "apia_block":
        curve, steps = _nb_run_block(
            tok, a["W_in"], a["Wd"], a["Wi"], a["Wp"], a["alpha"],
            a["knots"], a["degree"], a["K"], a["W"], a["Pb"],
            a["Pbias"], a["lam"], a["trace_bound"], a["bias_bound"],
            eval_every, vocab)
        n_cons = 0
    elif arm in ("apia_ctxmem_refuse", "apia_ctxmem_overwrite"):
        sat_mode = 0 if arm.endswith("refuse") else 1
        S = 3
        W_slow = np.zeros((S,) + a["W"].shape)
        E = np.zeros(S)
        cons_counts = np.zeros(S, np.int64)
        (curve, steps, n_cons, cl_step, cl_tgt, cl_mode,
         gate_am, gate_E) = _nb_run_ctxmem_v3(
            tok, a["W_in"], a["Wd"], a["Wi"], a["Wp"], a["alpha"],
            a["knots"], a["degree"], a["K"], a["W"], a["Pb"],
            a["Pbias"], a["lam"], a["trace_bound"], a["bias_bound"],
            W_slow, E, 1.0, 0.1, 5.0, 0.05, 0.1, cons_every, 1.0,
            0.5, 1.5, cons_counts, eval_every, vocab, sat_mode)
        a["W_slow"] = W_slow
        cons_log = dict(step=[int(v) for v in cl_step],
                        target=[int(v) for v in cl_tgt],
                        mode=[int(v) for v in cl_mode],
                        slot_counts=[int(v) for v in cons_counts])
        gate_log = [int(v) for v in gate_am]
        gate_E = gate_E.tolist()
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
        peak_vram_mb=0, n_consolidations=int(n_cons),
        cons_every=cons_every, cons_log=cons_log,
        gate_argmax=gate_log, gate_E=gate_E)

def make_shift_stream(stream, vocab, a_len=100000, b_len=50000,
                      shift=1):
    """A -> B(shifted) -> A' stream. B is the SAME source chars as the
    A'-window predecessor segment, Caesar-shifted by `shift` mod vocab:
    pure symbol relabeling, identical bigram structure -> clean new
    context that the gate can in principle re-detect at return."""
    A = stream[:a_len]
    B = (stream[a_len:a_len + b_len] + shift) % vocab
    Ap = stream[a_len + b_len:2 * a_len]
    return np.concatenate([A, B, Ap]), a_len, b_len

def phase_metrics(run, a_len=100000, b_len=50000, onset=10000):
    """Onset/end BPC per phase. Onset = first `onset` steps of phase;
    end = last `onset` of phase. Steps are 1-based stream positions."""
    steps = np.asarray(run["curve_steps"])
    curve = np.asarray(run["bpc_curve"])
    bounds = [0, a_len, a_len + b_len, len(steps) * (steps[-1] // len(steps))]
    total = steps[-1]
    bounds[3] = total
    out = {}
    names = ["A", "B", "A'"]
    for i, nm in enumerate(names):
        lo, hi = bounds[i], bounds[i + 1]
        on = curve[(steps > lo) & (steps <= lo + onset)]
        en = curve[(steps > hi - onset) & (steps <= hi)]
        out[f"onset_{nm}"] = float(on.mean()) if len(on) else None
        out[f"end_{nm}"] = float(en.mean()) if len(en) else None
    return out


# chunked + throttled runners: same math as _nb_run_block/_nb_run_ctxmem_v3,
# split into 2000-token njit calls with h/state carried across calls.
# Sleep between chunks keeps sustained CPU duty below the Kaggle watchdog.

@njit
def _nb_block_c(tokens, W_in, Wd, Wi, Wp, alpha, knots, degree, K,
                W, Pb, Pbias, lam, trace_bound, bias_bound, h, vocab):
    D = W_in.shape[0]
    F = D * K + 1
    C = vocab
    n = tokens.shape[0] - 1
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
    nll, cnt = 0.0, 0
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
    return nll, cnt


@njit
def _nb_ctxmem_c(tokens, W_in, Wd, Wi, Wp, alpha, knots, degree, K,
                 W, Pb, Pbias, lam, trace_bound, bias_bound,
                 W_slow, E, beta, gamma, eta, a_ema, a_fast,
                 cons_every, e_gate, tau, p_inflate, cons_counts,
                 st, cl_step, cl_tgt, cl_mode, n_cl, h, vocab, sat_mode,
                 force_slot, hard_gate, force_steps, force_mode):
    """Chunked _nb_run_ctxmem_v3. st=[e_fast,m2,step_c] persisted.
    Returns (nll, cnt, gate_argmax); cons attempts appended to cl_*/n_cl."""
    D = W_in.shape[0]
    F = D * K + 1
    C = vocab
    S = W_slow.shape[0]
    n = tokens.shape[0] - 1
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
    se_buf = np.empty(S)
    nll, cnt = 0.0, 0
    e_fast, m2, step_c = st[0], st[1], st[2]
    nf, ns_, cooldown = st[3], st[4], st[5]
    a_m2 = 0.02
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
        for s_ in range(S):
            mx_s = ss[s_, 0]
            for m in range(1, C):
                if ss[s_, m] > mx_s:
                    mx_s = ss[s_, m]
            se_s = 0.0
            for m in range(C):
                se_s += np.exp(ss[s_, m] - mx_s)
            acc = mx_s - ss[s_, tgt] + np.log(se_s)
            se_buf[s_] = acc
        # CAUSAL ORDER: the gate must see E built from history only;
        # the current target's per-slot NLL (se_buf) is folded into E
        # after the prediction is scored, not before the gate.
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
        if force_slot >= 0:
            wsum = 0.0
            for s_ in range(S):
                w[s_] = 1.0 if s_ == force_slot else 0.0
                wsum += w[s_]
        elif hard_gate:
            hk = 0
            for s_ in range(1, S):
                if E[s_] < E[hk]:
                    hk = s_
            wsum = 0.0
            for s_ in range(S):
                w[s_] = 1.0 if s_ == hk else 0.0
                wsum += w[s_]
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
        for s_ in range(S):
            E[s_] = (1.0 - a_ema) * E[s_] + a_ema * se_buf[s_]
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
        nll_i = -np.log(np.exp(scores[tgt] - mx) / se) / ln2
        nf = (1.0 - 0.05) * nf + 0.05 * nll_i
        ns_ = (1.0 - 0.002) * ns_ + 0.002 * nll_i
        if cooldown > 0.0:
            cooldown -= 1.0
        step_c += 1.0
        sc = int(step_c)
        forced = False
        if force_mode == 1:
            for fs in range(force_steps.shape[0]):
                if force_steps[fs] == sc:
                    forced = True
        elif (force_mode == 2 and cooldown <= 0.0 and sc > 2000
                and nf > ns_ * 1.02):
            forced = True
            cooldown = cons_every
        if (forced or (not force_mode and sc % cons_every == 0
                       and e_fast <= e_gate * m2)):
            k = 0
            for s_ in range(1, S):
                if E[s_] < E[k]:
                    k = s_
            target, mode = -1, -1
            if not forced and E[k] < tau * m2:
                target, mode = k, 0
            else:
                cmin = cons_counts[0]
                for s_ in range(1, S):
                    if cons_counts[s_] < cmin:
                        cmin = cons_counts[s_]
                if cmin > 0:
                    if sat_mode == 1:
                        target = 0
                        for s_ in range(1, S):
                            if cons_counts[s_] < cons_counts[target]:
                                target = s_
                        mode = 3
                else:
                    target = 0
                    for s_ in range(1, S):
                        if cons_counts[s_] < cons_counts[target]:
                            target = s_
                    mode = 1
            ni = int(n_cl[0])
            cl_step[ni] = int(step_c)
            cl_tgt[ni] = target
            cl_mode[ni] = mode
            n_cl[0] += 1
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
    st[0], st[1], st[2] = e_fast, m2, step_c
    st[3], st[4], st[5] = nf, ns_, cooldown
    am = 0
    for s_ in range(1, S):
        if w[s_] > w[am]:
            am = s_
    return nll, cnt, am


_FSTEP_PAD = np.full(8, -1, np.int64)
def run_arm_throttled(arm, stream, vocab, seed=0, cons_every=5000,
                      chunk=400, work_s=0.9, sleep_s=0.4, seg_bounds=None):
    """Duty-cycled arm runner: ~0.2s CPU quanta + 0.4s sleep per chunk.
    Yields one eval point per chunk (eval_every=chunk)."""
    _s_, _r_, a = _apia_init(vocab, 512, 10, seed)
    h = np.zeros(512)
    curve, steps = [], []
    tok_full = np.ascontiguousarray(stream.astype(np.int64))
    S = 3
    W_slow = np.zeros((S,) + a['W'].shape)
    E = np.zeros(S)
    cons_counts = np.zeros(S, np.int64)
    st = np.zeros(6)
    cl_step = np.empty(512, np.int64)
    cl_tgt = np.empty(512, np.int64)
    cl_mode = np.empty(512, np.int64)
    n_cl = np.zeros(1, np.int64)
    gate_am, gate_E = [], []
    t0 = time.perf_counter()
    pos, n_done = 0, len(tok_full) - 1
    while pos < n_done:
        seg = tok_full[pos:pos + chunk + 1]
        if len(seg) < 2:
            break
        if arm == 'apia_block':
            nll, cnt = _nb_block_c(seg, a['W_in'], a['Wd'], a['Wi'], a['Wp'],
                                   a['alpha'], a['knots'], a['degree'], a['K'],
                                   a['W'], a['Pb'], a['Pbias'], a['lam'],
                                   a['trace_bound'], a['bias_bound'], h, vocab)
        else:
            sat = 0 if not arm.endswith('overwrite') else 1
            fslot = -1
            hard = 0
            if arm == 'apia_ctxmem_oracle' and seg_bounds is not None:
                seg_i = 0
                for i in range(len(seg_bounds) - 1):
                    if seg_bounds[i] <= pos < seg_bounds[i + 1]:
                        seg_i = i
                        break
                fslot = 0 if seg_i % 2 == 0 else 1
            elif arm == 'apia_ctxmem_hardgate':
                hard = 1
            fmode = 0
            fsteps = _FSTEP_PAD
            if arm == 'apia_ctxmem_forced':
                hard = 1
                fmode = 1
            elif arm == 'apia_ctxmem_detect':
                hard = 1
                fmode = 2
                fsteps = np.asarray(
                    [b for b in (seg_bounds or [])[1:4]] + [-1] * 8,
                    dtype=np.int64)[:8]
            nll, cnt, am = _nb_ctxmem_c(
                seg, a['W_in'], a['Wd'], a['Wi'], a['Wp'], a['alpha'],
                a['knots'], a['degree'], a['K'], a['W'], a['Pb'], a['Pbias'],
                a['lam'], a['trace_bound'], a['bias_bound'], W_slow, E,
                1.0, 0.1, 5.0, 0.05, 0.1, cons_every, 1.0, 0.5, 1.5,
                cons_counts, st, cl_step, cl_tgt, cl_mode, n_cl, h, vocab,
                sat, fslot, hard, fsteps, fmode)
            gate_am.append(int(am))
            gate_E.append([float(v) for v in E])
        curve.append(float(nll / max(cnt, 1)))
        steps.append(int(pos + len(seg) - 1))
        pos += len(seg) - 1
        if len(steps) % 10 == 0:
            print('   t=%d bpc=%.3f' % (steps[-1], curve[-1]), flush=True)
        time.sleep(sleep_s)
    n_cons = int(cons_counts.sum())
    cons_log = dict(step=[int(v) for v in cl_step[:n_cl[0]]],
                    target=[int(v) for v in cl_tgt[:n_cl[0]]],
                    mode=[int(v) for v in cl_mode[:n_cl[0]]],
                    slot_counts=[int(v) for v in cons_counts])
    wall = time.perf_counter() - t0
    r = dict(arm=arm, seed=seed, n_params=512 * 10 * vocab,
             final_bpc=float(np.mean(curve[-3:])) if curve else None,
             bpc_curve=[float(v) for v in curve], curve_steps=steps,
             tok_per_s=round(n_done / wall, 1), wall_s=round(wall, 1),
             n_consolidations=n_cons, cons_log=cons_log,
             gate_log=gate_am, gate_E=gate_E, n_res=512, n_basis=10,
             hard_gate=arm in ('apia_ctxmem_hardgate', 'apia_ctxmem_forced',
                        'apia_ctxmem_detect'))
    return r


# ---------- causal-ordering re-verification grid: A->B->A->B->A ----------
OUT3 = '/kaggle/working/native_lm_v3_causal.json'
res3 = json.load(open(OUT3)) if os.path.exists(OUT3) else {}
res3.setdefault('runs', [])
stream, vocab = load_chars()
stream = stream[:250001]
sh = np.concatenate([np.ascontiguousarray(stream[0:50000]),
                     np.ascontiguousarray(stream[50000:100000][::-1]),
                     np.ascontiguousarray(stream[100000:150000]),
                     np.ascontiguousarray(stream[150000:200000][::-1]),
                     np.ascontiguousarray(stream[200000:250000])])
bounds = [0, 50000, 100000, 150000, 200000, 250000]
labels = ['A1', 'B1', 'A2', 'B2', 'A3']
print('v3-CAUSAL-250K stream:', len(sh), 'bounds', bounds, flush=True)


def alt_phase_metrics(r, bounds, labels, win=2000):
    out = {}
    steps, curve = r['curve_steps'], r['bpc_curve']
    for i in range(len(labels)):
        lo, hi = bounds[i], bounds[i + 1]
        on = [c for s, c in zip(steps, curve) if lo < s <= lo + win]
        en = [c for s, c in zip(steps, curve) if hi - win < s <= hi]
        out['onset_' + labels[i]] = float(np.mean(on)) if on else None
        out['end_' + labels[i]] = float(np.mean(en)) if en else None
    return out


print('warming numba...', flush=True)
_wa = _apia_init(vocab, 512, 10, 0)[2]
_wh = np.zeros(512)
_wt = np.ascontiguousarray(np.arange(8, dtype=np.int64) % vocab)
_nb_block_c(_wt, _wa['W_in'], _wa['Wd'], _wa['Wi'], _wa['Wp'], _wa['alpha'],
            _wa['knots'], _wa['degree'], _wa['K'], _wa['W'], _wa['Pb'],
            _wa['Pbias'], _wa['lam'], _wa['trace_bound'], _wa['bias_bound'],
            _wh, vocab)
time.sleep(0.5)
_wS = np.zeros((3,) + _wa['W'].shape)
_wE = np.zeros(3)
_wcc = np.zeros(3, np.int64)
_wst = np.zeros(6)
_wcl = np.empty(8, np.int64)
_wcl2 = np.empty(8, np.int64)
_wcl3 = np.empty(8, np.int64)
_wncl = np.zeros(1, np.int64)
_nb_ctxmem_c(_wt, _wa['W_in'], _wa['Wd'], _wa['Wi'], _wa['Wp'], _wa['alpha'],
             _wa['knots'], _wa['degree'], _wa['K'], _wa['W'], _wa['Pb'],
             _wa['Pbias'], _wa['lam'], _wa['trace_bound'], _wa['bias_bound'],
             _wS, _wE, 1.0, 0.1, 5.0, 0.05, 0.1, 5000, 1.0, 0.5, 1.5,
             _wcc, _wst, _wcl, _wcl2, _wcl3, _wncl, _wh, vocab, 0, -1, 0, _FSTEP_PAD, 0)
_nb_ctxmem_c(_wt, _wa['W_in'], _wa['Wd'], _wa['Wi'], _wa['Wp'], _wa['alpha'],
             _wa['knots'], _wa['degree'], _wa['K'], _wa['W'], _wa['Pb'],
             _wa['Pbias'], _wa['lam'], _wa['trace_bound'], _wa['bias_bound'],
             _wS, _wE, 1.0, 0.1, 5.0, 0.05, 0.1, 5000, 1.0, 0.5, 1.5,
             _wcc, _wst, _wcl, _wcl2, _wcl3, _wncl, _wh, vocab, 0, 0, 1, _FSTEP_PAD, 0)
print('numba warm', flush=True)
for arm in ['apia_block', 'apia_ctxmem', 'apia_ctxmem_hardgate',
            'apia_ctxmem_oracle', 'apia_ctxmem_forced',
            'apia_ctxmem_detect']:
    for seed in [0, 1, 2]:
        if any(r['arm'] == arm and r['seed'] == seed for r in res3['runs']):
            print('skip', arm, seed, flush=True)
            continue
        print('==', arm, 'seed', seed, flush=True)
        r = run_arm_throttled(arm, sh, vocab, seed=seed, cons_every=5000,
                              seg_bounds=bounds)
        r['phases'] = alt_phase_metrics(r, bounds, labels)
        res3['runs'].append(r)
        json.dump(res3, open(OUT3, 'w'), indent=1)
        print('__V3ARM__' + json.dumps(r), flush=True)
        pm = r['phases']
        print('   -> ' + '  '.join('%s.on=%.3f %s.en=%.3f' % (
            l, pm['onset_' + l] or -1, l, pm['end_' + l] or -1)
            for l in labels) + '  cons=%d' % r.get('n_consolidations', 0),
            flush=True)
print('CAUSAL_DONE ->', OUT3, flush=True)
