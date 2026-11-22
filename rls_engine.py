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
