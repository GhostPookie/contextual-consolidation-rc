"""
APIA 2.0 Research Prototype
Module: reservoir.py
Purpose: manage Echo State Network (ESN) reservoir dynamics with a
         strict mathematical ESP (Echo State Property) guarantee.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm as sparse_norm

# Dimension limit below which the exact spectral norm is computed via
# dense SVD; above it, iterative svds is used (approximate result).
DENSE_SVD_LIMIT = 2048


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


# ==========================================
# TEST BLOCK (verifies module correctness)
# ==========================================
if __name__ == "__main__":
    print("=== APIA 2.0 Reservoir Engine Test ===")

    INPUT_DIM = 5
    RES_SIZE = 100
    ALPHA = 0.3

    reservoir = ReservoirEngine(
        input_dim=INPUT_DIM,
        reservoir_size=RES_SIZE,
        sparsity=0.1,
        leakage_rate=ALPHA,
        input_scaling=0.8,
        seed=42)

    # 0. Sparsity check (verifies the unique-index fix)
    actual_nnz = reservoir.W_res.nnz
    expected_nnz = int(RES_SIZE * RES_SIZE * 0.1)
    w_vals = reservoir.W_res.data
    print(f"\n[CHECK] nnz = {actual_nnz} (expected {expected_nnz})")
    print(f"[CHECK] |W_res| max = {np.abs(w_vals).max():.4f} "
          "(must be <= 1.0; duplicate summation would exceed this)")

    # 1. Theoretical ESP check
    norm_val = reservoir._spectral_norm()
    esp_check = (1 - ALPHA) + ALPHA * norm_val
    print(f"\n[CHECK] Theoretical ESP: (1-a) + a*||W_res||_2 = "
          f"{esp_check:.4f} < 1.0 -> "
          f"{'PASS' if esp_check < 1.0 else 'FAIL'}")

    # 2. Empirical ESP check (5 trials, max distance)
    is_esp_valid = reservoir.validate_esp_empirical(num_steps=500,
                                                  num_trials=5)
    print(f"[CHECK] Empirical ESP Convergence: "
          f"{'PASS' if is_esp_valid else 'FAIL'}")

    # 3. Forward-step test
    test_input = np.random.default_rng(0).uniform(-1, 1, INPUT_DIM)
    next_state = reservoir.step(test_input)
    print(f"\n[CHECK] Forward pass ok. State shape: {next_state.shape}")
    print(f"        State min/max: [{next_state.min():.4f}, "
          f"{next_state.max():.4f}] (expected: (-1, 1))")
