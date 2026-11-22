"""
APIA 2.0 Research Prototype
Module: memory.py
Purpose: dual-rate memory mechanism (W_fast / W_slow) — analytic and
      DISCONTINUITY-FREE consolidation.

Prediction model:
      y_hat = phi' @ W_fast + beta * phi' @ W_slow

Consolidation rule (continuity guaranteed, y_hat unchanged):
      W_slow <- W_slow + (1-gamma)/beta * W_fast
      W_fast <- gamma * W_fast

  Check:   gW_f + b(W_s + (1-g)/b W_f) = gW_f + bW_s + (1-g)W_f
          = W_f + bW_s  (same as before)  => NO jump discontinuity.

Double-counting prevention:
      The RESIDUAL is used as the W_fast training target:
          e = y - (phi'W_fast + beta*phi'W_slow)
      so W_fast does not relearn the signal W_slow already explains.
      Implementation: the model is fed the target (y - beta*phi'W_slow);
      the engine's internal error already equals the total prediction
      error.

Windup interaction:
      This layer requires trace-bounding together with lambda<1
      (rls_engine windup_mult). In addition, P is slightly inflated at
      consolidation to compensate for the damped W_fast (plasticity
      is reopened).
"""

import numpy as np


class DualMemory:
    """
    Wraps a dual-rate memory around BlockDiagRLS (or any engine with a
    get/set_weights + inflate_P interface).
    """

    def __init__(self, model, beta: float = 1.0, gamma: float = 0.1,
                 consolidate_every: int = 250, p_inflate: float = 1.5):
        """
        :param model: RLS engine with get_weights/set_weights/inflate_P
        :param beta: gain of W_slow's contribution to prediction
        :param gamma: fraction of W_fast retained at consolidation (0<g<=1)
        :param consolidate_every: steps between consolidations
        :param p_inflate: P inflation factor at consolidation (1.0 = off)
        """
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0,1], got {gamma}")
        if beta == 0.0:
            raise ValueError("beta must be nonzero")
        self.model = model
        self.beta = beta
        self.gamma = gamma
        self.consolidate_every = consolidate_every
        self.p_inflate = p_inflate

        self.W_slow = np.zeros_like(model.get_weights())
        self._steps = 0
        self.n_consolidations = 0
        # For validation: prediction-continuity deviation at consolidation
        self.max_discontinuity = 0.0

    def _slow_part(self, phi: np.ndarray) -> np.ndarray:
        return self.beta * (phi @ self.W_slow)

    def predict(self, phi: np.ndarray) -> np.ndarray:
        """Total prediction: fast + beta*slow component."""
        return self.model.predict(phi) + self._slow_part(phi)

    def update(self, phi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Online update on the residual target.
        The returned error is the TOTAL prediction error (fast+slow),
        usable directly for evaluation.
        """
        residual_target = y - self._slow_part(phi)
        e_total = self.model.update(phi, residual_target)

        self._steps += 1
        if self._steps % self.consolidate_every == 0:
            self.consolidate(phi)
        return e_total

    def consolidate(self, phi_probe: np.ndarray = None):
        """
        W_slow += (1-gamma)/beta * W_fast ;  W_fast *= gamma ; inflate P.
        If phi_probe is given, the continuity deviation is measured
        (for testing).
        """
        W_fast = self.model.get_weights()

        if phi_probe is not None:
            before = (phi_probe @ W_fast
                      + self.beta * (phi_probe @ self.W_slow))

        self.W_slow += (1.0 - self.gamma) / self.beta * W_fast
        self.model.set_weights(self.gamma * W_fast)

        if self.p_inflate != 1.0:
            self.model.inflate_P(self.p_inflate)

        if phi_probe is not None:
            after = (phi_probe @ self.model.get_weights()
                     + self.beta * (phi_probe @ self.W_slow))
            d = float(np.max(np.abs(before - after)))
            self.max_discontinuity = max(self.max_discontinuity, d)

        self.n_consolidations += 1
