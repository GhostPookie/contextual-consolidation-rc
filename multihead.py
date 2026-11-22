"""
APIA 2.0 Research Prototype
Module: multihead.py
Purpose: closest competitor baseline — Bereska et al. (PMLR 2022)-style
      competitive MULTI-READOUT-HEAD continual learning.

Mechanism (as in the literature):
  - S independent readout heads (each a full RLS engine).
  - Per step, an EMA prediction error is tracked for each head.
  - Winner-take-all: the head with the lowest EMA error predicts
    AND is the only one updated.
  - PASSIVE protection: losing heads are never updated -> no
    forgetting, but also no consolidation/transfer.

Key differences from CTX (Related-Work comparison):
  - Our system has a single shared W_fast + gated slow slots; here
    there are S independent weight sets (memory cost S*P: 3x more
    covariance memory).
  - No active write-gating; no continuity-preserving analytic
    transfer; no occupancy-oracle targeting.
  - The gate selection uses the same EMA logic as ours (fair
    comparison).
"""

import numpy as np


class MultiHeadMemory:
    """
    S independent RLS heads; argmin-EMA wins and is updated.

    :param model_factory: callable producing a fresh RLS engine
    :param n_slots: number of heads
    :param alpha_ema: head-error EMA coefficient (same as CTX: 0.01)
    """

    def __init__(self, model_factory, n_slots: int = 3,
                 alpha_ema: float = 0.01, alloc_thresh=None,
                 alpha_m2: float = 0.02, soft_predict: bool = False,
                 eta: float = 2.0):
        self.heads = [model_factory() for _ in range(n_slots)]
        self.n_slots = n_slots
        self.alpha_ema = alpha_ema
        # C-attribution variant: on the prediction side, CTX's
        # normalized soft gate (w ~ exp(-eta E/E_min)) instead of hard
        # argmin. The update rule remains winner-only, as in Bereska.
        self.soft_predict = soft_predict
        self.eta = eta
        # Channel-isolation ablation (multihead+alloc): if min E_i >
        # alloc_thresh*m2 and a never-updated head exists, write to
        # that head. Adds CTX's write-protection channel to the
        # competitor architecture — isolates the channel missing from
        # passive protection.
        self.alloc_thresh = alloc_thresh
        self.alpha_m2 = alpha_m2
        self._m2 = 1.0
        self.upd_counts = np.zeros(n_slots, dtype=int)
        self.E = np.zeros(n_slots)
        self.slot_log = []

    def update(self, phi, y):
        # EMA error of each head (before updating, causal)
        for i, h in enumerate(self.heads):
            e_i = y - h.predict(phi)
            self.E[i] = ((1 - self.alpha_ema) * self.E[i]
                         + self.alpha_ema * float(np.sum(e_i * e_i)))
        self._m2 = ((1 - self.alpha_m2) * self._m2
                    + self.alpha_m2 * float(np.sum(y * y)))
        k = int(np.argmin(self.E))
        if (self.alloc_thresh is not None
                and self.upd_counts.min() == 0
                and self.E[k] > self.alloc_thresh * self._m2):
            k = int(np.argmin(self.upd_counts))
        self.slot_log.append(k)
        self.upd_counts[k] += 1
        # The reported error must be the actual prediction error: in
        # soft_predict the mixture (pre-update, causal), in hard mode
        # the winner's.
        e_rep = (y - self.predict(phi)) if self.soft_predict else None
        ret = self.heads[k].update(phi, y)
        return e_rep if e_rep is not None else ret

    def predict(self, phi):
        if not self.soft_predict:
            return self.heads[int(np.argmin(self.E))].predict(phi)
        e_min = max(self.E.min(), 1e-8)
        z = -self.eta * self.E / e_min
        z -= z.max()
        w = np.exp(z)
        w /= w.sum()
        preds = np.stack([h.predict(phi) for h in self.heads])
        return w @ preds
