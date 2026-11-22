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
