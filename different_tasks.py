"""
APIA 2.0 Research Prototype
Module: different_tasks.py
Purpose: generalization test of the contextual memory on DIFFERENT
dynamics.

Two experiments (5 seeds, 1500 steps per phase + 500 washout):
  TEST A "MG regimes":   Mackey-Glass tau=17 -> tau=30 -> tau=17
                          (same system family, different time dynamics)
  TEST B "mixed dynamics": NARMA-10 -> MG-17 -> NARMA-10
                          (externally driven nonlinear memory task
                           <-> autonomous chaotic system)

Each phase uses its own generator; (u_t, y_t) pairs are produced.
Methods: block (memoryless), blockmem (naive, ablation), ctxmem.
Metric: F3-onset / F3-end NMSE (relative to whole-stream variance).
"""

import numpy as np

from reservoir import ReservoirEngine
from rls_engine import SplineFeatureMap, BlockDiagRLS, mackey_glass
from memory import DualMemory
from contextual_memory import ContextualMemory
from multihead import MultiHeadMemory

SEEDS = 5
D, K = 100, 10
WASHOUT = 500
PHASE_LEN = 1500
LAM = 0.98
INIT_VAR = 1e2
WINDUP_MULT = 1e4
WINDOW = 250
TOTAL = WASHOUT + 3 * PHASE_LEN


# ---------------- Phase generators: (u, y) -> (PHASE_LEN+1,) ----------------
def gen_mg(n, seed, tau):
    x = mackey_glass(n + 1, tau=tau, seed=seed)
    return x[:-1], x[1:]          # input x_t, target x_{t+1}


def gen_wave(n, seed, kind):
    t = np.arange(n + 1)
    s = np.sin(2 * np.pi * t / 80.0)
    if kind == "sine":
        w = s
    elif kind == "square":
        w = np.sign(s)
    elif kind == "triangle":
        w = 2 * np.abs(2 * (t / 80.0 - np.floor(t / 80.0 + 0.5))) - 1
    return w[:-1], w[1:]


def gen_narma(n, seed, order):
    """NARMA-<order>: y(t+1)=0.3y(t)+0.05y(t)sum_{i<order}y(t-i)
       +1.5 u(t-order+1)u(t)+0.1 ; u~U[0,0.5]."""
    rng = np.random.default_rng(seed)
    lag = order - 1
    u = rng.uniform(0.0, 0.5, n + 1 + lag)
    y = np.zeros(n + 1 + lag)
    for t in range(lag, n + lag):
        y[t + 1] = (0.3 * y[t]
                    + 0.05 * y[t] * np.sum(y[t - lag:t + 1])
                    + 1.5 * u[t - lag] * u[t] + 0.1)
    return u[lag:lag + n], y[lag + 1:lag + n + 1]  # (u_t, y_{t+1})


GENS = {
    "mg": gen_mg,
    "sine": gen_wave, "square": gen_wave, "triangle": gen_wave,
    "narma": gen_narma,
}


def build_stream(spec, seed):
    """spec: [(gen_name, arg), ...] -> unified (u_all, y_all)."""
    u_parts, y_parts = [], []
    for name, arg in spec:
        u, y = GENS[name](PHASE_LEN, seed, arg)
        u_parts.append(u)
        y_parts.append(y)
    u_all = np.concatenate(u_parts)
    y_all = np.concatenate(y_parts)
    # washout: act as the continuation of the phase-0 generator
    # (instead of a plain sine)
    u0, y0 = GENS[spec[0][0]](WASHOUT, seed + 777, spec[0][1])
    u_all = np.concatenate([u0, u_all])
    y_all = np.concatenate([y0, y_all])
    mu, sd = u_all.mean(), u_all.std()
    u_all = (u_all - mu) / sd
    return u_all, y_all


def run_once(kind, Phi, Y, sig_var):
    F = D * K + 1
    if kind == 'block':
        m = BlockDiagRLS(D, K, 1, LAM, INIT_VAR, WINDUP_MULT)
        upd = m.update
    elif kind == 'blockmem':
        base = BlockDiagRLS(D, K, 1, LAM, INIT_VAR, WINDUP_MULT)
        m = DualMemory(base, beta=1.0, gamma=0.1,
                       consolidate_every=250, p_inflate=1.5)
        upd = m.update
    elif kind == 'multihead':
        m = MultiHeadMemory(
            lambda: BlockDiagRLS(D, K, 1, LAM, INIT_VAR, WINDUP_MULT),
            n_slots=3)
        upd = m.update
    else:
        base = BlockDiagRLS(D, K, 1, LAM, INIT_VAR, WINDUP_MULT)
        m = ContextualMemory(base, n_slots=3, beta=1.0, gamma=0.1,
                             eta=2.0, consolidate_every=250,
                             e_gate=0.5,
                             p_inflate=1.5)
        upd = m.update
    errs = np.empty(len(Phi))
    for i in range(len(Phi)):
        e = upd(Phi[i], Y[i])
        errs[i] = float(np.sum(e ** 2))
    return errs, m


def wnmse(errs, var, lo, hi):
    return float(np.mean(errs[lo:hi]) / var)


def run_test(name, spec):
    print(f"\n{'#'*78}\nTEST {name}: {' -> '.join(s[0]+str(s[1]) for s in spec)}")
    print(f"{'#'*78}")
    methods = ['block', 'blockmem', 'multihead', 'ctxmem']
    labels = {'block': 'BlockRLS', 'blockmem': '+naive',
              'multihead': 'MultiHead', 'ctxmem': '+CTX'}
    met = {m: {p: {'onset': [], 'end': []} for p in range(3)} for m in methods}
    pats = []

    for seed in range(SEEDS):
        u_all, y_all = build_stream(spec, seed)
        sig_var = float(np.var(y_all))
        res = ReservoirEngine(1, D, 0.1, 0.3, 0.8, seed=seed)
        fmap = SplineFeatureMap(D, K, 3)
        Phi = np.empty((TOTAL - WASHOUT, fmap.n_features))
        for t in range(TOTAL):
            x = res.step(np.array([u_all[t]]))
            if t >= WASHOUT:
                Phi[t - WASHOUT] = fmap.transform(x)
        Y = y_all[WASHOUT:].reshape(-1, 1)

        for mth in methods:
            errs, m = run_once(mth, Phi, Y, sig_var)
            for p in range(3):
                lo, hi = p * PHASE_LEN, (p + 1) * PHASE_LEN
                met[mth][p]['onset'].append(wnmse(errs, sig_var, lo, lo + WINDOW))
                met[mth][p]['end'].append(wnmse(errs, sig_var, hi - WINDOW, hi))
            if mth == 'ctxmem':
                log = np.array(m.slot_log)
                rep = [int(np.bincount(log[p*PHASE_LEN:(p+1)*PHASE_LEN],
                                       minlength=3).argmax()) for p in range(3)]
                pats.append(rep)
        b = met['block'][2]['onset'][-1]
        c = met['ctxmem'][2]['onset'][-1]
        print(f"  seed {seed}: block={b:.3e} ctx={c:.3e} | ctx slots={rep}")

    print(f"\n--- {name} summary (NMSE mean+/-std) ---")
    print(f"{'Method':<10} | {'F3 onset':<22} | {'F3 end':<22}")
    for mth in methods:
        o, e = met[mth][2]['onset'], met[mth][2]['end']
        print(f"{labels[mth]:<10} | {np.mean(o):.3e}+/-{np.std(o):.1e}"
              f"     | {np.mean(e):.3e}+/-{np.std(e):.1e}")
    base_o = np.mean(met['block'][2]['onset'])
    ctx_o = np.mean(met['ctxmem'][2]['onset'])
    print(f"-> CTX F3-onset advantage: x{base_o/max(ctx_o,1e-12):.2f}")
    print(f"-> ctx slot patterns: {pats}")


if __name__ == "__main__":
    run_test("A", [("mg", 17), ("mg", 30), ("mg", 17)])
    run_test("B", [("narma", 10), ("mg", 17), ("narma", 10)])
