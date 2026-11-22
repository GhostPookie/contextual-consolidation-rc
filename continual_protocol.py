"""
APIA 2.0 Research Prototype
Module: continual_protocol.py
Purpose: test of the "no-forgetting" claim — sine -> square -> sine
regime change.

Comparison (all with lambda=0.98, P-bounding active via
windup_mult=1e4):
  (a) FullRLS+bound : does P-bounding rescue full RLS? (ablation)
  (b) BlockRLS+bound: memoryless reference — does it relearn sine
      from scratch on return?
  (c) BlockRLS+mem  : DualMemory (gamma=0.1, consolidation every 250 steps)

Primary metric: phase-3 (sine return) "onset" NMSE (first 250 steps).
  If memory helps, (c)'s onset error should be clearly lower than (b)'s.
Secondary: phase-end NMSEs + consolidation continuity deviation
  (should be ~0).
"""

import numpy as np

from reservoir import ReservoirEngine
from rls_engine import SplineFeatureMap, FullRLS, BlockDiagRLS
from memory import DualMemory
from contextual_memory import ContextualMemory
from multihead import MultiHeadMemory

# ---------------- Protocol constants ----------------
SEEDS = 10
D, K = 100, 10
WASHOUT = 500
PHASE_LEN = 1500
PHASES = ["sine", "square", "sine"]     # sine -> square -> sine
PERIOD = 80.0
LAM = 0.98
INIT_VAR = 1e2
WINDUP_MULT = 1e4
WINDOW = 250                          # onset / end measurement window

MEM_BETA = 1.0
MEM_GAMMA = 0.1
MEM_CONS_EVERY = 250
MEM_P_INFLATE = 1.5

TOTAL = WASHOUT + PHASE_LEN * len(PHASES)


def waveform(task: str, t: float) -> float:
    s = np.sin(2.0 * np.pi * t / PERIOD)
    if task == "square":
        return float(np.sign(s))
    if task == "triangle":
        return float(2.0 * np.abs(2.0 * ((t / PERIOD) % 1.0)) - 1.0)
    return s


def build_stream(seed, phases=None, phase_len=None):
    """Different waveform per phase -> u(t); target u(t+1)."""
    phases = PHASES if phases is None else phases
    phase_len = PHASE_LEN if phase_len is None else phase_len
    total = WASHOUT + phase_len * len(phases)
    u = np.zeros(total)
    for t in range(total):
        if t < WASHOUT:
            u[t] = np.sin(2.0 * np.pi * t / PERIOD)
        else:
            ph = phases[(t - WASHOUT) // phase_len]
            u[t] = waveform(ph, t - WASHOUT)
    mu, sd = u.mean(), u.std()
    u = (u - mu) / sd
    # np.roll wrap artifact: to avoid y[-1] = u[0], generate the last
    # sample as the continuation of the final phase (standardized).
    nxt = (waveform(phases[-1], total - WASHOUT) - mu) / sd
    y = np.empty(total)
    y[:-1] = u[1:]
    y[-1] = nxt
    return u, y


def make_stream_states(u_all, seed, fmap):
    """Generate the reservoir/data stream once, shared by all methods."""
    total = len(u_all)
    res = ReservoirEngine(1, D, 0.1, 0.3, 0.8, seed=seed)
    Phi = np.empty((total - WASHOUT, fmap.n_features))
    for t in range(total):
        x = res.step(np.array([u_all[t]]))
        if t >= WASHOUT:
            Phi[t - WASHOUT] = fmap.transform(x)
    return Phi


def run_method(kind, Phi, Y, sig_var):
    """kind in {'full','block','blockmem','ctxmem'} -> per-step error series."""
    F = D * K + 1
    if kind == 'multihead':
        model = MultiHeadMemory(
            lambda: BlockDiagRLS(D, K, 1, forgetting=LAM,
                                 init_var=INIT_VAR,
                                 windup_mult=WINDUP_MULT),
            n_slots=3)
        upd = model.update
    elif kind == 'full':
        model = FullRLS(F, 1, forgetting=LAM, init_var=INIT_VAR,
                        windup_mult=WINDUP_MULT)
        upd = model.update
    elif kind == 'block':
        model = BlockDiagRLS(D, K, 1, forgetting=LAM, init_var=INIT_VAR,
                             windup_mult=WINDUP_MULT)
        upd = model.update
    elif kind == 'blockmem':
        base = BlockDiagRLS(D, K, 1, forgetting=LAM, init_var=INIT_VAR,
                            windup_mult=WINDUP_MULT)
        model = DualMemory(base, beta=MEM_BETA, gamma=MEM_GAMMA,
                           consolidate_every=MEM_CONS_EVERY,
                           p_inflate=MEM_P_INFLATE)
        upd = model.update
    else:  # ctxmem
        base = BlockDiagRLS(D, K, 1, forgetting=LAM, init_var=INIT_VAR,
                            windup_mult=WINDUP_MULT)
        model = ContextualMemory(base, n_slots=3, beta=MEM_BETA,
                                 gamma=MEM_GAMMA, eta=2.0,
                                 consolidate_every=MEM_CONS_EVERY,
                                 e_gate=0.5,
                                 p_inflate=MEM_P_INFLATE)
        upd = model.update

    errs = np.empty(Phi.shape[0])
    for i in range(Phi.shape[0]):
        e = upd(Phi[i], Y[i])
        errs[i] = float(np.sum(e ** 2))
    return errs, model


def window_nmse(errs, y, lo, hi):
    """Mean e^2 in the window / full-series variance (cross-phase compare)."""
    return float(np.mean(errs[lo:hi]) / np.var(y))


def main():
    fmap = SplineFeatureMap(D, K, 3)
    methods = ['full', 'block', 'blockmem', 'multihead', 'ctxmem']
    labels = {'full': 'FullRLS+bound', 'block': 'BlockRLS+bound',
              'blockmem': 'BlockRLS+naive', 'multihead': 'MultiHead',
              'ctxmem': 'BlockRLS+CTX'}

    # metric[method][phase][metric] = [seed values]
    met = {m: {p: {'onset': [], 'end': []} for p in range(3)}
           for m in methods}
    disc = {m: [] for m in ('blockmem', 'ctxmem')}
    slot_reports = []

    for seed in range(SEEDS):
        u_all, y_all = build_stream(seed)
        Phi = make_stream_states(u_all, seed, fmap)
        Y = y_all[WASHOUT:].reshape(-1, 1)
        sig_var = float(np.var(y_all))
        print(f"seed {seed}: stream ready ({TOTAL} steps)")

        for m in methods:
            errs, model = run_method(m, Phi, Y, sig_var)
            for p in range(3):
                lo, hi = p * PHASE_LEN, (p + 1) * PHASE_LEN
                met[m][p]['onset'].append(
                    window_nmse(errs, y_all, lo, lo + WINDOW))
                met[m][p]['end'].append(
                    window_nmse(errs, y_all, hi - WINDOW, hi))
            if m in disc:
                disc[m].append(model.max_discontinuity)
            if m == 'ctxmem':
                # Which slot won in which phase? (diagnostic)
                log = np.array(model.slot_log)
                rep = [int(np.bincount(log[p*PHASE_LEN:(p+1)*PHASE_LEN],
                                       minlength=3).argmax())
                       for p in range(3)]
                slot_reports.append(rep)
                print(f"  {labels[m]:<15}: P3 onset NMSE = "
                      f"{met[m][2]['onset'][-1]:.3e} | slots(P1/2/3)={rep}"
                      f" | consolidations={model.n_consolidations}")
            else:
                print(f"  {labels[m]:<15}: P3 onset NMSE = "
                      f"{met[m][2]['onset'][-1]:.3e}")

    # ---------------- Report ----------------
    print("\n" + "=" * 78)
    print("SINE -> SQUARE -> SINE: window NMSE (mean +/- std, 10 seeds)")
    print("=" * 78)
    print(f"{'Method':<15} | {'F1-sine onset':<14} | {'F1-sine end':<14} | "
          f"{'F2-sq end':<14} | {'F3-sine ONSET':<14} | {'F3-sine end':<14}")
    print("-" * 78)
    for m in methods:
        def f(p, k):
            v = met[m][p][k]
            return f"{np.mean(v):.2e}+/-{np.std(v):.0e}"
        print(f"{labels[m]:<15} | {f(0,'onset'):<14} | {f(0,'end'):<14} | "
              f"{f(1,'end'):<14} | {f(2,'onset'):<14} | {f(2,'end'):<14}")
    print("-" * 78)
    for m in disc:
        print(f"{labels[m]} continuity deviation (max, ~0 expected): "
              f"{max(disc[m]):.2e}")
    # Key claim: ctxmem's F3 onset advantage (vs both bare block and naive)
    base = np.mean(met['block'][2]['onset'])
    naive = np.mean(met['blockmem'][2]['onset'])
    ctx = np.mean(met['ctxmem'][2]['onset'])
    print(f"-> F3 onset: block={base:.3e} | naive={naive:.3e} | "
          f"ctx={ctx:.3e}")
    print(f"-> CTX advantage vs block: x{base/max(ctx,1e-12):.2f} | "
          f"vs naive: x{naive/max(ctx,1e-12):.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
