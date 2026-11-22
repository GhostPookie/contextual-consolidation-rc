"""
APIA 2.0 Research Prototype
Module: benchmark_block_rls.py
Purpose: statistically characterize the accuracy/speed/memory
      trade-off of Block-Diagonal RLS vs Full RLS and Ridge over
      10 seeds.

Protocol:
  - Data: Mackey-Glass tau=17 (dt=1, literature standard),
    input+target standardized
  - Reservoir: reservoir.py (ESP-guaranteed, unique sparse indices)
  - Features: rls_engine.SplineFeatureMap (true Cox-de Boor cubic
    B-spline)
  - Scenarios: train lengths {2000, 5000} — combined in a single
    stream via checkpoints (same data, same test window => clean
    comparison)
  - lambda_forget: {1.0, 0.98}
  - Seeds: 0..9 (reservoir, data and spline share the same seed)
  - Test: fixed 1000-step window after training, NMSE (divided by
    test variance)
"""

import numpy as np
import time

from reservoir import ReservoirEngine
from rls_engine import (SplineFeatureMap, RidgeReadout, FullRLS,
                        BlockDiagRLS, mackey_glass, nmse)

# ---------------- Experiment constants ----------------
N_SEEDS = 10
D, K = 100, 10
WASHOUT = 500
TRAIN_MAX = 5000            # single stream; 2000 read as checkpoint
CHECKPOINTS = [200, 500, 1000, 2000, 3500, 5000]
N_TEST = 1000
LAMBDAS = [1.0, 0.98]
INIT_VAR = 1e2
RIDGE_LAM = 1e-6
TOTAL = WASHOUT + TRAIN_MAX + N_TEST


def prepare_stream(seed):
    """For one seed: MG series -> reservoir states -> Phi (all)."""
    series = mackey_glass(TOTAL, seed=seed)
    u_all = (series - series.mean()) / series.std()
    y_all = np.roll(u_all, -1)

    res = ReservoirEngine(input_dim=1, reservoir_size=D, sparsity=0.1,
                          leakage_rate=0.3, input_scaling=0.8, seed=seed)
    fmap = SplineFeatureMap(input_dim=D, n_basis=K, degree=3)

    tr_end = WASHOUT + TRAIN_MAX
    te_end = tr_end + N_TEST

    Phi_tr = np.empty((TRAIN_MAX, fmap.n_features))
    Y_tr = np.empty((TRAIN_MAX, 1))
    Phi_te = np.empty((N_TEST, fmap.n_features))
    Y_te = np.empty((N_TEST, 1))

    for t in range(TOTAL):
        x = res.step(np.array([u_all[t]]))
        if t >= WASHOUT:
            phi = fmap.transform(x)
            if t < tr_end:
                Phi_tr[t - WASHOUT] = phi
                Y_tr[t - WASHOUT, 0] = y_all[t]
            else:
                Phi_te[t - tr_end] = phi
                Y_te[t - tr_end, 0] = y_all[t]
    return Phi_tr, Y_tr, Phi_te, Y_te


def run_online(model, Phi_tr, Y_tr, Phi_te, Y_te):
    """Run the online model on the stream; measure test NMSE at checkpoints."""
    curves = {}
    t_sum = 0.0
    for i in range(TRAIN_MAX):
        t0 = time.perf_counter()
        model.update(Phi_tr[i], Y_tr[i])
        t_sum += time.perf_counter() - t0
        step = i + 1
        if step in CHECKPOINTS:
            curves[step] = nmse(Y_te, Phi_te @ model.W)
    return curves, t_sum / TRAIN_MAX  # s/step


def main():
    F = D * K + 1
    mem_full_kb = F * F * 8 / 1024
    mem_blk_kb = D * K * K * 8 / 1024
    print("=" * 78)
    print(f"Block-RLS characterization: {N_SEEDS} seeds, D={D}, K={K}, "
          f"F={F}, lambdas={LAMBDAS}")
    print(f"Memory: Full P={mem_full_kb/1024:.1f} MB | Block P={mem_blk_kb:.0f} KB"
          f" ({mem_full_kb/mem_blk_kb:.0f}x)")
    print("=" * 78)

    # method -> lam -> {'nmse@cp': {cp: [seed...]}, 'time': [seed...]}
    methods = ["Ridge", "FullRLS", "BlockRLS"]
    res = {m: {l: {'nmse': {c: [] for c in CHECKPOINTS}, 'time': []}
               for l in LAMBDAS} for m in methods}

    for seed in range(N_SEEDS):
        Phi_tr, Y_tr, Phi_te, Y_te = prepare_stream(seed)
        print(f"seed {seed}: stream ready ({TRAIN_MAX+N_TEST} steps)")

        for lam in LAMBDAS:
            # Ridge (batch): separate fit per checkpoint — no memory cost
            for cp in CHECKPOINTS:
                r = RidgeReadout(F, 1).fit(Phi_tr[:cp], Y_tr[:cp], RIDGE_LAM)
                res["Ridge"][lam]['nmse'][cp].append(
                    nmse(Y_te, Phi_te @ r.W))
            res["Ridge"][lam]['time'].append(0.0)

            fr = FullRLS(F, 1, forgetting=lam, init_var=INIT_VAR)
            c_f, dt_f = run_online(fr, Phi_tr, Y_tr, Phi_te, Y_te)
            for cp in CHECKPOINTS:
                res["FullRLS"][lam]['nmse'][cp].append(c_f[cp])
            res["FullRLS"][lam]['time'].append(dt_f)

            br = BlockDiagRLS(D, K, 1, forgetting=lam, init_var=INIT_VAR)
            c_b, dt_b = run_online(br, Phi_tr, Y_tr, Phi_te, Y_te)
            for cp in CHECKPOINTS:
                res["BlockRLS"][lam]['nmse'][cp].append(c_b[cp])
            res["BlockRLS"][lam]['time'].append(dt_b)
            print(f"  lam={lam}: FullRLS NMSE@5000={c_f[5000]:.2e} "
                  f"({dt_f*1e3:.2f} ms/step) | "
                  f"BlockRLS NMSE@5000={c_b[5000]:.2e} "
                  f"({dt_b*1e6:.0f} us/step)")

    # ---------------- Report ----------------
    print("\n" + "=" * 78)
    print("RESULTS: NMSE (test, mean ± std, 10 seeds)")
    print("=" * 78)
    for lam in LAMBDAS:
        print(f"\n--- lambda_forget = {lam} ---")
        hdr = "Method".ljust(10) + "".join(
            f"@{c}".rjust(20) for c in CHECKPOINTS) + "ms/step".rjust(14)
        print(hdr)
        print("-" * 78)
        for m in methods:
            row = m.ljust(10)
            for cp in CHECKPOINTS:
                v = res[m][lam]['nmse'][cp]
                row += f"{np.mean(v):.2e}+/-{np.std(v):.0e}".rjust(20)
            tv = res[m][lam]['time']
            row += (f"{np.mean(tv)*1e3:.2f}" if np.mean(tv) > 0
                    else "offline").rjust(14)
            print(row)

    # ---------------- Trade-off summary (train=5000, both lam) ----------------
    print("\n" + "=" * 78)
    print("TRADE-OFF SUMMARY (train=5000 steps)")
    for lam in LAMBDAS:
        nf = np.array(res['FullRLS'][lam]['nmse'][5000])
        nb = np.array(res['BlockRLS'][lam]['nmse'][5000])
        tf = np.mean(res['FullRLS'][lam]['time'])
        tb = np.mean(res['BlockRLS'][lam]['time'])
        print(f" lam={lam}: NMSE cost x{np.mean(nb/nf):.1f} "
              f"| speed gain x{tf/tb:.0f} "
              f"| memory gain x{mem_full_kb/mem_blk_kb:.0f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
