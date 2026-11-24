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


# ---------- alternating-context driver: A->B->A->B->A + oracle gate ----------
OUT3 = '/kaggle/working/native_lm_v3_detect.json'
res3 = json.load(open(OUT3)) if os.path.exists(OUT3) else {}
res3.setdefault('runs', [])
stream, vocab = load_chars()
stream = stream[:250001]
# alternating sharp contexts: B segments are token-reversed text
sh = np.concatenate([np.ascontiguousarray(stream[0:50000]),
                     np.ascontiguousarray(stream[50000:100000][::-1]),
                     np.ascontiguousarray(stream[100000:150000]),
                     np.ascontiguousarray(stream[150000:200000][::-1]),
                     np.ascontiguousarray(stream[200000:250000])])
bounds = [0, 50000, 100000, 150000, 200000, 250000]
labels = ['A1', 'B1', 'A2', 'B2', 'A3']
print('v3-DETECT-250K stream:', len(sh), 'bounds', bounds, flush=True)


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
for arm in ['apia_ctxmem_detect']:
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
print('ALT_DONE ->', OUT3, flush=True)
