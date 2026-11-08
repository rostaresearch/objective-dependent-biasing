"""Step 3/4 evaluation: pick the best optimum of each case from runs/, verify
gradients by finite differences, check first-order (KKT) conditions, and
compute the cross-objective table with Omega_J, C_TS and ESS/N.

Writes results_C_pentapeptide.json and optima_biases.npz.
All numbers are model-level: exact properties of the estimated MSM.
"""
from __future__ import annotations

import json

import numpy as np

import C_lib as C
from C_lib import L, PSC, MPS
from step3_optimize import RUNS, generators, ab_sets, flatten_start

US = (1.0, 2.0, 3.0)


def load_run(name):
    meta = json.loads((RUNS / f'{name}.json').read_text())
    arrs = np.load(RUNS / f'{name}.npz')
    return {k: dict(meta[k], x=arrs[k]) for k in meta}


def pick_spectral(K0, names):
    """Evaluate every start with the imported L.spectral_gap_K and return the best."""
    cand = {}
    for nm in names:
        if not (RUNS / f'{nm}.json').exists():
            continue
        for k, r in load_run(nm).items():
            cand[f'{nm}:{k}'] = (L.spectral_gap_K(L.tilt_generator(K0, r['x'])), r['x'])
    vals = {k: v[0] for k, v in cand.items()}
    best = max(vals, key=vals.get)
    v = np.array(list(vals.values()))
    return best, cand[best][1], dict(values=vals, best=best, max=float(v.max()), min=float(v.min()),
                                     median=float(np.median(v)),
                                     spread_rel=float((v.max() - v.min()) / v.max()))


def pick_mfpt(K0, A, B, name):
    cand = {}
    for k, r in load_run(name).items():
        cand[f'{name}:{k}'] = (L.mfpt_K(L.tilt_generator(K0, r['x']), A, B), r['x'])
    vals = {k: v[0] for k, v in cand.items()}
    best = min(vals, key=vals.get)
    v = np.array(list(vals.values()))
    return best, cand[best][1], dict(values=vals, best=best, min=float(v.min()), max=float(v.max()),
                                     median=float(np.median(v)),
                                     spread_rel=float((v.max() - v.min()) / v.min()))


# ----------------------------------------------------------------- gradient checks

def fd_spectral(K0, u, idx, eps=1e-5):
    fd = []
    for i in idx:
        up = u.copy(); up[i] += eps
        um = u.copy(); um[i] -= eps
        fd.append((L.spectral_gap_K(L.tilt_generator(K0, up)) -
                   L.spectral_gap_K(L.tilt_generator(K0, um))) / (2 * eps))
    return np.array(fd)


def fd_mfpt(K0, u, A, B, idx, eps=1e-5):
    fd = []
    for i in idx:
        up = u.copy(); up[i] += eps
        um = u.copy(); um[i] -= eps
        fd.append((L.mfpt_K(L.tilt_generator(K0, up), A, B) -
                   L.mfpt_K(L.tilt_generator(K0, um), A, B)) / (2 * eps))
    return np.array(fd)


def gradient_checks(K0, A, B, points):
    rng = np.random.default_rng(11)
    eng = C.SymGap(K0, C.stationary(K0))
    out = {}
    for name, u in points.items():
        g, grad = PSC.gap_and_grad(u, K0)
        _, grad_sym = eng.gap_grad(u)
        idx = rng.choice(len(u), size=24, replace=False)
        fd = fd_spectral(K0, u, idx)
        g0, err_psc, _, _, _ = PSC.verify_gradient(K0, u)            # imported check, 8 components
        m, mgrad = MPS.mfpt_and_grad(u, K0, A, B)
        mfd = fd_mfpt(K0, u, A, B, np.arange(len(u)))
        m0, err_mps, _, _, _ = MPS._verify_gradient(K0, u, A, B)      # imported check, 8 components
        Kb = L.tilt_generator(K0, u)
        w, _ = C.reversible_spectrum(Kb, C.stationary(Kb))
        out[name] = dict(
            spectral=dict(gap=g, lambda2_minus_lambda3=float(w[1] - w[2]),
                          fd_components=len(idx), eps=1e-5,
                          max_rel_err_fd=float(np.max(np.abs(fd - grad[idx])) / np.max(np.abs(grad[idx]))),
                          max_rel_err_imported_verify_gradient=float(err_psc),
                          max_rel_diff_symmetric_route_all_components=float(
                              np.max(np.abs(grad - grad_sym)) / np.max(np.abs(grad)))),
            mfpt=dict(mfpt=m, fd_components=len(u), eps=1e-5,
                      max_rel_err_fd=float(np.max(np.abs(mfd - mgrad)) / np.max(np.abs(mgrad))),
                      max_rel_err_imported_verify_gradient=float(err_mps)),
        )
        print('grad check', name, json.dumps(C.to_jsonable(out[name])), flush=True)
    return out


def kkt(u, grad_min, U, tol=1e-6):
    """First-order check for minimising f over the box: projected gradient."""
    up = u >= U - tol
    lo = u <= -U + tol
    pg = grad_min.copy()
    pg[up] = np.maximum(grad_min[up], 0.0)
    pg[lo] = np.minimum(grad_min[lo], 0.0)
    return dict(n_upper=int(up.sum()), n_lower=int(lo.sum()), n_interior=int((~up & ~lo).sum()),
                max_abs_grad=float(np.abs(grad_min).max()),
                max_abs_projected_grad=float(np.abs(pg).max()),
                rel_projected_grad=float(np.abs(pg).max() / np.abs(grad_min).max()))


def strip(m):
    m = dict(m)
    m.pop('last_crossing_probability', None)
    return m


def evaluate_bias(sys_, b, U, objective):
    m = sys_.metrics(b)
    m['IPR_biased_last_crossing'] = C.ipr(m['last_crossing_probability'])
    m['U'] = U
    if objective == 'spectral':
        g, grad = PSC.gap_and_grad(b, sys_.K0)
        m['kkt'] = kkt(b, -grad, U)
    elif objective in ('mfpt_AB', 'mfpt_BA'):
        A, B = (sys_.A, sys_.B) if objective == 'mfpt_AB' else (sys_.B, sys_.A)
        _, grad = MPS.mfpt_and_grad(b, sys_.K0, A, B)
        m['kkt'] = kkt(b, grad, U)
    return m


def main():
    G = generators()
    S = ab_sets()
    K = G['K']
    A, B = S['primary']
    sysP = C.System(K, A, B, 'primary')
    sysA = C.System(K, *S['alternative'], 'alternative')
    sysK2 = C.System(G['K2'], A, B, 'K2')

    res = dict(
        note='Model-level results: exact properties of the estimated 250-state pentapeptide MSM '
             '(generator K=(T-I)/tau), conditional on that estimate. Times in units of the lag tau.',
        unbiased=dict(primary=dict(A=A, B=B, gap0=sysP.gap0, mfpt0_AB=sysP.t0, mfpt0_BA=sysP.t0_BA,
                                   n_cut_edges=len(sysP.p0), IPR=C.ipr(sysP.p0)),
                      alternative=dict(A=sysA.A, B=sysA.B, gap0=sysA.gap0, mfpt0_AB=sysA.t0,
                                       mfpt0_BA=sysA.t0_BA, n_cut_edges=len(sysA.p0), IPR=C.ipr(sysA.p0)),
                      K2=dict(gap0=sysK2.gap0, mfpt0_AB=sysK2.t0, mfpt0_BA=sysK2.t0_BA,
                              n_cut_edges=len(sysK2.p0), IPR=C.ipr(sysK2.p0))),
        optimizer_spread={}, cross_objective={}, robustness_U2={}, last_crossing={},
    )
    biases = {}
    last = dict(primary_unbiased=sysP.p0, alternative_unbiased=sysA.p0,
                cut_i=sysP.ref['i'], cut_j=sysP.ref['j'],
                alt_cut_i=sysA.ref['i'], alt_cut_j=sysA.ref['j'])

    # ---- Step 3: U = 1, 2, 3 on the primary A/B
    for U in US:
        names = [f'fast_spectral_K_U{U:g}', f'faithful_spectral_U{U:g}']
        kb, bs, spread_s = pick_spectral(K, names)
        km, bm, spread_m = pick_mfpt(K, A, B, f'mfpt_K_AB_primary_U{U:g}')
        res['optimizer_spread'][f'spectral_U{U:g}'] = spread_s
        res['optimizer_spread'][f'mfpt_AB_U{U:g}'] = spread_m
        ms = evaluate_bias(sysP, bs, U, 'spectral')
        mm = evaluate_bias(sysP, bm, U, 'mfpt_AB')
        res['cross_objective'][f'U{U:g}'] = dict(spectral_optimum=strip(ms), mfpt_AB_optimum=strip(mm))
        biases[f'spectral_U{U:g}'] = bs
        biases[f'mfpt_AB_U{U:g}'] = bm
        last[f'spectral_U{U:g}'] = np.array(ms['last_crossing_probability'])
        last[f'mfpt_AB_U{U:g}'] = np.array(mm['last_crossing_probability'])
        print(f'U={U}: spectral opt gap x{ms["gap_speedup"]:.3f} mfpt x{ms["mfpt_AB_speedup"]:.3f} '
              f'OmJ {ms["Omega_J"]:.3f} CTS {ms["C_TS"]:.3f} | mfpt opt gap x{mm["gap_speedup"]:.3f} '
              f'mfpt x{mm["mfpt_AB_speedup"]:.3f} OmJ {mm["Omega_J"]:.3f} CTS {mm["C_TS"]:.3f}', flush=True)

    # ---- Step 4: robustness at U = 2
    U = 2.0
    bs2 = biases['spectral_U2']
    # (a) alternative A/B: the spectral optimum does not depend on A/B; the MFPT optimum is re-optimized
    km, bm_alt, spread = pick_mfpt(K, sysA.A, sysA.B, 'mfpt_K_AB_alternative_U2')
    res['optimizer_spread']['mfpt_AB_alternative_U2'] = spread
    ms_alt = evaluate_bias(sysA, bs2, U, 'spectral')
    mm_alt = evaluate_bias(sysA, bm_alt, U, 'mfpt_AB')
    # cross-application: primary MFPT optimum evaluated on the alternative A/B and vice versa
    m_prim_on_alt = evaluate_bias(sysA, biases['mfpt_AB_U2'], U, None)
    m_alt_on_prim = evaluate_bias(sysP, bm_alt, U, None)
    res['robustness_U2']['alternative_AB'] = dict(
        spectral_optimum=strip(ms_alt), mfpt_AB_optimum=strip(mm_alt),
        primary_mfpt_optimum_evaluated_on_alternative=strip(m_prim_on_alt),
        alternative_mfpt_optimum_evaluated_on_primary=strip(m_alt_on_prim),
        bias_correlation_alt_vs_primary_mfpt_optimum=float(np.corrcoef(bm_alt, biases['mfpt_AB_U2'])[0, 1]))
    biases['mfpt_AB_alternative_U2'] = bm_alt
    last['alternative_spectral_U2'] = np.array(ms_alt['last_crossing_probability'])
    last['alternative_mfpt_AB_U2'] = np.array(mm_alt['last_crossing_probability'])
    # (b) direction: B->A MFPT optimum on the primary A/B
    km, bm_ba, spread = pick_mfpt(K, B, A, 'mfpt_K_BA_primary_U2')
    res['optimizer_spread']['mfpt_BA_U2'] = spread
    m_ba = evaluate_bias(sysP, bm_ba, U, 'mfpt_BA')
    res['robustness_U2']['direction_BA'] = dict(
        mfpt_BA_optimum=strip(m_ba),
        note='Evaluated with the primary A->B reference (q0, cut, Omega_J refer to A->B).',
        bias_correlation_BA_vs_AB_optimum=float(np.corrcoef(bm_ba, biases['mfpt_AB_U2'])[0, 1]))
    biases['mfpt_BA_U2'] = bm_ba
    # (c) generator construction K2 = (T^2 - I)/(2 tau)
    kb, bs_k2, spread_s = pick_spectral(G['K2'], ['fast_spectral_K2_U2'])
    km, bm_k2, spread_m = pick_mfpt(G['K2'], A, B, 'mfpt_K2_AB_primary_U2')
    res['optimizer_spread']['K2_spectral_U2'] = spread_s
    res['optimizer_spread']['K2_mfpt_AB_U2'] = spread_m
    res['robustness_U2']['generator_K2'] = dict(
        note='K2=(T^2-I)/(2 tau): same model propagated to 2 tau, converted by the same rule. '
             'A generator-construction sensitivity test, not an independent second-lag estimate.',
        spectral_optimum=strip(evaluate_bias(sysK2, bs_k2, U, 'spectral')),
        mfpt_AB_optimum=strip(evaluate_bias(sysK2, bm_k2, U, 'mfpt_AB')),
        K_optima_evaluated_on_K2=dict(spectral=strip(evaluate_bias(sysK2, bs2, U, None)),
                                      mfpt_AB=strip(evaluate_bias(sysK2, biases['mfpt_AB_U2'], U, None))),
        bias_correlation_K2_vs_K=dict(spectral=float(np.corrcoef(bs_k2, bs2)[0, 1]),
                                      mfpt_AB=float(np.corrcoef(bm_k2, biases['mfpt_AB_U2'])[0, 1])))
    biases['K2_spectral_U2'] = bs_k2
    biases['K2_mfpt_AB_U2'] = bm_k2

    # ---- Bias-profile similarity between the two objectives
    res['bias_similarity'] = {f'U{U:g}': dict(
        pearson=float(np.corrcoef(biases[f'spectral_U{U:g}'], biases[f'mfpt_AB_U{U:g}'])[0, 1]),
        pi_weighted_rms_difference=float(np.sqrt(np.dot(sysP.pi0, (
            (biases[f'spectral_U{U:g}'] - np.dot(sysP.pi0, biases[f'spectral_U{U:g}'])) -
            (biases[f'mfpt_AB_U{U:g}'] - np.dot(sysP.pi0, biases[f'mfpt_AB_U{U:g}']))) ** 2))))
        for U in US}

    # ---- Gradient verification by finite differences
    rng = np.random.default_rng(5)
    points = dict(zero=np.zeros(K.shape[0]), random_U2=rng.uniform(-2, 2, K.shape[0]),
                  flatten_U2=flatten_start(K, 2.0), mfpt_optimum_U2=biases['mfpt_AB_U2'],
                  spectral_optimum_U2=biases['spectral_U2'])
    res['gradient_checks'] = gradient_checks(K, A, B, points)

    C.dump(C.HERE / 'results_C_pentapeptide.json', res)
    np.savez(C.HERE / 'optima_biases.npz', **biases)
    np.savez(C.HERE / 'last_crossing_distributions.npz', **last)
    print('saved')


if __name__ == '__main__':
    main()
