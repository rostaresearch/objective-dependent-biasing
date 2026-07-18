"""Measure the per-edge reweighting error of the pooled adaptive-bias DHAM
inverse on the real Ras phosphate-unbinding data (review Critical 1).

The shipped estimator assigns ONE mean bias V_state[i] per microstate and
inverse-tilts the pooled counts.  Averaging a time-dependent bias before
exponentiating does not commute with the reweighting.  Rather than assert that
this is imperfect, measure it.

For every observed transition i->j we know the actual applied bias at both
endpoints, so the per-observation edge bias

    X = (V(t+lag) - V(t)) / kBT        [dimensionless]

is directly evaluable, and so are both sides of the Jensen inequality.

Reported per edge (i,j):

  R_minus = E_w[exp(-X/2)] * exp(+E_w[X]/2)   >= 1   (review's R_ij)
  R_plus  = E_w[exp(+X/2)] * exp(-E_w[X]/2)   >= 1   (same gap, reverse edge)

      Both are pure Jensen gaps: they compare exponentiate-then-average with
      average-then-exponentiate on the SAME observations.  Equality iff X is
      constant over every observation of that edge.

  Q_ij    = exp(+dV_meanfield/2) / E_w[exp(+X/2)]

      The ACTUAL error of the shipped estimator: the weight it applies divided
      by the weight it should have applied.  This is strictly more informative
      than R alone because it also carries the mean-field mismatch
      (V_state is a mean over ALL frames in a bin, not over this edge's
      transition endpoints), which R assumes away.

Each is summarised unweighted AND weighted by the reference reactive current
Jhat0 -- the number that matters, because an estimator error on an edge that
carries no reactive current does not move Omega_J.

Headline configuration (matches the shipped audit):
    feature Pi_Q61, A = {cv < 5}, B = {cv > 8}, 12 x 8 grid, lag 1.
"""
from __future__ import annotations
import sys, json
import numpy as np

sys.path.insert(0, r'C:\Users\edina\Dropbox\MSM_Roundtable_2026')
from mechanism_audit_ras_proper import (
    compute_diagnostics, dham_unbias, regularise_if_disconnected, KBT)
from mechanism_audit_highd_n20 import (
    stationary_from_K, committor_K, positive_net_current)

NPZ = r'C:\Users\edina\Dropbox\MSM_Roundtable_2026\pi_features.npz'
OUT = r'C:\Users\edina\Dropbox\MSM_Roundtable_2026\ras_pi_jensen_Rij.json'

N_CV, N_F = 12, 8
A_MAX, B_MIN, LAG = 5.0, 8.0, 1


def build_with_transitions(cvs, vbs, feats, feat_idx, n_cv=N_CV, n_f=N_F,
                           A_max=A_MAX, B_min=B_MIN, lag=LAG):
    """Mirror ras_pi_audit.build EXACTLY, but also return the per-transition
    record (i, j, X, prop_id) that the pooled estimator throws away."""
    cv_all = np.concatenate([np.asarray(c, float) for c in cvs])
    f_all = np.concatenate([np.asarray(f, float)[:, feat_idx] for f in feats])
    V_all = np.concatenate([np.asarray(v, float) for v in vbs])
    good = np.isfinite(f_all)
    e_cv = np.linspace(cv_all.min() - 1e-6, cv_all.max() + 1e-6, n_cv + 1)
    e_f = np.linspace(np.nanmin(f_all) - 1e-6, np.nanmax(f_all) + 1e-6, n_f + 1)
    ic = np.clip(np.digitize(cv_all, e_cv) - 1, 0, n_cv - 1)
    ifv = np.clip(np.digitize(np.where(good, f_all, e_f[0]), e_f) - 1, 0, n_f - 1)
    s = ic * n_f + ifv
    N = n_cv * n_f

    C = np.zeros((N, N))
    tr_i, tr_j, tr_X, tr_p = [], [], [], []
    cur = 0
    for p, c in enumerate(cvs):                 # strict propagation boundaries
        n = len(c); st, en = cur, cur + n; cur += n
        a = s[st:en - lag]; b_ = s[st + lag:en]
        np.add.at(C, (a, b_), 1.0)
        # the actual applied bias difference across each transition, in kBT
        Xp = (V_all[st + lag:en] - V_all[st:en - lag]) / KBT
        tr_i.append(a); tr_j.append(b_); tr_X.append(Xp)
        tr_p.append(np.full(len(a), p, dtype=int))
    C = 0.5 * (C + C.T)

    V_state = np.zeros(N); cnt = np.zeros(N)
    np.add.at(V_state, s, V_all); np.add.at(cnt, s, 1.0)
    ok = cnt > 0; V_state[ok] /= cnt[ok]

    cv_centers = 0.5 * (e_cv[:-1] + e_cv[1:])
    state_cv = np.repeat(cv_centers, n_f)
    A = [i for i in range(N) if state_cv[i] < A_max and cnt[i] > 0]
    B = [i for i in range(N) if state_cv[i] > B_min and cnt[i] > 0]
    return dict(C=C, V_state=V_state, A=A, B=B, cnt=cnt, N=N,
                tr_i=np.concatenate(tr_i), tr_j=np.concatenate(tr_j),
                tr_X=np.concatenate(tr_X), tr_p=np.concatenate(tr_p))


def reference_current(C, V_state, A, B):
    """Jhat0: normalised positive net reactive current of the reference MSM,
    built exactly as compute_diagnostics does."""
    C_used, _ = regularise_if_disconnected(C, A, B, n_cv=N_CV, n_f=N_F,
                                           alpha=1e-3)
    M0 = dham_unbias(C_used, V_state)
    K0 = M0 - np.eye(M0.shape[0])
    pi0 = stationary_from_K(K0)
    q0 = committor_K(K0, A, B, eps=1e-12)
    return positive_net_current(K0, pi0, q0)


def main():
    d = np.load(NPZ, allow_pickle=True)
    names = [str(x) for x in d['feature_names']]
    cvs, vbs, feats = d['cv'], d['vbias'], d['feat']
    fi = names.index('Pi_Q61')
    print(f'features={names}\nheadline feature = Pi_Q61 (index {fi})')
    print(f'kBT = {KBT:.4f} kJ/mol\n')

    B_ = build_with_transitions(cvs, vbs, feats, fi)
    N = B_['N']
    print(f"{len(cvs)} propagations, {len(B_['tr_X'])} transitions, "
          f"{N} microstates, |A|={len(B_['A'])}, |B|={len(B_['B'])}")

    chk = compute_diagnostics(B_['C'], B_['V_state'], B_['A'], B_['B'],
                              n_cv=N_CV, n_f=N_F, regularise=True,
                              alpha_pseudo=1e-3)
    print(f"reproduces shipped audit: Omega_J={chk['Omega_J']:.4f} "
          f"(expect 0.9823), D_edge={chk['D_edge']:.4f} (expect 0.0579)\n")

    Jhat0 = reference_current(B_['C'], B_['V_state'], B_['A'], B_['B'])

    # ---- group the per-transition X by edge -------------------------------
    i_, j_, X_ = B_['tr_i'], B_['tr_j'], B_['tr_X']
    key = i_.astype(np.int64) * N + j_.astype(np.int64)
    order = np.argsort(key, kind='stable')
    key_s, X_s = key[order], X_[order]
    bounds = np.flatnonzero(np.diff(key_s)) + 1
    groups = np.split(np.arange(len(key_s)), bounds)

    V = B_['V_state']
    rows = []
    for g in groups:
        k = key_s[g[0]]
        i, j = divmod(int(k), N)
        X = X_s[g]
        n_obs = len(X)
        Ex = float(X.mean())
        R_minus = float(np.mean(np.exp(-X / 2)) * np.exp(+Ex / 2))
        R_plus = float(np.mean(np.exp(+X / 2)) * np.exp(-Ex / 2))
        dV_mf = float((V[j] - V[i]) / KBT)          # what the estimator uses
        Q = float(np.exp(+dV_mf / 2) / np.mean(np.exp(+X / 2)))
        rows.append(dict(i=i, j=j, n_obs=n_obs, sd_X=float(X.std()),
                         E_X=Ex, dV_meanfield=dV_mf,
                         R_minus=R_minus, R_plus=R_plus, Q=Q,
                         J=float(Jhat0[i, j])))

    n_obs = np.array([r['n_obs'] for r in rows], float)
    Rm = np.array([r['R_minus'] for r in rows])
    Rp = np.array([r['R_plus'] for r in rows])
    Q = np.array([r['Q'] for r in rows])
    J = np.array([r['J'] for r in rows])
    sdX = np.array([r['sd_X'] for r in rows])

    print(f'{len(rows)} observed directed edges; '
          f'{int((J > 0).sum())} carry positive reactive current '
          f'(Jhat0 mass on observed edges = {J.sum():.4f})\n')

    def summarise(name, v):
        pct = np.percentile(v, [50, 90, 99, 100])
        # count-weighted and current-weighted means
        cw = float(np.sum(v * n_obs) / n_obs.sum())
        jw = float(np.sum(v * J) / J.sum()) if J.sum() > 0 else float('nan')
        print(f'  {name:9s} median={pct[0]:.5f}  p90={pct[1]:.5f}  '
              f'p99={pct[2]:.5f}  max={pct[3]:.5f}')
        print(f'  {"":9s} count-weighted mean={cw:.5f}   '
              f'CURRENT-WEIGHTED mean={jw:.5f}')
        return dict(median=float(pct[0]), p90=float(pct[1]), p99=float(pct[2]),
                    max=float(pct[3]), count_weighted=cw, current_weighted=jw)

    print('Jensen error factor (>=1; =1 iff the bias is constant over the edge):')
    s_Rm = summarise('R_minus', Rm)
    s_Rp = summarise('R_plus', Rp)
    print('\nActual pooled-estimator weight error Q = w_shipped / w_correct:')
    s_Q = summarise('Q', Q)
    print('\n  |log Q| (magnitude of the multiplicative error):')
    s_lQ = summarise('|log Q|', np.abs(np.log(Q)))
    print('\nper-edge spread of the applied bias, sd(X) [kBT]:')
    s_sd = summarise('sd_X', sdX)

    # current-weighted error expressed as a rate factor
    jw_lQ = s_lQ['current_weighted']
    print(f'\n  => current-weighted mean rate factor exp(|log Q|) = '
          f'{np.exp(jw_lQ):.4f}')
    print(f'  => compare with D_edge = {chk["D_edge"]:.4f} '
          f'(current-weighted mean |db|)')

    res = dict(
        config=dict(feature='Pi_Q61', A_max=A_MAX, B_min=B_MIN, n_cv=N_CV,
                    n_f=N_F, lag=LAG, kBT_kJmol=KBT,
                    n_props=int(len(cvs)), n_transitions=int(len(X_)),
                    n_edges=int(len(rows))),
        reproduced=dict(Omega_J=chk['Omega_J'], D_edge=chk['D_edge']),
        Jhat0_mass_on_observed_edges=float(J.sum()),
        R_minus=s_Rm, R_plus=s_Rp, Q=s_Q, abs_log_Q=s_lQ, sd_X=s_sd,
        current_weighted_rate_factor=float(np.exp(jw_lQ)),
        edges=rows,
    )
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'\nsaved {OUT}')


if __name__ == '__main__':
    main()
