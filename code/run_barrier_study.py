"""
Barrier/rate-relevance umbrella sampling study.

Two networks:
  1. 2D grid 40x20 with double-well + bottleneck (FE range 12 kBT, deep barrier)
  2. Pentapeptide MSM (250 states), with K derived as (T - I)/tau as a proxy
     generator.  Embeddability is imperfect but adequate as a first analytic pass.

For each network and each CV in {x (or basic), EigVec2, Committor, MFPT-CV}:
  * Sweep number of windows W in {1, 4, 8, 12, 16}
  * Sweep harmonic kappa in {1, 2, 4, 8}
  * Compute analytic barrier-error and rate-weighted error speedups vs unbiased
  * Pick the (W, kappa) that maximises each speedup

Saves results.json + arrays for figure generation.
"""
from __future__ import annotations
import os
import json
import numpy as np
import analytic_lib as L

# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _T_to_K(T: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Cheap proxy generator: K = (T - I) / tau.  Negative off-diagonals are
       clipped to zero (rarely triggered for well-built MSMs at moderate lag)."""
    K = (T - np.eye(T.shape[0])) / tau
    off = K.copy(); np.fill_diagonal(off, 0.0)
    if off.min() < 0:
        # zero out negative off-diagonals; rescale diagonal so rows sum to 0
        K2 = np.where(K < 0, 0.0, K)
        np.fill_diagonal(K2, 0.0)
        np.fill_diagonal(K2, -K2.sum(axis=1))
        K = K2
    return K


def make_grid_network():
    K0, coords, F = L.grid_2d_generator(nx=40, ny=20, barrier_height=4.0,
                                         bottleneck=True)
    pi0 = L.stationary_distribution_from_K(K0)
    x = coords[:, 0]
    A = np.where(x < -0.85)[0]
    B = np.where(x >  0.85)[0]
    cvs = dict(
        x         = (x - x.min()) / (x.max() - x.min()),   # natural physical CV
        EigVec2   = (lambda v: (v - v.min())/(v.max()-v.min()))(L.right_eigvec2_K(K0)),
        Committor = L.committor_K(K0, A, B),
        MFPTcv    = (lambda v: 1.0 - (v - v.min())/(v.max()-v.min()+1e-300))(
                       _mfpt_field(K0, B)),
    )
    return dict(label='2D grid (n=800)', K0=K0, pi0=pi0, A=A, B=B, cvs=cvs,
                coords=coords, F=F)


def _mfpt_field(K, target):
    n = K.shape[0]
    mask = np.ones(n, dtype=bool); mask[list(target)] = False
    non = np.where(mask)[0]
    KNN = K[np.ix_(non, non)]
    tau = np.linalg.solve(-KNN, np.ones(len(non)))
    full = np.zeros(n); full[non] = tau
    return full


def make_pentapeptide_network():
    T = np.load(os.path.join(PATH, 'pentapeptide_T.npy'))
    K0 = _T_to_K(T, tau=1.0)
    pi0 = L.stationary_distribution_from_K(K0)
    # define basins via slowest eigenvector extremes
    v2 = L.right_eigvec2_K(K0)
    n = K0.shape[0]
    # take the 10% most extreme of each end as basins
    n_basin = max(5, n // 20)
    A = np.argsort(v2)[:n_basin]
    B = np.argsort(v2)[-n_basin:]
    cvs = dict(
        Basic     = np.linspace(0.0, 1.0, n),
        EigVec2   = (lambda v: (v - v.min())/(v.max()-v.min()))(L.right_eigvec2_K(K0)),
        Committor = L.committor_K(K0, A, B),
        MFPTcv    = (lambda v: 1.0 - (v - v.min())/(v.max()-v.min()+1e-300))(
                       _mfpt_field(K0, B)),
    )
    return dict(label='Pentapeptide MSM (n=250)', K0=K0, pi0=pi0, A=A, B=B, cvs=cvs)


def study(net, total_budget: float = 1e5,
          Ws: list = (1, 2, 4, 8, 12, 16, 24),
          kappas: list = (2.0, 8.0, 32.0, 128.0, 512.0),
          band: tuple = (0.4, 0.6)) -> dict:
    """Variance approximation uses effective independent samples per window,
       N_k^eff = T_k / (2 tau_k).  Unbiased baseline uses T_total/(2 tau_0).
       Coverage failure diagnostic reported separately.
    """
    K0, pi0 = net['K0'], net['pi0']
    A, B = net['A'], net['B']
    q_ref = L.committor_K(K0, A, B)
    S = np.where((q_ref > band[0]) & (q_ref < band[1]))[0]
    bw = band[1] - band[0]
    # only widen if user did NOT request a tight band on purpose
    while len(S) < 3 and bw < 1.0 and (band[1] - band[0]) >= 0.1:
        bw += 0.1
        lo = max(0.0, 0.5 - bw/2)
        hi = min(1.0, 0.5 + bw/2)
        S = np.where((q_ref > lo) & (q_ref < hi))[0]
    p_dagger_ref = float(pi0[S].sum())
    rate_ref = L.reactive_flux_rate_K(K0, pi0, A, B)

    print(f"\n=== {net['label']} ===")
    print(f"  MFPT(A->B)  = {L.mfpt_K(K0, A, B):.4g}")
    print(f"  k_AB (TPT)  = {rate_ref['kAB']:.4g}    1/k_AB = {1/rate_ref['kAB']:.4g}")
    print(f"  spec gap    = {L.spectral_gap_K(K0):.4g}    tau0 = {1/L.spectral_gap_K(K0):.3g}")
    print(f"  band {band[0]:.3f}-{band[1]:.3f}  |S|={len(S)}  p_dagger = {p_dagger_ref:.4e}")

    # --- Oracle-flattening baseline (W=1, b=log pi) ---
    # NOTE: u_i = +kT log pi_i  =>  pi^b uniform.  Sign fixed per E.R.'s correction.
    b_or = L.oracle_flatten_bias(pi0)
    pib_or = L.biased_pi_from_reference(pi0, b_or)
    Kb_or = L.tilt_generator(K0, b_or)
    tau_or = 1.0 / max(L.spectral_gap_K(Kb_or), 1e-14)
    N_eff_or = total_budget / max(2.0 * tau_or, 1e-14)
    I_or = N_eff_or * pib_or
    var_pi_rel_or = 1.0 / np.maximum(I_or, 1e-9)
    tau0 = 1.0 / max(L.spectral_gap_K(K0), 1e-14)
    N_eff_un = total_budget / max(2.0 * tau0, 1e-14)
    var_pi_rel_un = 1.0 / np.maximum(N_eff_un * pi0, 1e-9)
    E_barrier_or = float(np.sum(pi0[S] ** 2 * var_pi_rel_or[S]) / max(p_dagger_ref ** 2, 1e-14))
    E_barrier_un = float(np.sum(pi0[S] ** 2 * var_pi_rel_un[S]) / max(p_dagger_ref ** 2, 1e-14))
    rho = L.main_path_weights_from_committor(q_ref, pi0, mode='tube')
    E_rate_or = float(np.sum(rho * var_pi_rel_or))
    E_rate_un = float(np.sum(rho * var_pi_rel_un))
    oracle = dict(
        barrier_speedup=float(E_barrier_un / max(E_barrier_or, 1e-14)),
        rate_speedup=float(E_rate_un / max(E_rate_or, 1e-14)),
        tau=float(tau_or), N_eff=float(N_eff_or))
    print(f"  oracle flatten:  N_eff={N_eff_or:.3g}  barrier {oracle['barrier_speedup']:7.2f}x"
          f"   rate {oracle['rate_speedup']:7.2f}x")

    results = []
    for cv_name, cv in net['cvs'].items():
        for W in Ws:
            for kappa in kappas:
                centres = np.linspace(cv.min() + 1e-4, cv.max() - 1e-4, W) \
                    if W > 1 else np.array([0.5 * (cv.min() + cv.max())])
                sample_time = total_budget / W
                try:
                    out = L.umbrella_family_metrics(
                        K0, cv, centres, kappa=kappa, pi0=pi0,
                        sample_time_per_window=sample_time,
                        barrier_set=S, basin_A=A, basin_B=B,
                    )
                except Exception as e:
                    results.append(dict(cv=cv_name, W=W, kappa=kappa, error=str(e)))
                    continue
                row = dict(
                    cv=cv_name, W=W, kappa=kappa,
                    barrier_speedup=float(out['barrier_speedup']),
                    rate_speedup=float(out['rate_speedup']),
                    coverage_failure_pdagger=float(out.get('coverage_failure_pdagger', 0.0)),
                    coverage_failure_rate=float(out.get('coverage_failure_rate', 0.0)),
                    N_eff_total_US=float(out['N_eff_total_US']),
                    median_overlap=float(np.median(out['neighbor_overlap']))
                                    if W > 1 else float('nan'),
                    slowest_window_timescale=float(out['timescale'].max()),
                )
                results.append(row)

    return dict(
        label=net['label'],
        n=int(K0.shape[0]),
        nA=int(len(A)), nB=int(len(B)),
        mfpt_unbiased=float(L.mfpt_K(K0, A, B)),
        spectral_gap=float(L.spectral_gap_K(K0)),
        p_dagger=p_dagger_ref,
        k_AB=float(rate_ref['kAB']),
        tau_unbiased=float(tau0),
        N_eff_unbiased=float(N_eff_un),
        oracle=oracle,
        rows=results,
        band_used=[float(band[0]), float(band[1])],
        nS=int(len(S)),
    )


def best_per_cv(rows, target='barrier_speedup'):
    bests = {}
    for r in rows:
        if 'error' in r: continue
        cv = r['cv']
        if cv not in bests or r[target] > bests[cv][target]:
            bests[cv] = r
    return bests


def main():
    out = {}
    cases = [
        ('grid',         make_grid_network,         (0.40, 0.60)),
        ('pentapeptide', make_pentapeptide_network, (0.495, 0.505)),  # tight per E.R.
    ]
    for name, builder, band in cases:
        net = builder()
        out[name] = study(net, total_budget=1e5, band=band)
        # print best per CV
        print(f"\n  --- best per CV (barrier) ---")
        for cv, r in best_per_cv(out[name]['rows'], 'barrier_speedup').items():
            print(f"    {cv:10s}  W={r['W']:2d} κ={r['kappa']:4.1f}"
                  f"  barrier speedup {r['barrier_speedup']:7.2f}x"
                  f"  rate speedup {r['rate_speedup']:7.2f}x")
        print(f"  --- best per CV (rate) ---")
        for cv, r in best_per_cv(out[name]['rows'], 'rate_speedup').items():
            print(f"    {cv:10s}  W={r['W']:2d} κ={r['kappa']:4.1f}"
                  f"  rate speedup {r['rate_speedup']:7.2f}x"
                  f"  barrier speedup {r['barrier_speedup']:7.2f}x")

    with open(os.path.join(PATH, 'barrier_study.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print('\nSaved barrier_study.json')


if __name__ == '__main__':
    main()
