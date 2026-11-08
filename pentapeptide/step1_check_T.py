"""Steps 1 and 2: check the estimated pentapeptide transition matrix,
build the generator K = (T - I)/tau, and fix the A/B sets.

Writes step1_T_check.json and step1_arrays.npz.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.linalg import logm
from scipy.sparse.csgraph import connected_components

import C_lib as C
from C_lib import L


def isa_pcca(X: np.ndarray) -> np.ndarray:
    """Inner-simplex algorithm (Deuflhard and Weber 2005) on the right
    eigenvectors X (n x k, first column constant). Returns memberships chi."""
    n, k = X.shape
    ortho = X.copy()
    ind = np.zeros(k, int)
    ind[0] = int(np.argmax(np.linalg.norm(ortho, axis=1)))
    ortho = ortho - X[ind[0]][None, :]
    for m in range(1, k):
        if m > 1:
            t = ortho[ind[m - 1]].copy()
            t /= np.linalg.norm(t)
            ortho = ortho - np.outer(ortho @ t, t)
        ind[m] = int(np.argmax(np.linalg.norm(ortho, axis=1)))
    return X @ np.linalg.inv(X[ind])


def main():
    T = C.load_T()
    n = T.shape[0]
    us = json.loads(C.US_RESULTS.read_text())
    out = dict(inputs=dict(T=C.rel(C.T_PATH), T_sha256=C.sha256(C.T_PATH),
                           pentapeptide_results=C.rel(C.US_RESULTS),
                           pcca_labels=C.rel(C.PCCA_PATH), pcca_sha256=C.sha256(C.PCCA_PATH)),
               model_level_note='All quantities are exact properties of the estimated MSM; '
                                'they are conditional on that estimate.',
               lag_time='Not recorded in pentapeptide_us.py or the data files; '
                        'times are in units of the lag tau. Speedups are ratios and do not depend on tau.')

    # ---- 1a. Stochasticity, stationary distribution, detailed balance, connectivity
    rs = T.sum(axis=1)
    w, VL = np.linalg.eig(T.T)
    o = np.argsort(-w.real); w = w[o]; VL = VL[:, o]
    pi_T = np.real(VL[:, 0]); pi_T = pi_T / pi_T.sum()
    flux = pi_T[:, None] * T
    db = np.abs(flux - flux.T)
    off = flux.copy(); np.fill_diagonal(off, 0.0)
    nscc, _ = connected_components((T > 0).astype(int), directed=True, connection='strong')
    out['T'] = dict(
        n_states=n, nonzero_entries=int((T > 0).sum()),
        nonzero_offdiagonal=int((off > 0).sum()),
        min_entry=float(T.min()), max_row_sum_deviation=float(np.abs(rs - 1).max()),
        sparsity_pattern_symmetric=bool(np.array_equal(T > 0, T.T > 0)),
        diagonal_mean=float(T.diagonal().mean()), diagonal_max=float(T.diagonal().max()),
        n_zero_diagonal=int((T.diagonal() == 0).sum()),
        pi_min=float(pi_T.min()), pi_max=float(pi_T.max()),
        free_energy_range_kT=float(np.log(pi_T.max() / pi_T.min())),
        stationary_residual=float(np.abs(pi_T @ T - pi_T).max()),
        detailed_balance_max_abs=float(db.max()),
        detailed_balance_max_rel_to_max_flux=float(db.max() / off.max()),
        detailed_balance_total_variation=float(0.5 * db.sum() / off.sum()),
        n_strongly_connected_components=int(nscc),
        reversible_within_tolerance=bool(db.max() / off.max() < 1e-10),
    )

    # ---- 1b. Eigenvalues, implied timescales, embeddability
    lam = w.real
    neg = np.sort(lam[lam < 0])
    sep_neg = float(np.min(np.diff(neg))) if len(neg) > 1 else np.nan
    its = [-C.TAU / np.log(l) for l in lam[1:11]]
    K = C.generator_from_T(T)
    L.check_row_generator(K)
    pi = C.stationary(K)
    mu, psi = C.reversible_spectrum(K, pi)
    relax = [1.0 / -m for m in mu[1:11]]
    # Principal matrix logarithm: report why it is not used
    Lg = logm(T)
    Lg_real = np.real(Lg)
    off_log = Lg_real.copy(); np.fill_diagonal(off_log, 0.0)
    out['spectrum'] = dict(
        max_abs_imag_eigenvalue=float(np.abs(w.imag).max()),
        eigenvalues_T_top11=lam[:11].tolist(),
        n_negative_eigenvalues=int(len(neg)), min_eigenvalue=float(lam.min()),
        min_separation_between_negative_eigenvalues=sep_neg,
        implied_timescales_T_over_tau=its,
        generator_eigenvalues_top11=mu[:11].tolist(),
        generator_relaxation_times_over_tau=relax,
        logm=dict(max_abs_imag=float(np.abs(np.imag(Lg)).max()),
                  most_negative_offdiagonal_real_part=float(off_log.min()),
                  n_negative_offdiagonal_real_part=int((off_log < -1e-12).sum())),
        embeddability_note=(
            'T is diagonalizable with real spectrum and has simple negative eigenvalues. '
            'A real matrix logarithm exists only if negative eigenvalues come in pairs '
            '(Culver 1966), so T has no real logarithm and is not embeddable: '
            'logm(T) is complex. K=(T-I)/tau is used instead.'),
    )
    # ---- 1c. Generator checks
    jump_T = T / np.where(1 - T.diagonal() > 0, 1 - T.diagonal(), 1)[:, None]
    np.fill_diagonal(jump_T, 0.0)
    jump_K = K / (-K.diagonal())[:, None]
    np.fill_diagonal(jump_K, 0.0)
    Kflux = pi[:, None] * K
    out['generator'] = dict(
        definition='K = (T - I)/tau, tau = 1',
        min_offdiagonal=float(np.min(K + np.diag(np.full(n, np.inf)))),
        max_row_sum=float(np.abs(K.sum(axis=1)).max()),
        jump_chain_max_difference=float(np.abs(jump_T - jump_K).max()),
        pi_K_vs_pi_T_max_difference=float(np.abs(pi - pi_T).max()),
        detailed_balance_max_rel=float(np.abs(Kflux - Kflux.T).max() / np.abs(Kflux - np.diag(np.diag(Kflux))).max()),
        spectral_gap=float(L.spectral_gap_K(K)),
        spectral_gap_symmetric_route=float(-mu[1]),
        eigenvalue_relation_max_error=float(np.max(np.abs(np.sort(mu)[::-1][:20] - (np.sort(lam)[::-1][:20] - 1) / C.TAU))),
    )

    # ---- 2. A/B sets
    # Primary: reproduce the pentapeptide_us.py choice (extremes of the second
    # eigenvector of T.T, which is the left eigenvector of T).
    v2 = np.real(VL[:, 1])
    v2s = (v2 - v2.min()) / (v2.max() - v2.min())
    A_us, B_us = int(np.argmin(v2s)), int(np.argmax(v2s))
    assert (A_us, B_us) == (us['A'], us['B']) or (B_us, A_us) == (us['A'], us['B'])
    A_us, B_us = us['A'], us['B']
    mfpt_AB = L.mfpt_K(K, [A_us], [B_us])
    out['AB_primary'] = dict(
        source='pentapeptide_us.py: A = argmin, B = argmax of the second eigenvector of T.T '
               '(left eigenvector of T, i.e. pi times the slowest right eigenvector); '
               'reproduced here and equal to pentapeptide_results.json',
        A=[A_us], B=[B_us], pi_A=float(pi[A_us]), pi_B=float(pi[B_us]),
        F_A_kT=float(-np.log(pi[A_us] / pi.max())), F_B_kT=float(-np.log(pi[B_us] / pi.max())),
        B_is_most_populated_state=bool(B_us == int(np.argmax(pi))),
        mfpt_AB_generator=mfpt_AB, mfpt_AB_pentapeptide_results_json=us['mfpt_ref_AB_steps'],
        mfpt_AB_relative_difference=abs(mfpt_AB - us['mfpt_ref_AB_steps']) / us['mfpt_ref_AB_steps'],
        mfpt_BA_generator=L.mfpt_K(K, [B_us], [A_us]),
        mfpt_AB_fundamental_matrix=C.mfpt_fundamental(K, pi, [A_us], [B_us]),
    )

    # Alternative: PCCA+ macrostates separated by the slowest right eigenvector.
    labels = np.loadtxt(C.PCCA_PATH).argmax(axis=1)
    macro = []
    for c in range(labels.max() + 1):
        m = labels == c
        macro.append(dict(label=c, n_states=int(m.sum()), pi=float(pi[m].sum()),
                          psi2_pi_mean=float(np.average(psi[m, 1], weights=pi[m])),
                          psi3_pi_mean=float(np.average(psi[m, 2], weights=pi[m])),
                          contains_A_primary=bool(labels[A_us] == c),
                          contains_B_primary=bool(labels[B_us] == c)))
    psi2_means = np.array([mm['psi2_pi_mean'] for mm in macro])
    # Sign-free choice: B = most populated macrostate, A = the macrostate
    # farthest from B along psi_2 (eigenvector signs are arbitrary).
    cB = int(np.argmax([mm['pi'] for mm in macro]))
    cA = int(np.argmax(np.abs(psi2_means - psi2_means[cB])))
    A_alt = np.flatnonzero(labels == cA); B_alt = np.flatnonzero(labels == cB)
    # Independent PCCA check (inner simplex algorithm on the 5 slowest modes)
    chi = isa_pcca(psi[:, :5])
    lab_isa = chi.argmax(axis=1)
    mapping = {}
    for c in range(5):
        mapping[c] = int(np.bincount(labels[lab_isa == c], minlength=5).argmax()) if np.any(lab_isa == c) else -1
    lab_isa_mapped = np.array([mapping[c] for c in lab_isa])
    disagree = np.flatnonzero(lab_isa_mapped != labels)
    out['pcca'] = dict(
        macrostates=macro,
        isa_check=dict(n_states_assigned_differently=int(len(disagree)),
                       pi_mass_assigned_differently=float(pi[disagree].sum()),
                       mapping_isa_to_supplied=mapping,
                       alt_A_identical=bool(np.array_equal(np.flatnonzero(lab_isa_mapped == cA), A_alt)),
                       alt_B_n_states_isa=int((lab_isa_mapped == cB).sum())),
    )
    out['AB_alternative'] = dict(
        source='PCCA+ (5 macrostates, crisp labels supplied with T). B = the most populated '
               'macrostate; A = the macrostate whose pi-weighted mean of the slowest right '
               'eigenvector psi_2 is farthest from that of B. MFPT starts uniformly over A '
               '(code convention).',
        A=A_alt.tolist(), B=B_alt.tolist(), macro_A=cA, macro_B=cB,
        pi_A=float(pi[A_alt].sum()), pi_B=float(pi[B_alt].sum()),
        mfpt_AB=L.mfpt_K(K, A_alt, B_alt), mfpt_BA=L.mfpt_K(K, B_alt, A_alt),
        primary_A_in_alt_A=bool(A_us in A_alt), primary_B_in_alt_B=bool(B_us in B_alt),
    )

    # Reference cut summaries for both A/B choices
    for key in ('AB_primary', 'AB_alternative'):
        s = C.System(K, out[key]['A'], out[key]['B'])
        out[key]['cut'] = dict(n_S=int((s.q0 < 0.5).sum()), n_cut_edges=int(len(s.p0)),
                               IPR=C.ipr(s.p0),
                               capacity_relative_residual=float(s.ref['capacity_relative_residual']),
                               spectral_gap0=s.gap0, mfpt0_AB=s.t0, mfpt0_BA=s.t0_BA)

    C.dump(C.HERE / 'step1_T_check.json', out)
    np.savez(C.HERE / 'step1_arrays.npz', pi=pi, psi=psi[:, :6], mu=mu, lam_T=lam,
             labels=labels, q0_primary=C.System(K, [A_us], [B_us]).q0)
    print(json.dumps(C.to_jsonable({k: out[k] for k in ('T', 'generator')}), indent=1))
    print(json.dumps(C.to_jsonable(out['spectrum']), indent=1)[:2500])
    print(json.dumps(C.to_jsonable({k: out[k] for k in ('AB_primary', 'pcca')}), indent=1)[:4000])
    alt = dict(out['AB_alternative']); alt.pop('A'); alt.pop('B')
    print(json.dumps(C.to_jsonable(alt), indent=1))


if __name__ == '__main__':
    main()
