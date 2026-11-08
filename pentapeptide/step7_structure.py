"""Descriptive numbers used in the README: where the optimal biases sit at the
box bounds (by PCCA+ macrostate), the biased macrostate populations, and an
exact split of the primary A->B MFPT into escape from macrostate 0 and the
subsequent search for state B inside the main basin.

Writes structure_C.json. Model-level.
"""
from __future__ import annotations

import json

import numpy as np

import C_lib as C
from C_lib import L
from step3_optimize import generators, ab_sets


def main():
    K = generators()['K']
    S = ab_sets()
    A, B = S['primary']
    Aalt, Balt = S['alternative']
    pi = C.stationary(K)
    lab = np.load(C.HERE / 'step1_arrays.npz')['labels']
    b = np.load(C.HERE / 'optima_biases.npz')
    out = dict(note=__doc__.strip().splitlines()[0], biases={})
    for key in b.files:
        x = b[key]
        U = float(np.round(np.abs(x).max(), 6))
        up = np.abs(x - U) < 1e-6
        lo = np.abs(x + U) < 1e-6
        pib = L.biased_pi_from_reference(pi, x)
        out['biases'][key] = dict(
            U=U, b_A=float(x[A[0]]), b_B=float(x[B[0]]),
            n_upper=int(up.sum()), n_lower=int(lo.sum()),
            upper_by_macrostate=np.bincount(lab[up], minlength=5).tolist(),
            lower_by_macrostate=np.bincount(lab[lo], minlength=5).tolist(),
            biased_macrostate_population=[float(pib[lab == c].sum()) for c in range(5)])
    out['unbiased_macrostate_population'] = [float(pi[lab == c].sum()) for c in range(5)]
    out['correlations'] = dict(
        spectral_U2_vs_mfpt_alternative_U2=float(np.corrcoef(b['spectral_U2'], b['mfpt_AB_alternative_U2'])[0, 1]),
        spectral_U2_vs_mfpt_primary_U2=float(np.corrcoef(b['spectral_U2'], b['mfpt_AB_U2'])[0, 1]),
        mfpt_primary_U2_vs_mfpt_alternative_U2=float(np.corrcoef(b['mfpt_AB_U2'], b['mfpt_AB_alternative_U2'])[0, 1]),
        mfpt_AB_U2_vs_mfpt_BA_U2=float(np.corrcoef(b['mfpt_AB_U2'], b['mfpt_BA_U2'])[0, 1]))

    # Exact split (strong Markov property): MFPT(A -> B) = MFPT(A -> main basin)
    # + E[MFPT(entry state -> B)], entry state = first state hit in the main basin.
    def split(Kx):
        n = Kx.shape[0]
        N = np.setdiff1d(np.arange(n), Balt)
        H = np.linalg.solve(-Kx[np.ix_(N, N)], Kx[np.ix_(N, Balt)])
        h = H[np.searchsorted(N, A[0])]
        keep = np.setdiff1d(np.arange(n), B)
        mB = np.zeros(n)
        mB[keep] = np.linalg.solve(-Kx[np.ix_(keep, keep)], np.ones(len(keep)))
        esc = L.mfpt_K(Kx, A, Balt)
        search = float(h @ mB[Balt])
        return dict(escape_from_macrostate0=esc, search_for_B_in_main_basin=search,
                    total=esc + search, direct=L.mfpt_K(Kx, A, B))
    out['mfpt_split'] = dict(unbiased=split(K))
    for key in ('spectral_U2', 'mfpt_AB_U2'):
        out['mfpt_split'][key] = split(L.tilt_generator(K, b[key]))
    # Last crossings grouped by the macrostate on the far side of each cut edge,
    # and the overlap with the unbiased network after this grouping.
    lc = np.load(C.HERE / 'last_crossing_distributions.npz')
    dest = lab[lc['cut_j']]
    groups = sorted(set(dest.tolist()))
    p0 = np.array([lc['primary_unbiased'][dest == g].sum() for g in groups])
    out['last_crossing_by_destination_macrostate'] = dict(
        destination_macrostates=groups,
        n_cut_edges_per_destination=[int((dest == g).sum()) for g in groups],
        unbiased=p0.tolist())
    for key in ('spectral_U1', 'mfpt_AB_U1', 'spectral_U2', 'mfpt_AB_U2', 'spectral_U3', 'mfpt_AB_U3'):
        pg = np.array([lc[key][dest == g].sum() for g in groups])
        out['last_crossing_by_destination_macrostate'][key] = dict(
            weights=pg.tolist(), overlap_with_unbiased=float(np.minimum(p0, pg).sum()))
    C.dump(C.HERE / 'structure_C.json', out)
    print(json.dumps(C.to_jsonable(out['last_crossing_by_destination_macrostate']), indent=1))
    print(json.dumps(C.to_jsonable(out['mfpt_split']), indent=1))
    print(json.dumps(C.to_jsonable(out['correlations']), indent=1))


if __name__ == '__main__':
    main()
