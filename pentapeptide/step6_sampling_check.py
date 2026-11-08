"""Independent Monte Carlo check of the exact model numbers (MFPT, C_TS, Omega_J).

Primary A/B (A is a single state): unconditioned first-passage trajectories of
the biased continuous-time jump chain, started at A and stopped on hitting B.
This route uses no committor and no linear solve. The reactive segment is the
part after the last visit to A; its last S -> not-S crossing (S = {q0 < 1/2})
gives the last-crossing histogram, and its net edge traversals give the
normalised positive reactive current.

Alternative A/B (A is a 6-state set): reactive paths are drawn with the
committor-conditioned (Doob) jump chain, launched from A with weight
pi_i K_ij q_j (as in common/validate_sampling.py); MFPTs are
checked with unconditioned trajectories started uniformly in A.

Writes sampling_check.json. Model-level: sampling from the estimated MSM.
"""
from __future__ import annotations

import numpy as np

import C_lib as C
from C_lib import L, MA
from step3_optimize import generators, ab_sets

N_PATHS = 20000
N_BATCH = 20
N_TOP_EDGES = 40


def jump_tables(Kb, weights=None):
    n = Kb.shape[0]
    W = Kb.copy(); np.fill_diagonal(W, 0.0)
    if weights is not None:
        W = W * weights[None, :]
    rate = -np.diag(Kb)
    dest, cdf = [], []
    for i in range(n):
        d = np.flatnonzero(W[i] > 0)
        c = np.cumsum(W[i, d])
        dest.append(d)
        cdf.append(c / c[-1] if len(c) else c)
    return rate, dest, cdf


def first_passage(Kb, starts, Bset, S, rng, n_paths, record_reactive=True):
    n = Kb.shape[0]
    rate, dest, cdf = jump_tables(Kb)
    inB = np.zeros(n, bool); inB[Bset] = True
    times = np.empty(n_paths)
    last = {}
    net = np.zeros((N_BATCH, n, n))
    Aset = set(int(a) for a in starts)
    for p in range(n_paths):
        s = int(starts[rng.integers(len(starts))])
        traj = [s]; t = 0.0
        while not inB[s]:
            t += rng.exponential(1.0 / rate[s])
            s = int(dest[s][np.searchsorted(cdf[s], rng.random())])
            traj.append(s)
        times[p] = t
        if record_reactive:
            tr = np.array(traj)
            k0 = max(k for k, x in enumerate(traj) if x in Aset)
            seg = tr[k0:]
            cross = np.flatnonzero(S[seg[:-1]] & ~S[seg[1:]])
            e = (int(seg[cross[-1]]), int(seg[cross[-1] + 1]))
            last[e] = last.get(e, 0) + 1
            np.add.at(net[p * N_BATCH // n_paths], (seg[:-1], seg[1:]), 1.0)
    return times, last, net


def doob_paths(Kb, pib, A, B, S, rng, n_paths):
    n = Kb.shape[0]
    q = L.committor_K(Kb, A, B)
    rate, dest, cdf = jump_tables(Kb, weights=q)
    off = Kb.copy(); np.fill_diagonal(off, 0.0)
    launch = pib[A, None] * off[A] * q[None, :]
    ai, aj = np.nonzero(launch)
    c0 = np.cumsum(launch[ai, aj]); c0 /= c0[-1]
    inB = np.zeros(n, bool); inB[B] = True
    last = {}
    net = np.zeros((N_BATCH, n, n))
    for p in range(n_paths):
        k = int(np.searchsorted(c0, rng.random()))
        seg = [int(A[ai[k]]), int(aj[k])]
        s = seg[-1]
        while not inB[s]:
            s = int(dest[s][np.searchsorted(cdf[s], rng.random())])
            seg.append(s)
        seg = np.array(seg)
        cross = np.flatnonzero(S[seg[:-1]] & ~S[seg[1:]])
        e = (int(seg[cross[-1]]), int(seg[cross[-1] + 1]))
        last[e] = last.get(e, 0) + 1
        np.add.at(net[p * N_BATCH // n_paths], (seg[:-1], seg[1:]), 1.0)
    return last, net


def hist_on_cut(last, ref):
    lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(ref['i'], ref['j']))}
    h = np.zeros(len(lookup))
    for e, c in last.items():
        h[lookup[e]] += c
    return h / h.sum()


def jhat_from_counts(net):
    tot = net.sum(axis=0)
    J = np.maximum(tot - tot.T, 0.0)
    return J / J.sum()


def edge_current_test(net, Kb, pib, qb, n_paths):
    """Expected net traversals per reactive path on edge i->j equal J_ij/F with
    J_ij = pi_i K_ij (q_j - q_i) and F the reactive flux. Batch-means z-scores on
    the N_TOP_EDGES edges carrying the largest exact current."""
    J = pib[:, None] * Kb * (qb[None, :] - qb[:, None])
    np.fill_diagonal(J, 0.0)
    F = float(np.maximum(J, 0.0).sum(axis=1)[qb == 0].sum())
    expect = J / F
    per = (net - np.transpose(net, (0, 2, 1))) / (n_paths / net.shape[0])
    flat = np.argsort(-np.maximum(J, 0.0), axis=None)[:N_TOP_EDGES]
    ii, jj = np.unravel_index(flat, J.shape)
    mean = per[:, ii, jj].mean(axis=0)
    se = per[:, ii, jj].std(axis=0, ddof=1) / np.sqrt(net.shape[0])
    z = (mean - expect[ii, jj]) / se
    return dict(n_edges=N_TOP_EDGES, n_batches=int(net.shape[0]), max_abs_z=float(np.abs(z).max()),
                rms_z=float(np.sqrt(np.mean(z ** 2))),
                max_rel_err=float(np.max(np.abs(mean - expect[ii, jj]) / expect[ii, jj])),
                share_of_positive_current=float(np.maximum(J, 0)[ii, jj].sum() / np.maximum(J, 0).sum()))


def run_case(sys_, b, rng, label, mode):
    Kb = L.tilt_generator(sys_.K0, b)
    pib = L.biased_pi_from_reference(sys_.pi0, b)
    qb = L.committor_K(Kb, sys_.A, sys_.B)
    S = sys_.q0 < 0.5
    exact = sys_.metrics(b)
    pe = np.array(exact['last_crossing_probability'])
    Je = MA.positive_net_current(Kb, pib, qb)
    if mode == 'first_passage':
        times, last, net = first_passage(Kb, sys_.A, sys_.B, S, rng, N_PATHS)
    else:
        times, _, _ = first_passage(Kb, sys_.A, sys_.B, S, rng, N_PATHS, record_reactive=False)
        last, net = doob_paths(Kb, pib, sys_.A, sys_.B, S, rng, N_PATHS)
    ps = hist_on_cut(last, sys_.ref)
    Js = jhat_from_counts(net)
    edge_test = edge_current_test(net, Kb, pib, qb, N_PATHS)
    out = dict(
        n_paths=N_PATHS, mode=mode,
        mfpt_AB_exact=exact['mfpt_AB'], mfpt_AB_MC=float(times.mean()),
        mfpt_AB_MC_se=float(times.std(ddof=1) / np.sqrt(N_PATHS)),
        mfpt_z=float((times.mean() - exact['mfpt_AB']) / (times.std(ddof=1) / np.sqrt(N_PATHS))),
        last_crossing_TV_sampled_vs_exact=float(0.5 * np.abs(ps - pe).sum()),
        C_TS_exact=exact['C_TS'],
        C_TS_exact_ref_vs_sampled_bias=float(np.minimum(sys_.p0, ps).sum()),
        current_TV_sampled_vs_exact=float(0.5 * np.abs(Js - Je).sum()),
        Omega_J_exact=exact['Omega_J'],
        Omega_J_exact_ref_vs_sampled_bias=float(np.minimum(MA.positive_net_current(
            sys_.K0, sys_.pi0, sys_.q0), Js).sum()),
        current_TV_sampled_vs_exact_unbiased=float(0.5 * np.abs(Js - MA.positive_net_current(
            sys_.K0, sys_.pi0, sys_.q0)).sum()),
        edge_current_test=edge_test,
    )
    print(label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    return out, ps, Js


def main():
    G = generators()
    S_ = ab_sets()
    K = G['K']
    sysP = C.System(K, *S_['primary'])
    sysA = C.System(K, *S_['alternative'])
    biases = np.load(C.HERE / 'optima_biases.npz')
    rng = np.random.default_rng(20261011)
    res = dict(note=__doc__.strip().splitlines()[0], primary={}, alternative={})
    zero = np.zeros(K.shape[0])
    cases = [('unbiased', zero)] + [(k, biases[k]) for k in
                                    ('spectral_U1', 'spectral_U2', 'spectral_U3',
                                     'mfpt_AB_U1', 'mfpt_AB_U2', 'mfpt_AB_U3')]
    store = {}
    for name, b in cases:
        res['primary'][name], store[name + '_p'], store[name + '_J'] = run_case(sysP, b, rng, name, 'first_passage')
    # Replicate of the unbiased sample: two-sample noise level for C_TS and Omega_J
    _, p_rep, J_rep = run_case(sysP, zero, rng, 'unbiased_replicate', 'first_passage')
    res['primary']['two_sample_noise_unbiased'] = dict(
        C_TS_two_samples=float(np.minimum(store['unbiased_p'], p_rep).sum()),
        Omega_J_two_samples=float(np.minimum(store['unbiased_J'], J_rep).sum()))
    for name, b in [('unbiased', zero), ('spectral_U2', biases['spectral_U2']),
                    ('mfpt_AB_alternative_U2', biases['mfpt_AB_alternative_U2'])]:
        res['alternative'][name], _, _ = run_case(sysA, b, rng, 'alt_' + name, 'doob')
    C.dump(C.HERE / 'sampling_check.json', res)
    print('saved')


if __name__ == '__main__':
    main()
