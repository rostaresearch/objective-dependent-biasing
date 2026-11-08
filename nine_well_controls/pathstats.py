"""Reactive-path sampling and exact path moments (part 1).

Sampler: the jump chain conditioned to reach B before A (Doob h-transform with the
forward committor q of the sampled network), launched from A with weight
pi_a K_aj q_j. This is the sampler of common/validate_sampling.py,
reimplemented in numba so that ~10^8 paths are affordable. For every path it records
  - the last crossing of the fixed unbiased cut S = {q0 < 1/2} (edge i in S -> j not in S);
  - the net number of traversals of every undirected edge (a<b): +1 for a->b, -1 for b->a.
Each set of N paths is accumulated separately; sets have their own seeds.

Exact moments: per-path mean, variance and covariance of the net traversal counts,
computed by linear solves on the same conditioned chain. These give a large-N
(central-limit) prediction of the two-sample noise in Omega_J that uses no sampling.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy.linalg import lu_factor, lu_solve

from common import harmonic, tilt


def tables(sys_, b):
    """Transition tables of the conditioned jump chain for bias b."""
    Kb = tilt(sys_.K0, b)
    pib = sys_.bias_pi(b)
    q = harmonic(Kb, sys_.A, sys_.B)
    dense = Kb.toarray()
    np.fill_diagonal(dense, 0.)
    n = sys_.n
    cut_lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(sys_.ref['i'], sys_.ref['j']))}
    S = sys_.S

    def edge_of(i, j):
        a, c = (i, j) if i < j else (j, i)
        return sys_.edge_index[(a, c)], (1 if i < j else -1)

    def cut_of(i, j):
        return cut_lookup[(i, j)] if (S[i] and not S[j]) else -1

    launch = pib[sys_.A, None]*dense[sys_.A]*q[None, :]
    ai, lj = np.nonzero(launch)
    lw = launch[ai, lj]
    lsrc = sys_.A[ai]
    l_eid, l_sgn, l_cut = [], [], []
    for i, j in zip(lsrc, lj):
        e, sg = edge_of(int(i), int(j)); l_eid.append(e); l_sgn.append(sg); l_cut.append(cut_of(int(i), int(j)))
    ptr = [0]; dest = []; cdf = []; eid = []; sgn = []; cut = []; prob = []
    for i in range(n):
        d = np.flatnonzero((dense[i] > 0) & (q > 0))
        w = dense[i, d]*q[d]
        if len(d):
            c = np.cumsum(w); c /= c[-1]; c[-1] = 1.
            p = w/w.sum()
        else:
            c = np.zeros(0); p = np.zeros(0)
        for j, cc, pp in zip(d, c, p):
            e, sg = edge_of(i, int(j))
            dest.append(int(j)); cdf.append(cc); eid.append(e); sgn.append(sg); cut.append(cut_of(i, int(j))); prob.append(pp)
        ptr.append(len(dest))
    isB = np.zeros(n, np.bool_); isB[sys_.B] = True
    lc = np.cumsum(lw); lc /= lc[-1]; lc[-1] = 1.
    return dict(q=q, launch_cdf=lc, launch_w=lw/lw.sum(), launch_src=lsrc.astype(np.int64),
                launch_dst=lj.astype(np.int64), launch_eid=np.array(l_eid, np.int64),
                launch_sgn=np.array(l_sgn, np.int64), launch_cut=np.array(l_cut, np.int64),
                ptr=np.array(ptr, np.int64), dest=np.array(dest, np.int64), cdf=np.array(cdf),
                prob=np.array(prob), eid=np.array(eid, np.int64), sgn=np.array(sgn, np.int64),
                cut=np.array(cut, np.int64), isB=isB, n_edges=sys_.n_edges, n_cut=len(sys_.p0))


@njit(parallel=True, cache=False)
def _sample_sets(seeds, n_paths, launch_cdf, launch_dst, launch_eid, launch_sgn, launch_cut,
                 ptr, dest, cdf, eid, sgn, cut, isB, n_edges, n_cut, max_steps):
    n_sets = seeds.shape[0]
    net = np.zeros((n_sets, n_edges), np.int64)
    last = np.zeros((n_sets, n_cut), np.int64)
    steps = np.zeros(n_sets, np.int64)
    bad = np.zeros(n_sets, np.int64)
    for s in prange(n_sets):
        np.random.seed(seeds[s])
        for p in range(n_paths):
            u = np.random.random()
            k = np.searchsorted(launch_cdf, u)
            lc = launch_cut[k]
            net[s, launch_eid[k]] += launch_sgn[k]
            cur = launch_dst[k]
            nst = 0
            while not isB[cur]:
                u = np.random.random()
                m = ptr[cur]
                end = ptr[cur+1]-1
                while m < end and cdf[m] < u:
                    m += 1
                net[s, eid[m]] += sgn[m]
                if cut[m] >= 0:
                    lc = cut[m]
                cur = dest[m]
                nst += 1
                if nst > max_steps:
                    bad[s] += 1
                    break
            if lc < 0:
                bad[s] += 1
            else:
                last[s, lc] += 1
            steps[s] += nst
    return net, last, steps, bad


def sample_sets(tab, seeds, n_paths, max_steps=10_000_000):
    net, last, steps, bad = _sample_sets(
        np.asarray(seeds, np.int64), int(n_paths), tab['launch_cdf'], tab['launch_dst'], tab['launch_eid'],
        tab['launch_sgn'], tab['launch_cut'], tab['ptr'], tab['dest'], tab['cdf'], tab['eid'], tab['sgn'],
        tab['cut'], tab['isB'], tab['n_edges'], tab['n_cut'], max_steps)
    if bad.any():
        raise RuntimeError(f'{bad.sum()} paths failed (too long or no cut crossing)')
    return net, last, steps


def seeds_for(*key, n):
    return np.random.SeedSequence([20261010, *key]).generate_state(n, dtype=np.uint32).astype(np.int64)


# ---------- overlaps of sampled histograms -------------------------------------------

def norm_signed(net):
    net = np.asarray(net, float)
    return net/np.abs(net).sum(axis=-1, keepdims=True)


def omega_rows(x, y):
    """Omega_J row-wise between signed normalized vectors (broadcasts)."""
    return 1 - .5*np.abs(x - y).sum(axis=-1)


def cts_rows(c1, c2):
    p1 = c1/c1.sum(axis=-1, keepdims=True); p2 = c2/c2.sum(axis=-1, keepdims=True)
    return np.minimum(p1, p2).sum(axis=-1)


# ---------- exact per-path moments of the net traversal counts ------------------------

def path_moments(sys_, tab, sigma):
    """Exact per-path mean/second moments of net edge traversal counts N_e of the conditioned chain.

    sigma: +-1 per edge, the sign of the exact net current; T = sum_e sigma_e N_e is the
    per-path contribution to the normalization sum_e |sum_paths N_e| at large N.
    Returns mean N_e, Var N_e, Cov(N_e, T), mean T, Var T, mean jumps.
    """
    n, E = sys_.n, sys_.n_edges
    ptr, dest, prob, eid, sgn = tab['ptr'], tab['dest'], tab['prob'], tab['eid'], tab['sgn']
    rows = np.repeat(np.arange(n), np.diff(ptr))
    trans = np.isin(np.arange(n), np.r_[sys_.A, sys_.B], invert=True)    # transient states
    T_idx = np.flatnonzero(trans)
    P = np.zeros((n, n)); P[rows, dest] = prob
    P[~trans] = 0.                     # rows of A are not used (launch handled separately), B absorbing
    M = np.eye(len(T_idx)) - P[np.ix_(T_idx, T_idx)]
    lu = lu_factor(M)

    def solve(rhs):                    # rhs: n x k, returns n x k with zeros off T
        out = np.zeros_like(rhs)
        out[T_idx] = lu_solve(lu, rhs[T_idx])
        return out

    keep = trans[rows]
    r_, d_, p_, e_, s_ = rows[keep], dest[keep], prob[keep], eid[keep], sgn[keep]
    # first moments: reward sgn on edge eid
    R1 = np.zeros((n, E)); np.add.at(R1, (r_, e_), p_*s_)
    m = solve(R1)
    rT = sigma[e_]*s_                  # T reward on each transition
    R1T = np.zeros((n, 1)); np.add.at(R1T[:, 0], r_, p_*rT)
    mT = solve(R1T)[:, 0]
    R1J = np.zeros((n, 1)); np.add.at(R1J[:, 0], r_, p_)
    mJ = solve(R1J)[:, 0]              # expected number of jumps
    # second moments of N_e
    Q2 = np.zeros((n, E)); np.add.at(Q2, (r_, e_), p_*(1 + 2*s_*m[d_, e_]))
    s2 = solve(Q2)
    # cross moments E[N_e T]
    PRT = np.zeros((n, n)); np.add.at(PRT, (r_, d_), p_*rT)
    Qc = PRT@m
    np.add.at(Qc, (r_, e_), p_*(sigma[e_] + s_*mT[d_]))
    c = solve(Qc)
    # E[T^2]
    QT = np.zeros((n, 1)); np.add.at(QT[:, 0], r_, p_*(rT*rT + 2*rT*mT[d_]))
    sT = solve(QT)[:, 0]
    # launch step
    w, j = tab['launch_w'], tab['launch_dst']
    le, ls = tab['launch_eid'], tab['launch_sgn']
    re = np.zeros((len(w), E)); re[np.arange(len(w)), le] = ls
    rTl = sigma[le]*ls
    EN = w@(re + m[j])
    EN2 = w@(re*re + 2*re*m[j] + s2[j])
    ET = float(w@(rTl + mT[j]))
    ET2 = float(w@(rTl*rTl + 2*rTl*mT[j] + sT[j]))
    ENT = w@(re*rTl[:, None] + re*mT[j][:, None] + rTl[:, None]*m[j] + c[j])
    jumps = float(w@(1 + mJ[j]))
    return dict(mean=EN, var=EN2 - EN**2, cov_T=ENT - EN*ET, mean_T=ET, var_T=ET2 - ET**2, mean_jumps=jumps)


def clt_two_sample_omega(mom, N):
    """Large-N mean of 1-Omega_J between two independent sets of N paths (delta method)."""
    x, T = mom['mean'], mom['mean_T']
    v = mom['var']/T**2 - 2*x*mom['cov_T']/T**3 + x**2*mom['var_T']/T**4
    v = np.maximum(v, 0)
    return float(np.sum(np.sqrt(v/(np.pi*N))))


def clt_two_sample_cts(p, N):
    """Large-N mean of 1-C_TS between two independent multinomial histograms of size N."""
    p = np.asarray(p)
    return float(np.sum(np.sqrt(p*(1-p)/(np.pi*N))))
