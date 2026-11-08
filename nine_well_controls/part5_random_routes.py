"""Part 5: routes that ignore the energy landscape, on the UNBIASED system ("random routes").

Compared with the true unbiased reactive current J0 and last-crossing law p0 (fixed cut S = {q0 < 1/2}):
 (i)  flat landscape: the same grid, same A and B, F = 0 (all nearest-neighbour rates 1). Its reactive
      current and its last-crossing law on the fixed cut, from law() of the TS audit.
 (ii) random lattice routes: loop-erased random walks (LERW) from A to B of the simple random walk on the
      same grid (each step to a uniformly chosen nearest neighbour; this is the jump chain of the F = 0
      network). A and B are each treated as one node (a return to any A cell restarts the route; the walk
      stops at the first B cell). Each route is self-avoiding and carries one unit of current along its
      edges. N routes are summed; Omega_J and C_TS of the sum against J0 and p0 are reported versus N
      (R sets per N, median and 5-95%).

Cross-checks:
 - (i) by two routes: the F = 0 generator directly, and the TS-audit tilt of K0 by b = -F.
 - Kirchhoff: the expected net edge traversal of the LERW equals the unit electrical current between A and
   B (unit conductances, A and B each shorted), which is the normalized reactive current of the F = 0
   network. Checked three ways: Laplacian solve, law() of the F = 0 network, and the large-N LERW average.
 - LERW last-crossing law by a second sampler: the A-to-B path in a uniform spanning tree drawn with
   Wilson's algorithm (random vertex order, A last), which has the same law as the LERW (Pemantle 1991).
All values are network-model results.
"""
from __future__ import annotations

import json
import time

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from common import HERE, systems, law, generator_grid, metrics, signed_current, overlap_signed, overlap_prob, to_jsonable
from pathstats import seeds_for, norm_signed, omega_rows, cts_rows

NS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]   # the route average has converged by 3e4
R = 100
SYS_ID = {'nine_well': 1, 'bottleneck_grid': 2}


def walk_tables(s):
    """Nearest-neighbour tables of the simple random walk, with edge ids, signs and cut ids."""
    K = s.K0.tocsr()
    cut_lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(s.ref['i'], s.ref['j']))}
    isA = np.zeros(s.n, np.bool_); isA[s.A] = True
    isB = np.zeros(s.n, np.bool_); isB[s.B] = True
    ptr, dst, eid, sgn, cut = [0], [], [], [], []
    for i in range(s.n):
        for j in K.indices[K.indptr[i]:K.indptr[i+1]]:
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            dst.append(j); eid.append(s.edge_index[(a, b)]); sgn.append(1 if i < j else -1)
            cut.append(cut_lookup[(i, j)] if (s.S[i] and not s.S[j]) else -1)
        ptr.append(len(dst))
    ptr = np.array(ptr, np.int64); dst = np.array(dst, np.int64)
    # launch edges: every grid edge from an A cell to a non-A cell, chosen uniformly
    lk = [k for i in s.A for k in range(ptr[i], ptr[i+1]) if not isA[dst[k]]]
    return dict(ptr=ptr, dst=dst, eid=np.array(eid, np.int64), sgn=np.array(sgn, np.int64),
                cut=np.array(cut, np.int64), launch=np.array(lk, np.int64), isA=isA, isB=isB,
                n=s.n, n_edges=s.n_edges, n_cut=len(s.p0))


@njit(inline='always')
def _step(cur, Anode, launch, ptr, dst, isA):
    if cur == Anode:
        k = launch[np.random.randint(0, launch.shape[0])]
    else:
        k = ptr[cur] + np.random.randint(0, ptr[cur+1] - ptr[cur])
    nxt = dst[k]
    if isA[nxt]:
        nxt = Anode
    return k, nxt


@njit(parallel=True)
def _lerw_sets(seeds, n_paths, ptr, dst, eid, sgn, cut, launch, isA, isB, n, n_edges, n_cut):
    n_sets = seeds.shape[0]
    net = np.zeros((n_sets, n_edges), np.int64)
    last = np.zeros((n_sets, n_cut), np.int64)
    length = np.zeros(n_sets, np.int64)
    walk = np.zeros(n_sets, np.int64)
    Anode = n
    for s in prange(n_sets):
        np.random.seed(seeds[s])
        pos = np.full(n+1, -1, np.int64)
        node = np.empty(n+2, np.int64)
        code = np.empty(n+2, np.int64)
        for p in range(n_paths):
            L = 0; node[0] = Anode; pos[Anode] = 0; cur = Anode
            while True:
                k, nxt = _step(cur, Anode, launch, ptr, dst, isA)
                walk[s] += 1
                if pos[nxt] >= 0:                       # loop: erase back to nxt (restart if nxt is A)
                    kk = pos[nxt]
                    for t in range(kk+1, L+1):
                        pos[node[t]] = -1
                    L = kk; cur = nxt
                    continue
                L += 1; node[L] = nxt; code[L-1] = k; pos[nxt] = L; cur = nxt
                if isB[nxt]:
                    break
            lc = -1
            for t in range(L):
                k = code[t]
                net[s, eid[k]] += sgn[k]
                if cut[k] >= 0:
                    lc = cut[k]
            last[s, lc] += 1
            length[s] += L
            for t in range(L+1):
                pos[node[t]] = -1
    return net, last, length, walk


@njit(parallel=True)
def _wilson_paths(seeds, others, ptr, dst, eid, sgn, cut, launch, isA, isB, n, n_edges, n_cut):
    """A-to-B path of a uniform spanning tree (Wilson's algorithm, B as root, A as one node, A last)."""
    m = seeds.shape[0]
    last = np.zeros((m, n_cut), np.int64)
    length = np.zeros(m, np.int64)
    Anode = n
    for s in prange(m):
        np.random.seed(seeds[s])
        in_tree = np.zeros(n+1, np.bool_)
        for i in range(n):
            if isB[i]:
                in_tree[i] = True
        nxt_node = np.empty(n+1, np.int64)
        nxt_code = np.empty(n+1, np.int64)
        order = np.random.permutation(others)
        for idx in range(order.shape[0]+1):
            u = order[idx] if idx < order.shape[0] else Anode
            v = u
            while not in_tree[v]:
                k, w = _step(v, Anode, launch, ptr, dst, isA)
                nxt_code[v] = k; nxt_node[v] = w
                v = w
            v = u
            while not in_tree[v]:
                in_tree[v] = True
                v = nxt_node[v]
        v = Anode; lc = -1; L = 0
        while v == Anode or not isB[v]:
            k = nxt_code[v]
            if cut[k] >= 0:
                lc = cut[k]
            v = nxt_node[v]; L += 1
        last[s, lc] = 1
        length[s] = L
    return last, length


@njit
def _wilson_net(seeds, others, ptr, dst, eid, sgn, cut, launch, isA, isB, n, n_edges):
    """Serial version that accumulates the net traversals of the Wilson A-to-B paths."""
    net = np.zeros(n_edges, np.int64)
    Anode = n
    for s in range(seeds.shape[0]):
        np.random.seed(seeds[s])
        in_tree = np.zeros(n+1, np.bool_)
        for i in range(n):
            if isB[i]:
                in_tree[i] = True
        nxt_node = np.empty(n+1, np.int64)
        nxt_code = np.empty(n+1, np.int64)
        order = np.random.permutation(others)
        for idx in range(order.shape[0]+1):
            u = order[idx] if idx < order.shape[0] else Anode
            v = u
            while not in_tree[v]:
                k, w = _step(v, Anode, launch, ptr, dst, isA)
                nxt_code[v] = k; nxt_node[v] = w
                v = w
            v = u
            while not in_tree[v]:
                in_tree[v] = True
                v = nxt_node[v]
        v = Anode
        while v == Anode or not isB[v]:
            k = nxt_code[v]
            net[eid[k]] += sgn[k]
            v = nxt_node[v]
    return net


def lerw_sets(tab, seeds, n_paths):
    return _lerw_sets(np.asarray(seeds, np.int64), int(n_paths), tab['ptr'], tab['dst'], tab['eid'], tab['sgn'],
                      tab['cut'], tab['launch'], tab['isA'], tab['isB'], tab['n'], tab['n_edges'], tab['n_cut'])


def kirchhoff_current(s):
    """Unit electrical current A -> B with unit conductances (A at 0, B at 1), signed per undirected edge."""
    n = s.n
    rows, cols = s.ea, s.eb
    W = csr_matrix((np.ones(2*len(rows)), (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n))
    Lap = csr_matrix(np.diag(np.asarray(W.sum(axis=1)).ravel())) - W
    fixed = np.r_[s.A, s.B]
    free = np.setdiff1d(np.arange(n), fixed)
    v = np.zeros(n); v[s.B] = 1.
    v[free] = spsolve(Lap[free][:, free].tocsc(), -Lap[free][:, s.B]@np.ones(len(s.B)))
    x = v[s.eb] - v[s.ea]                # current a -> b for unit conductance
    return x/np.abs(x).sum()


def band(v):
    v = np.asarray(v)
    return dict(median=float(np.median(v)), p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                mean=float(np.mean(v)))


def main():
    S = systems()
    out = dict(description='routes that ignore the landscape, compared with the true unbiased current and '
                           'last-crossing law; network-model results', N_values=NS, sets_per_N=R, systems={})
    t0 = time.time()
    raw = {}
    for name, s in S.items():
        res = {}
        # (i) flat landscape
        Kf = generator_grid(np.zeros(s.shape))
        pif = np.ones(s.n)/s.n
        lf = law(Kf, pif, s.A, s.B, s.q0)
        xf = signed_current(lf, s.ea, s.eb)
        mt = metrics(s, -s.F)                       # second route: tilt K0 by b = -F
        xk = kirchhoff_current(s)
        res['flat'] = dict(C_TS=overlap_prob(s.p0, lf['probability']), Omega_J=overlap_signed(s.J0, xf),
                           via_tilt=dict(C_TS=mt['C_TS'], Omega_J=mt['Omega_J'], mfpt_speedup=mt['mfpt_speedup'],
                                         ess_frac=mt['ess_frac'], max_abs_b=mt['max_abs_b']),
                           kirchhoff_vs_flat_law_max_abs=float(np.abs(xk - xf).max()),
                           probability=lf['probability'].tolist())
        # (ii) LERW route ensembles
        tab = walk_tables(s)
        rows = {}
        for N in NS:
            net, last, length, walk = lerw_sets(tab, seeds_for(50+SYS_ID[name], N, n=R), N)
            raw[f'{name}_N{N}_net'] = net.astype(np.int32); raw[f'{name}_N{N}_last'] = last.astype(np.int32)
            xs = norm_signed(net)
            rows[str(N)] = dict(Omega_J=band(omega_rows(xs, s.J0[None, :])),
                                C_TS=band(cts_rows(last, np.broadcast_to(s.p0, last.shape))),
                                Omega_J_vs_flat=band(omega_rows(xs, xf[None, :])),
                                mean_route_edges=float(length.sum()/(R*N)), mean_walk_steps=float(walk.sum()/(R*N)))
            print(name, 'LERW N', N, f"Omega_J {rows[str(N)]['Omega_J']['median']:.4f}",
                  f"C_TS {rows[str(N)]['C_TS']['median']:.4f}", f"vs flat {rows[str(N)]['Omega_J_vs_flat']['median']:.4f}",
                  f'{time.time()-t0:.0f}s', flush=True)
        res['lerw'] = rows
        # pooled LERW (all sets at the two largest N): large-N limits
        pooled_net = sum(raw[f'{name}_N{N}_net'].astype(np.int64).sum(0) for N in NS[-2:])
        pooled_last = sum(raw[f'{name}_N{N}_last'].astype(np.int64).sum(0) for N in NS[-2:])
        xp = pooled_net/np.abs(pooled_net).sum()
        pl = pooled_last/pooled_last.sum()
        res['lerw_pooled'] = dict(n_paths=int(pooled_last.sum()), Omega_J_vs_true=overlap_signed(s.J0, xp),
                                  C_TS_vs_true=overlap_prob(s.p0, pl), Omega_J_vs_flat=overlap_signed(xf, xp),
                                  Omega_J_vs_kirchhoff=overlap_signed(xk, xp),
                                  C_TS_vs_flat_law=overlap_prob(lf['probability'], pl),
                                  probability=pl.tolist())
        # Wilson's algorithm: independent sampler of the LERW law
        nW = 20000
        seedsW = seeds_for(60+SYS_ID[name], n=nW)
        others = np.flatnonzero(~tab['isA'] & ~tab['isB']).astype(np.int64)
        lastW, lengthW = _wilson_paths(seedsW, others, tab['ptr'], tab['dst'], tab['eid'], tab['sgn'], tab['cut'], tab['launch'],
                                       tab['isA'], tab['isB'], tab['n'], tab['n_edges'], tab['n_cut'])
        netW = _wilson_net(seedsW[:5000], others, tab['ptr'], tab['dst'], tab['eid'], tab['sgn'], tab['cut'], tab['launch'],
                           tab['isA'], tab['isB'], tab['n'], tab['n_edges'])
        pw = lastW.sum(0)/nW
        # two-sample chi-square between Wilson and pooled LERW last-crossing counts
        cW, cL = lastW.sum(0).astype(float), pooled_last.astype(float)
        tot = cW + cL
        keep = tot > 20
        eW = tot*cW.sum()/(cW.sum()+cL.sum()); eL = tot*cL.sum()/(cW.sum()+cL.sum())
        chi2 = float((((cW-eW)**2/eW + (cL-eL)**2/eL)[keep]).sum())
        res['wilson_check'] = dict(n_trees=nW, C_TS_wilson_vs_lerw=overlap_prob(pw, pl),
                                   chi2=chi2, dof=int(keep.sum()-1),
                                   Omega_J_wilson_vs_kirchhoff=overlap_signed(xk, netW/np.abs(netW).sum()),
                                   mean_route_edges_wilson=float(lengthW.mean()),
                                   mean_route_edges_lerw=float(res['lerw'][str(NS[-1])]['mean_route_edges']))
        print(name, 'flat', res['flat']['C_TS'], res['flat']['Omega_J'], 'pooled', res['lerw_pooled'],
              'wilson', res['wilson_check'], f'{time.time()-t0:.0f}s', flush=True)
        out['systems'][name] = res
    out['seconds'] = time.time()-t0
    (HERE/'results_part5.json').write_text(json.dumps(to_jsonable(out), indent=1)+'\n')
    np.savez_compressed(HERE/'part5_route_counts.npz', **raw)


if __name__ == '__main__':
    main()
