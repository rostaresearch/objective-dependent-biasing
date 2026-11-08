"""Ras joint reconstruction (7 Nov 2026; self-consistent DHAM from 7 Nov evening).

Joint data-driven discretization of all 73 Ras runs, a joint directed DHAM reference, and a
synthetic-run null: biased runs are simulated from the joint reference with each propagation's actual
Gaussian bias and the real propagation lengths, then re-estimated with the same DHAM pipeline.

The bias depends only on the P-Mg distance, so DHAM accuracy depends only on the P-Mg width of a state.
States are narrow P-Mg slabs (region-wise, exact cuts at 5 and 8 A), each split along the Gln61 distance
by pooled quantiles where the data allow it. Sparse states and states outside the strongly connected
core (lags 1 and 2) are merged into their nearest neighbour in the same region (A, transition, B).

Usage: python joint_dham.py real            -> real-data overlaps for all configurations
       python joint_dham.py null CONFIG NSYN -> synthetic null for one configuration
"""
import sys, json, glob, time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

RAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAS / 'code'))
from current_analysis import current, generator          # same TPT/current solver as the paper
from extract_periodic import gaussian
from analyse_directed import KT

OUT = RAS / 'results'
RECORDS = [dict(np.load(p, allow_pickle=False)) for p in sorted((RAS / 'features').glob('*.npz'))]
assert len(RECORDS) == 73

# ---------------- propagation table ----------------
def propagations():
    """List of (run, k, frames, params) for every propagation with at least 2 frames, in time order."""
    out = []
    for r, z in enumerate(RECORDS):
        f = z['frames']
        for k, p in enumerate(z['parameters']):
            x = f[f[:, 1] == k]
            if len(x) >= 2: out.append((r, k, x, p))
    return out
PROPS = propagations()
ALLF = np.concatenate([p[2] for p in PROPS])

# ---------------- discretization ----------------
def region_edges(lo, hi, w):
    n = max(1, int(round((hi - lo) / w)))
    return np.linspace(lo, hi, n + 1)

def build_states(w, gmax, gmin, min_frames=30, lags=(1, 2)):
    x, y = ALLF[:, 2], ALLF[:, 3]
    xe = np.unique(np.r_[region_edges(x.min() - 1e-9, 5.0, w), region_edges(5.0, 8.0, w),
                         region_edges(8.0, x.max() + 1e-9, max(w, (x.max() - 8) / 2))])
    nsl = len(xe) - 1
    def slab(v): return np.clip(np.searchsorted(xe, v, side='right') - 1, 0, nsl - 1)
    ye = []
    for s in range(nsl):
        m = slab(x) == s; n = m.sum()
        g = int(np.clip(n // gmin, 1, gmax)) if n else 1
        ye.append(np.quantile(y[m], np.linspace(0, 1, g + 1)[1:-1]) if g > 1 else np.array([]))
    off = np.cumsum([0] + [len(e) + 1 for e in ye])
    def cell(v, u):
        s = slab(v); c = np.empty(len(v), int)
        for t in np.unique(s):
            m = s == t; c[m] = off[t] + np.searchsorted(ye[t], u[m], side='right')
        return c
    ncell = off[-1]
    lo = np.array([xe[s] for s in range(nsl) for _ in range(len(ye[s]) + 1)])
    reg = np.where(lo < 5 - 1e-9, 0, np.where(lo >= 8 - 1e-9, 2, 1))
    lab = np.arange(ncell)                        # cell -> state label (merged)
    cf = cell(x, y)
    def stats():
        st = lab[cf]; ids = np.unique(st)
        cnt = {i: int((st == i).sum()) for i in ids}
        cen = {i: (x[st == i].mean(), y[st == i].mean()) for i in ids}
        return st, ids, cnt, cen
    def nearest(i, cands, cen):
        d = [np.hypot(cen[i][0] - cen[j][0], cen[i][1] - cen[j][1]) for j in cands]
        return cands[int(np.argmin(d))]
    for it in range(200):
        st, ids, cnt, cen = stats()
        small = [i for i in ids if cnt[i] < min_frames]
        if small:
            i = min(small, key=lambda s: cnt[s])
            cands = [j for j in ids if j != i and reg[j] == reg[i]] or [j for j in ids if j != i]
            lab[lab == i] = nearest(i, cands, cen); continue
        bad = set()
        for lag in lags:
            C = counts_by_label(cf, lab, lag)
            active = np.flatnonzero((C.sum(0) + C.sum(1)) > 0)
            n, comp = connected_components(csr_matrix(C[np.ix_(active, active)] > 0), directed=True, connection='strong')
            if n > 1:
                sizes = np.bincount(comp, weights=[cnt.get(a, 0) for a in active])
                main = np.argmax(sizes)
                bad |= {int(a) for a, c in zip(active, comp) if c != main}
        if not bad: break
        core = [j for j in ids if j not in bad]
        i = min(bad, key=lambda s: cnt[s])
        cands = [j for j in core if reg[j] == reg[i]] or core
        lab[lab == i] = nearest(i, cands, cen)
    else:
        raise RuntimeError('merging did not converge')
    st, ids, cnt, cen = stats()
    remap = {i: k for k, i in enumerate(ids)}
    lab = np.array([remap[l] if l in remap else -1 for l in lab])
    # cells never visited keep -1 only if they had no frames; map them to nearest visited cell's state
    return dict(xe=xe, ye=ye, off=off, lab=lab, cell=cell, region=np.array([reg[i] for i in ids]),
                cx=np.array([cen[i][0] for i in ids]), cy=np.array([cen[i][1] for i in ids]),
                counts=np.array([cnt[i] for i in ids]), n=len(ids))

def counts_by_label(cf, lab, lag):
    """Directed counts on merged labels for the pooled data (never across propagation boundaries)."""
    n = lab.max() + 1; C = np.zeros((n, n)); pos = 0
    for (_, _, f, _) in PROPS:
        L = len(f); s = lab[cf[pos:pos + L]]; pos += L
        if L > lag: np.add.at(C, (s[:-lag], s[lag:]), 1)
    return C

def grid_states(es, ef):
    """The published 14 x 8 grid (bias at P-Mg bin centres), for validation."""
    ns, nf = len(es) - 1, len(ef) - 1
    def cell(v, u):
        return np.clip(np.searchsorted(es, v, side='right') - 1, 0, ns - 1) * nf + \
               np.clip(np.searchsorted(ef, u, side='right') - 1, 0, nf - 1)
    cx = np.repeat((es[:-1] + es[1:]) / 2, nf)
    reg = np.where(cx < 5, 0, np.where(cx >= 8, 2, 1))
    return dict(cell=cell, lab=np.arange(ns * nf), cx=cx, region=reg, n=ns * nf, grid=True)

# ---------------- per-propagation data on a discretization ----------------
def state_bias(S, mode):
    """u[k, i]: bias (kcal/mol) of state i during propagation k. 'centroid' uses the state's mean P-Mg
    distance (bin centre for the grid); 'average' uses -kT log <exp(-V/kT)> over pooled frames in the state."""
    P = np.array([p[3] for p in PROPS])
    if mode == 'centroid' or S.get('grid'):
        return np.array([gaussian(S['cx'], p) for p in P])
    st = S['lab'][S['cell'](ALLF[:, 2], ALLF[:, 3])]
    rng = np.random.default_rng(1)
    u = np.zeros((len(P), S['n']))
    for i in range(S['n']):
        xs = ALLF[st == i, 2]
        if len(xs) > 300: xs = rng.choice(xs, 300, replace=False)
        V = np.stack([gaussian(xs, p) for p in P])
        u[:, i] = -KT * np.log(np.mean(np.exp(-(V - V.min(1, keepdims=True)) / KT), 1)) + V.min(1)
    return u

def dham_P(C, nk, u, normalized=True, iters=5000, tol=1e-10):
    """Directed DHAM transition matrix from pooled counts C (n,n), per-propagation visit counts nk (K,n) and
    state biases u (K,n, kcal/mol). With phi^k_ij = exp(-(u^k_j-u^k_i)/2kT) = g^k_i h^k_j:
    plain DHAM      P_ij ∝ C_ij / sum_k n^k_i phi^k_ij;
    self-consistent P_ij ∝ C_ij / sum_k n^k_i phi^k_ij / Z^k_i, Z^k_i = sum_l P_il phi^k_il, i.e. the maximum-likelihood
    estimate for biased matrices P^k_ij = P_ij phi^k_ij / Z^k_i (row-normalized); plain DHAM is its first iterate (Z = 1).
    The fixed point is found with SQUAREM-accelerated iteration (Varadhan and Roland 2008)."""
    g = np.exp(u / (2 * KT)); h = 1.0 / g; ng = nk * g
    def solve(w):
        D = w.T @ h
        P = np.divide(C, D, out=np.zeros_like(C), where=D > 0)
        rs = P.sum(1, keepdims=True); return np.divide(P, rs, out=np.zeros_like(P), where=rs > 0)
    P = solve(ng)
    if not normalized: return P
    def F(P):
        Z = g * (h @ P.T)                                   # Z[k, i] = sum_l P_il phi^k_il
        return solve(np.divide(ng, Z, out=np.zeros_like(Z), where=Z > 0))
    best, bestP = np.inf, None
    for it in range(iters):
        P1 = F(P); r = P1 - P; res = np.abs(r).max()
        if res < tol: return P1
        if res < best: best, bestP = res, P1
        P2 = F(P1); v = P2 - P1 - r
        nv = np.sqrt((v * v).sum())
        a = -np.sqrt((r * r).sum()) / nv if nv > 0 else -1.0
        a = min(a, -1.0)
        Pa = P - 2 * a * r + a * a * v
        if (Pa < 0).any() or not np.isfinite(Pa).all(): Pa = P2
        P = F(Pa)
    if best < 1e-8: return bestP                            # round-off stagnation just above tol
    raise RuntimeError('self-consistent DHAM did not converge')

def observed_sequences(S):
    return [S['lab'][S['cell'](f[:, 2], f[:, 3])] for (_, _, f, _) in PROPS]

def estimate(seqs, u, n, region, lag=1, frame_bias=None, normalized=True):
    """Directed DHAM with propagation-specific denominators (self-consistent by default), then reference and
    biased currents."""
    C = np.zeros((n, n)); nk = np.zeros((len(seqs), n)); X = np.zeros((n, n)); V = np.zeros(n); occ = np.zeros(n)
    for k, s in enumerate(seqs):
        if len(s) <= lag: continue
        c = np.zeros((n, n)); np.add.at(c, (s[:-lag], s[lag:]), 1)
        uk = u[k]
        C += c; nk[k] = c.sum(1)
        vb = uk[s] if frame_bias is None else frame_bias[k]
        np.add.at(V, s, vb); np.add.at(occ, s, 1)
        dx = (vb[lag:] - vb[:-lag]) / KT
        np.add.at(X, (s[:-lag], s[lag:]), dx)
    active = (C.sum(0) + C.sum(1)) > 0; ix = np.flatnonzero(active)
    c = C[np.ix_(ix, ix)]
    ncomp, _ = connected_components(csr_matrix(c > 0), directed=True, connection='strong')
    if ncomp != 1: raise ValueError('not strongly connected')
    A = region[ix] == 0; B = region[ix] == 2
    if not A.any() or not B.any(): raise ValueError('empty A/B')
    P = dham_P(C[np.ix_(ix, ix)], nk[:, ix], u[:len(seqs), ix], normalized)
    K = generator(P)
    b = np.divide(V, occ, out=np.zeros_like(V), where=occ > 0)[ix] / KT
    J, pi, q, qm = current(K, A, B)
    Jb = current(generator(K * np.exp(-(b[None, :] - b[:, None]) / 2)), A, B)[0]
    Xm = np.divide(X, C, out=np.zeros_like(X), where=C > 0)[np.ix_(ix, ix)]
    Jr = current(generator(K * np.exp(-Xm / 2)), A, B)[0]
    full = lambda M: (lambda F: (F.__setitem__(np.ix_(ix, ix), M), F)[1])(np.zeros((n, n)))
    return dict(omega_state=float(np.minimum(J, Jb).sum()), omega_edge=float(np.minimum(J, Jr).sum()),
                states=int(len(ix)), A=int(A.sum()), B=int(B.sum()), P=full(P), J=full(J), Jb=full(Jb),
                escape_median=float(np.median(1 - np.diag(P))))

# ---------------- configurations ----------------
CONFIGS = {f'w{w}_g{g}': dict(w=w, gmax=g, gmin=2000) for w in (0.2, 0.25, 0.33, 0.5) for g in (1, 3)}

def real():
    out = {}
    grid = np.load(RAS / 'directed_pre0_lag1.npz')
    S = grid_states(grid['s_edges'], grid['feature_edges'])
    fb = [gaussian(f[:, 2], p) for (_, _, f, p) in PROPS]
    seqs = observed_sequences(S); u = state_bias(S, 'centroid')
    for lag in (1, 2):
        for norm in (False, True):
            r = estimate(seqs, u, S['n'], S['region'], lag, frame_bias=fb, normalized=norm)
            key = f'published_grid_lag{lag}' + ('' if norm else '_plainDHAM')
            out[key] = {k: v for k, v in r.items() if k not in ('P', 'J', 'Jb')}
            print(key, out[key], flush=True)
    for name, cfg in CONFIGS.items():
        S = build_states(**cfg); seqs = observed_sequences(S)
        for mode in ('centroid', 'average'):
            u = state_bias(S, mode)
            for lag in (1, 2):
                key = f'{name}_{mode}_lag{lag}'
                try:
                    r = estimate(seqs, u, S['n'], S['region'], lag, frame_bias=fb)
                    out[key] = {k: v for k, v in r.items() if k not in ('P', 'J', 'Jb')}
                except Exception as e:
                    out[key] = dict(error=str(e))
                out[key]['region_states'] = np.bincount(S['region'], minlength=3).tolist()
                print(key, out[key], flush=True)
        (OUT / 'real_results.json').write_text(json.dumps(out, indent=1))

def null(name, nsyn=200, mode='average', lag=1, seed=20261107):
    lag = int(lag)
    """Simulate nsyn synthetic 73-run datasets from the joint reference and re-estimate each."""
    t0 = time.time()
    S = build_states(**CONFIGS[name]); n = S['n']
    seqs = observed_sequences(S); u = state_bias(S, mode)
    ref = estimate(seqs, u, n, S['region'], lag)
    P0 = ref['P']; J0 = ref['J']
    vis = P0.sum(1) > 0
    rng = np.random.default_rng(seed)
    # biased one-step matrices per propagation: P0_ij exp(-(u_j-u_i)/2kT), row-normalized (the model that the
    # self-consistent DHAM inverts exactly)
    def Pk(k):
        M = P0 * np.exp(-(u[k][None, :] - u[k][:, None]) / (2 * KT))
        rs = M.sum(1, keepdims=True); return np.divide(M, rs, out=np.zeros_like(M), where=rs > 0)
    syn = [[None] * len(PROPS) for _ in range(nsyn)]
    state = None; prev_run = -1
    for k, (r, _, f, _) in enumerate(PROPS):
        cum = np.cumsum(Pk(k), 1); cum[:, -1] = 1.0
        L = -(-len(f) // lag)   # synthetic steps of one lag each
        if r != prev_run:
            state = np.full(nsyn, seqs[k][0]); prev_run = r
        else:
            state = (cum[state] < rng.random(nsyn)[:, None]).sum(1)
        traj = np.empty((nsyn, L), int); traj[:, 0] = state
        for t in range(1, L):
            state = (cum[state] < rng.random(nsyn)[:, None]).sum(1); traj[:, t] = state
        for d in range(nsyn): syn[d][k] = traj[d]
    print(name, 'simulated', round(time.time() - t0), 's', flush=True)
    res, Js, fails = [], [], {}
    for d in range(nsyn):
        try:
            r = estimate(syn[d], u, n, S['region'], 1)
            res.append(dict(omega_state=r['omega_state'], omega_edge=r['omega_edge'],
                            omega_vs_truth=float(np.minimum(r['J'], J0).sum())))
            Js.append(r['J'])
        except Exception as e:
            fails[str(e)] = fails.get(str(e), 0) + 1
    pair = [float(np.minimum(Js[2 * i], Js[2 * i + 1]).sum()) for i in range(len(Js) // 2)]
    np.savez(OUT / f'null_raw_{name}_{mode}_lag{lag}.npz', omega_vs_truth=[x['omega_vs_truth'] for x in res],
             omega_between=pair, omega_state=[x['omega_state'] for x in res], truth_omega_state=ref['omega_state'])
    pc = lambda v: np.percentile(v, [5, 50, 95]).round(3).tolist() if len(v) else None
    out = dict(config=name, mode=mode, lag=lag, states=n, region_states=np.bincount(S['region'], minlength=3).tolist(),
               truth_omega_state=ref['omega_state'], truth_omega_edge=ref['omega_edge'],
               nsyn=nsyn, estimable=len(res), failures=fails,
               omega_vs_truth_5_50_95=pc([x['omega_vs_truth'] for x in res]),
               omega_between_independent_synthetic_5_50_95=pc(pair),
               omega_state_synthetic_5_50_95=pc([x['omega_state'] for x in res]),
               omega_edge_synthetic_5_50_95=pc([x['omega_edge'] for x in res]),
               seconds=round(time.time() - t0))
    print(json.dumps(out), flush=True)
    p = OUT / 'null_results.json'
    allr = json.loads(p.read_text()) if p.exists() else {}
    allr[f'{name}_{mode}' + ('' if lag == 1 else f'_lag{lag}')] = out; p.write_text(json.dumps(allr, indent=1))

if __name__ == '__main__':
    if sys.argv[1] == 'real': real()
    else: null(sys.argv[2], int(sys.argv[3]), *(sys.argv[4:6] or ['average']))
