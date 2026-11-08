"""Approximate last-crossing overlap C_TS for Ras on the joint reconstructed MSM.
Reference committor q0 from the joint DHAM reference itself (as for the pentapeptide MSM). H = {q0 < 1/2} fixed.
For each model: pi, backward committor q-, and h_j = P_j(reach B before H) are recomputed, and the last-crossing
distribution on the fixed cut dH is p_ij ∝ pi_i q-_i K_ij h_j. Synthetic-run null as in joint_dham.null.
Usage: python cts_ras.py [lag] [nsyn]"""
import sys, json, time
import numpy as np
from scipy.linalg import solve
import joint_dham as jd

def last_crossing(K, A, B, H):
    J, pi, qp, qm = jd.current(K, A, B)
    n = len(K); h = np.zeros(n); h[B] = 1
    free = ~(H | B)
    Kn = K / np.max(-np.diag(K))
    h[free] = solve(Kn[np.ix_(free, free)], -Kn[np.ix_(free, B)].sum(1))
    L = pi[:, None] * qm[:, None] * K * h[None, :]
    mask = H[:, None] & (~H)[None, :]
    np.fill_diagonal(mask, False)
    L = np.where(mask & (K > 0), L, 0.0)
    return L / L.sum(), J

def models(P, b, A, B):
    """Reference generator and state-averaged biased generator from a full row-stochastic P (all states active)."""
    K = jd.generator(P)
    Kb = jd.generator(K * np.exp(-(b[None, :] - b[:, None]) / 2))
    return K, Kb

def raw_model(seqs, u, n, lag, frame_bias=None):
    """Self-consistent DHAM P and occupancy-weighted state bias b (in kT), all states required."""
    C = np.zeros((n, n)); nk = np.zeros((len(seqs), n)); V = np.zeros(n); occ = np.zeros(n)
    for k, s in enumerate(seqs):
        if len(s) <= lag: continue
        c = np.zeros((n, n)); np.add.at(c, (s[:-lag], s[lag:]), 1)
        uk = u[k]; C += c; nk[k] = c.sum(1)
        vb = uk[s] if frame_bias is None else frame_bias[k]
        np.add.at(V, s, vb); np.add.at(occ, s, 1)
    if ((C.sum(0) + C.sum(1)) == 0).any(): raise ValueError('inactive state')
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    if connected_components(csr_matrix(C > 0), directed=True, connection='strong')[0] != 1:
        raise ValueError('not strongly connected')
    P = jd.dham_P(C, nk, u[:len(seqs)])            # self-consistent DHAM
    return P, V / occ / jd.KT

def main(lag=1, nsyn=200, seed=20261107):
    t0 = time.time()
    S = jd.build_states(**jd.CONFIGS['w0.25_g3']); n = S['n']; A = S['region'] == 0; B = S['region'] == 2
    seqs = jd.observed_sequences(S); u = jd.state_bias(S, 'average')
    fb = [jd.gaussian(f[:, 2], p) for (_, _, f, p) in jd.PROPS]
    # real data (frame-level applied bias, as in Table 3)
    P, b = raw_model(seqs, u, n, lag, frame_bias=fb)
    K, Kb = models(P, b, A, B)
    q0 = jd.current(K, A, B)[2]; H = q0 < 0.5
    p0, J0 = last_crossing(K, A, B, H); pb, Jb = last_crossing(Kb, A, B, H)
    real = dict(C_TS=float(np.minimum(p0, pb).sum()), Omega_J=float(np.minimum(J0, Jb).sum()),
                H_states=int(H.sum()), cut_edges=int(((p0 > 0) | (pb > 0)).sum()))
    print('real', real, flush=True)
    # synthetic null from the u-based reference (as in joint_dham.null)
    P0, bu = raw_model(seqs, u, n, lag)
    K0, K0b = models(P0, bu, A, B)
    q00 = jd.current(K0, A, B)[2]; H0 = q00 < 0.5
    t0p, _ = last_crossing(K0, A, B, H0); t0b, _ = last_crossing(K0b, A, B, H0)
    truth = float(np.minimum(t0p, t0b).sum())
    rng = np.random.default_rng(seed)
    def Pk(k):
        M = P0 * np.exp(-(u[k][None, :] - u[k][:, None]) / (2 * jd.KT)); return M / M.sum(1, keepdims=True)
    syn = [[None] * len(jd.PROPS) for _ in range(nsyn)]; prev = -1
    for k, (r, _, f, _) in enumerate(jd.PROPS):
        cum = np.cumsum(Pk(k), 1); cum[:, -1] = 1.0; L = -(-len(f) // lag)
        state = np.full(nsyn, seqs[k][0]) if r != prev else (cum[state] < rng.random(nsyn)[:, None]).sum(1)
        prev = r; traj = np.empty((nsyn, L), int); traj[:, 0] = state
        for t in range(1, L):
            state = (cum[state] < rng.random(nsyn)[:, None]).sum(1); traj[:, t] = state
        for d in range(nsyn): syn[d][k] = traj[d]
    ps, vs, fails = [], [], 0
    for d in range(nsyn):
        try:
            Pd, bd = raw_model(syn[d], u, n, 1)
            Kd, Kdb = models(Pd, bd, A, B)
            pd_, _ = last_crossing(Kd, A, B, H0); pdb, _ = last_crossing(Kdb, A, B, H0)
            ps.append(pd_); vs.append(dict(vs_truth=float(np.minimum(pd_, t0p).sum()), ref_vs_biased=float(np.minimum(pd_, pdb).sum())))
        except Exception:
            fails += 1
    pair = [float(np.minimum(ps[2 * i], ps[2 * i + 1]).sum()) for i in range(len(ps) // 2)]
    pc = lambda v: np.percentile(v, [5, 50, 95]).round(3).tolist()
    out = dict(lag=lag, real=real, truth_C_TS_ref_vs_biased=truth, estimable=len(ps), failed=fails,
               C_TS_vs_truth_5_50_95=pc([x['vs_truth'] for x in vs]),
               C_TS_between_independent_5_50_95=pc(pair),
               C_TS_ref_vs_biased_synthetic_5_50_95=pc([x['ref_vs_biased'] for x in vs]), seconds=round(time.time() - t0))
    print(json.dumps(out), flush=True)
    p = jd.OUT / 'cts_results.json'
    allr = json.loads(p.read_text()) if p.exists() else {}
    allr[f'lag{lag}'] = out; p.write_text(json.dumps(allr, indent=1))

if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
