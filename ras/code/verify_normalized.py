"""Check of joint_dham.estimate (self-consistent DHAM): real-data overlap (plain vs self-consistent) and exact recovery of the
generating model from expected counts of the synthetic procedure, at lags 1 and 2."""
import numpy as np, joint_dham as jd
S = jd.build_states(**jd.CONFIGS['w0.25_g3']); n = S['n']; A = S['region'] == 0; B = S['region'] == 2
seqs = jd.observed_sequences(S); u = jd.state_bias(S, 'average')
fb = [jd.gaussian(f[:, 2], p) for (_, _, f, p) in jd.PROPS]
for lag in (1, 2):
    rp = jd.estimate(seqs, u, n, S['region'], lag, frame_bias=fb, normalized=False)
    rn = jd.estimate(seqs, u, n, S['region'], lag, frame_bias=fb)
    print(f'lag {lag}: Omega(ref,biased) plain {rp["omega_state"]:.3f}, self-consistent {rn["omega_state"]:.3f}, '
          f'Omega(ref_sc, ref_plain) {np.minimum(rp["J"], rn["J"]).sum():.3f}, escape median {rn["escape_median"]:.2f}')
    ref = jd.estimate(seqs, u, n, S['region'], lag); P0 = ref['P']; J0 = ref['J']
    phi = lambda k: np.exp(-(u[k][None, :] - u[k][:, None]) / (2 * jd.KT))
    Pk = lambda k: (lambda M: M / M.sum(1, keepdims=True))(P0 * phi(k))
    C = np.zeros((n, n)); nk = np.zeros((len(jd.PROPS), n)); prev = -1
    for k, (r, _, f, _) in enumerate(jd.PROPS):
        L = -(-len(f) // lag); Pkk = Pk(k)
        if r != prev: p = np.zeros(n); p[seqs[k][0]] = 1
        else: p = p @ Pkk
        prev = r; q = p.copy()
        for t in range(L - 1): nk[k] += q; q = q @ Pkk
        p = q; C += nk[k][:, None] * Pkk
    for norm in (False, True):
        Ph = jd.dham_P(C, nk, u, norm); Jh = jd.current(jd.generator(Ph), A, B)[0]
        print(f'   expected counts, {"self-consistent" if norm else "plain"}: max|dP| {np.abs(Ph-P0).max():.1e}, '
              f'Omega(est, generating) {np.minimum(Jh, J0).sum():.4f}')
