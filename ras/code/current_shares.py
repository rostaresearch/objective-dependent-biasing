"""Shares of the outgoing reactive current by P-Mg range (state centroids), 1 ps, grid w0.25_g3, as quoted in main 4.1."""
import numpy as np, json, joint_dham as jd
S = jd.build_states(**jd.CONFIGS['w0.25_g3']); seqs = jd.observed_sequences(S); u = jd.state_bias(S, 'average')
fb = [jd.gaussian(f[:, 2], p) for (_, _, f, p) in jd.PROPS]
out = {}
for norm in (False, True):
    r = jd.estimate(seqs, u, S['n'], S['region'], 1, frame_bias=fb, normalized=norm)
    row = {}
    for name, J in (('reference', r['J']), ('biased', r['Jb'])):
        v = J.sum(1); v = v / v.sum(); cx, cy = S['cx'], S['cy']
        row[name] = dict(A=v[cx < 5].sum(), mg5_6=v[(cx >= 5) & (cx < 6)].sum(), mg6_8=v[(cx >= 6) & (cx < 8)].sum(),
                         near_gln61_5_8=v[(cx >= 5) & (cx < 8) & (cy < 4.5)].sum())
        row[name] = {k: round(float(x), 3) for k, x in row[name].items()}
    out['self-consistent' if norm else 'plain'] = row; print('self-consistent' if norm else 'plain', row)
json.dump(out, open(jd.OUT / 'current_shares.json', 'w'), indent=1)
