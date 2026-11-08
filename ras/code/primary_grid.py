"""Reference/biased reactive-current overlaps of the MD data on the primary joint discretization
(w0.25_g3) at lags of 1 and 2 ps, with the frame-level applied bias (Table 3). Writes results/primary_grid_real.json."""
import json
import joint_dham as jd

GRID = 'w0.25_g3'
OUT = jd.OUT

S = jd.build_states(**jd.CONFIGS[GRID])
fb = [jd.gaussian(f[:, 2], p) for (_, _, f, p) in jd.PROPS]
seqs = jd.observed_sequences(S); u = jd.state_bias(S, 'average')
res = {lag: jd.estimate(seqs, u, S['n'], S['region'], lag, frame_bias=fb) for lag in (1, 2)}
for lag in (1, 2):
    print(lag, {k: v for k, v in res[lag].items() if k not in ('P', 'J', 'Jb')})
json.dump({lag: {k: v for k, v in r.items() if k not in ('P', 'J', 'Jb')} for lag, r in res.items()},
          open(OUT / 'primary_grid_real.json', 'w'), indent=1)
