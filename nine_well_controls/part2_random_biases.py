"""Part 2: random-bias reference at matched budget max|b| <= U, plus alpha-scaling curves.

Kinds of random bias (M per kind, per system and U):
  iid      b_i ~ Uniform[-U, U] independently per state;
  grf_long Gaussian random field, squared-exponential covariance exp(-r^2/(2 l^2)) with
           l = one well spacing (nine-well 2*pi/3 = 2.09; bottleneck 2.0, the A-B well distance),
           rescaled so that max|b| = U;
  grf_short the same with l = one third of a well spacing (0.70; 0.67).
For each bias: MFPT speedup (A->B, mean over A states as in the TS audit), spectral-gap
speedup, Omega_J, C_TS, and the reweighting statistics of part 4 (ESS/N, max weight,
Var log w). All exact network-model values.
"""
from __future__ import annotations

import json
import time
from multiprocessing import Pool

import numpy as np

from common import HERE, systems, metrics, to_jsonable

M = 2000
CONFIGS = [('nine_well', 3.), ('nine_well', 6.), ('bottleneck_grid', 3.)]
WELL_SPACING = {'nine_well': 2*np.pi/3, 'bottleneck_grid': 2.0}
KINDS = ['iid', 'grf_long', 'grf_short']
KEYS = ['mfpt_speedup', 'gap_speedup', 'C_TS', 'Omega_J', 'ess_frac', 'chi2', 'max_w', 'var_logw', 'max_abs_b']
OPT = {('nine_well', 3.): ['MFPT_U3', 'spectral_U3'], ('nine_well', 6.): ['MFPT_U6', 'spectral_U6'],
       ('bottleneck_grid', 3.): ['MFPT_U3', 'spectral_U3', 'polynomial_spectral_U3']}
ALPHAS = np.linspace(0, 1, 21)
_S = None


def grf_factor(coords, ell):
    d2 = ((coords[:, None, :]-coords[None, :, :])**2).sum(-1)
    C = np.exp(-d2/(2*ell**2))
    lam, V = np.linalg.eigh(C)
    return V*np.sqrt(np.clip(lam, 0, None))


def draw(s, kind, U, rng):
    if kind == 'iid':
        return rng.uniform(-U, U, size=(M, s.n))
    ell = WELL_SPACING[s.name]*(1 if kind == 'grf_long' else 1/3)
    L = grf_factor(s.coords, ell)
    z = rng.standard_normal((M, s.n))@L.T
    return np.clip(U*z/np.abs(z).max(axis=1, keepdims=True), -U, U)   # clip removes 1-ulp rounding only


def _init():
    global _S
    _S = systems()


def _work(args):
    name, B = args
    out = np.full((len(B), len(KEYS)), np.nan)
    errors = 0
    for k, b in enumerate(B):
        try:
            m = metrics(_S[name], b)
            out[k] = [m[key] for key in KEYS]
        except ValueError:
            errors += 1
    return out, errors


def run_batch(pool, name, B, chunk=50):
    parts = pool.map(_work, [(name, B[i:i+chunk]) for i in range(0, len(B), chunk)])
    return np.vstack([p[0] for p in parts]), sum(p[1] for p in parts)


def pct(v, x):
    v = v[np.isfinite(v)]
    return float(100*(np.sum(v < x) + .5*np.sum(v == x))/len(v))


def summary(v):
    v = v[np.isfinite(v)]
    return {f'p{q:02d}': float(np.percentile(v, q)) for q in [0, 5, 25, 50, 75, 95, 100]}


def main():
    S = systems()
    t0 = time.time()
    res = dict(description='random biases at matched budget; exact network-model metrics', M=M,
               kinds=dict(iid='per-state i.i.d. uniform in [-U, U]',
                          grf_long='Gaussian random field, squared-exponential covariance exp(-r^2/(2 l^2)), '
                                   'l = one well spacing, rescaled to max|b| = U',
                          grf_short='same, l = one third of a well spacing'),
               correlation_length={n: dict(grf_long=WELL_SPACING[n], grf_short=WELL_SPACING[n]/3) for n in WELL_SPACING},
               grid_spacing={n: [float(np.diff(np.unique(S[n].coords[:, 0]))[0]),
                                 float(np.diff(np.unique(S[n].coords[:, 1]))[0])] for n in S},
               configs={})
    arrays = {}
    # check of the field generator: empirical covariance of unscaled fields vs the target kernel
    res['grf_covariance_check'] = {}
    for name in S:
        for kind in ['grf_long', 'grf_short']:
            ell = WELL_SPACING[name]*(1 if kind == 'grf_long' else 1/3)
            L = grf_factor(S[name].coords, ell)
            z = np.random.default_rng(99).standard_normal((20000, S[name].n))@L.T
            d2 = ((S[name].coords[:, None, :]-S[name].coords[None, :, :])**2).sum(-1)
            C = np.exp(-d2/(2*ell**2))
            err = np.abs(z.T@z/len(z) - C)
            res['grf_covariance_check'][f'{name}_{kind}'] = dict(
                max_abs_error=float(err.max()), rms_error=float(np.sqrt((err**2).mean())),
                expected_sampling_error_scale=float(np.sqrt(2/len(z))))
    with Pool(12, initializer=_init) as pool:
        for ci, (name, U) in enumerate(CONFIGS):
            s = S[name]
            cfg = dict(system=name, U=U, kinds={}, optimized={}, alpha_curves={})
            for kid, kind in enumerate(KINDS):
                rng = np.random.default_rng([20261010, ci, kid])
                B = draw(s, kind, U, rng)
                vals, errors = run_batch(pool, name, B)
                arrays[f'{name}_U{U:g}_{kind}'] = vals
                cfg['kinds'][kind] = dict(n_failed=int(errors),
                                          percentiles={k: summary(vals[:, i]) for i, k in enumerate(KEYS)},
                                          fraction_mfpt_speedup_gt1=float(np.mean(vals[:, 0] > 1)),
                                          mean_abs_b=float(np.abs(B).mean()))
                print(name, U, kind, 'failed', errors, 'median MFPT su', np.nanmedian(vals[:, 0]),
                      'median C_TS', np.nanmedian(vals[:, 2]), 'median Omega', np.nanmedian(vals[:, 3]),
                      f'{time.time()-t0:.0f}s', flush=True)
            for key in OPT[(name, U)]:
                b = s.biases[key]
                m = metrics(s, b)
                ranks = {kind: {k: pct(arrays[f'{name}_U{U:g}_{kind}'][:, i], m[k]) for i, k in enumerate(KEYS)}
                         for kind in KINDS}
                cfg['optimized'][key] = dict(metrics={k: m[k] for k in KEYS}, percentile_rank=ranks,
                                             mean_abs_b=float(np.abs(b).mean()))
                curve, _ = run_batch(pool, name, np.outer(ALPHAS, b), chunk=3)
                arrays[f'{name}_{key}_alpha'] = curve
                cfg['alpha_curves'][key] = {k: curve[:, i].tolist() for i, k in enumerate(KEYS)}
            res['configs'][f'{name}_U{U:g}'] = cfg
    res['alphas'] = ALPHAS.tolist()
    res['metric_columns'] = KEYS
    res['seconds'] = time.time()-t0
    (HERE/'results_part2.json').write_text(json.dumps(to_jsonable(res), indent=1)+'\n')
    np.savez_compressed(HERE/'part2_random_metrics.npz', columns=np.array(KEYS), alphas=ALPHAS, **arrays)


if __name__ == '__main__':
    main()
