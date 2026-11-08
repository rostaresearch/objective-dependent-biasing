"""Exact finite-generator last-crossing laws on a FIXED reference committor cut.

S={q0<1/2}. For any tested generator K, let qminus be the probability that the
last visited basin was A and h=P(hit B before S). The unnormalized last-exit
measure on cut edges is pi_i*qminus_i*K_ij*h_j, i in S, j outside S.
Its total is the A->B reactive capacity. This differs from net-current flux
on individual cut edges. No learned matrix is symmetrized in this analysis.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
DATA = ROOT/'data'
OUT = ROOT/'last_crossing'


def rel(path):
    """Repository-relative path for provenance records."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.name


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def harmonic(K, zero, one):
    K = csr_matrix(K)
    n = K.shape[0]
    zero, one = np.asarray(zero, int), np.asarray(one, int)
    assert not np.intersect1d(zero, one).size
    free = np.setdiff1d(np.arange(n), np.r_[zero, one])
    q = np.zeros(n); q[one] = 1.
    if len(free):
        q[free] = spsolve(K[free][:, free], -np.asarray(K[free][:, one].sum(axis=1)).ravel())
        residual = np.max(np.abs((K@q)[free]))/max(1., abs(K.data).max())
        if residual > 1e-10 or q.min() < -1e-8 or q.max() > 1+1e-8:
            raise ValueError(f'Invalid harmonic solve: residual={residual}, range={q.min(),q.max()}')
    return np.clip(q, 0., 1.)


def generator_grid(F):
    F = np.asarray(F)
    indices = np.arange(F.size).reshape(F.shape)
    a, b = [], []
    for axis in range(F.ndim):
        left, right = [slice(None)]*F.ndim, [slice(None)]*F.ndim
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        x, y = indices[tuple(left)].ravel(), indices[tuple(right)].ravel()
        a.extend([x, y]); b.extend([y, x])
    i, j = np.concatenate(a), np.concatenate(b)
    rates = np.exp(-.5*(F.ravel()[j]-F.ravel()[i]))
    K = csr_matrix((rates, (i, j)), shape=(F.size, F.size))
    return K-diags(np.asarray(K.sum(axis=1)).ravel())


def tilt(K, bias):
    off = K.copy().tolil(); off.setdiag(0.); off = off.tocoo()
    value = off.data*np.exp(-.5*(bias[off.col]-bias[off.row]))
    out = csr_matrix((value, (off.row, off.col)), shape=K.shape)
    return out-diags(np.asarray(out.sum(axis=1)).ravel())


def boltzmann(F):
    w = np.exp(-(F-np.min(F)))
    return w/w.sum()


def mfpt(K, A, B):
    free = np.setdiff1d(np.arange(K.shape[0]), B)
    t = np.zeros(K.shape[0])
    t[free] = spsolve(-K[free][:, free], np.ones(len(free)))
    if t.min() < -1e-8 or not np.isfinite(t).all():
        raise ValueError('Invalid MFPT solve')
    return float(np.mean(t[A]))


def law(K, pi, A, B, q0, level=.5):
    K = csr_matrix(K)
    stationary_residual = np.max(abs(np.asarray(pi@K).ravel()))
    if stationary_residual > 1e-10*max(1., abs(K.data).max()):
        raise ValueError('Invalid supplied stationary distribution')
    q = harmonic(K, A, B)
    reverse = diags(1/pi)@K.T@diags(pi)
    qm = harmonic(reverse, B, A)
    S = q0 < level
    if not S[A].all() or S[B].any():
        raise ValueError('Cut must separate A and B')
    h = harmonic(K, np.flatnonzero(S), B)
    coo = K.tocoo()
    mask = (coo.row != coo.col) & (coo.data > 0)
    i, j, rates = coo.row[mask], coo.col[mask], coo.data[mask]
    cross = S[i] & ~S[j]
    mass = pi[i[cross]]*qm[i[cross]]*rates[cross]*h[j[cross]]
    capacity = float(np.sum(pi[A]*np.asarray(K[A]@q).ravel()))
    flux_error = abs(mass.sum()-capacity)/capacity
    if flux_error > 2e-7:
        raise ValueError(f'Last-crossing flux does not equal reactive capacity: {flux_error}')
    if mass.sum() <= 0 or not np.isfinite(mass).all():
        raise ValueError('No reactive crossing mass')
    gross = csr_matrix((pi[i]*qm[i]*rates*q[j], (i,j)), shape=K.shape)
    net = gross-gross.T
    net.data = np.maximum(net.data, 0.); net.eliminate_zeros()
    net /= net.sum()
    return dict(q=q, qminus=qm, h=h, i=i[cross], j=j[cross],
                probability=mass/mass.sum(), capacity=capacity,
                capacity_relative_residual=float(flux_error), current=net)


def overlap_current(a, b):
    return float(a.minimum(b).sum())


def cases():
    d = loadmat(DATA/'rkhs_2d_data.mat')
    coords = np.c_[d['X'].ravel(), d['Y'].ravel()]
    A = np.array([np.argmin(np.linalg.norm(coords-d['start_xy'].ravel(), axis=1))])
    B = np.array([np.argmin(np.linalg.norm(coords-d['end_xy'].ravel(), axis=1))])
    yield 'nine_well', d['F'], coords, A, B, {
        'MFPT_U3': d['u_mfpt_1'].ravel(), 'spectral_U3': d['u_spec_1'].ravel(),
        'MFPT_U6': d['u_mfpt_2'].ravel(), 'spectral_U6': d['u_spec_2'].ravel()}, DATA/'rkhs_2d_data.mat'
    d = loadmat(DATA/'grid_2d_data.mat')
    x, y = np.meshgrid(d['xs'].ravel(), d['ys'].ravel(), indexing='ij')
    coords = np.c_[x.ravel(), y.ravel()]
    A, B = np.flatnonzero(coords[:,0] < -.85), np.flatnonzero(coords[:,0] > .85)
    yield 'bottleneck_grid', d['F'], coords, A, B, {
        'MFPT_U3': d['u_ps_mfpt'].ravel(), 'spectral_U3': d['u_ps_spec'].ravel(),
        'polynomial_spectral_U3': d['u_poly_spec'].ravel()}, DATA/'grid_2d_data.mat'


def main():
    alphas = np.linspace(0, 1, 21)
    result = dict(definition='histogram intersection of exact last-exit edge laws on fixed q0<0.5 cut',
                  interpretation='finite-network model conditional; no trajectory sampling or bootstrap CI',
                  code_sha256=sha(__file__), cases={})
    for name, F, coords, A, B, biases, source in cases():
        K0 = generator_grid(F)
        pi0 = boltzmann(F.ravel())
        q0 = harmonic(K0, A, B)
        ref = law(K0, pi0, A, B, q0)
        t0 = mfpt(K0, A, B)
        frac = (.5-q0[ref['i']])/(q0[ref['j']]-q0[ref['i']])
        crossing_xy = coords[ref['i']]*(1-frac[:,None])+coords[ref['j']]*frac[:,None]
        row = dict(source=rel(source), source_sha256=sha(source), n_states=len(pi0),
                   n_cut_edges=len(frac), A=A.tolist(), B=B.tolist(), mfpt0=t0,
                   reference_flux_residual=ref['capacity_relative_residual'], protocols={})
        arrays = dict(coords=coords, F=F, q0=q0, A=A+1, B=B+1, crossing_xy=crossing_xy,
                      reference_probability=ref['probability'], alphas=alphas,
                      cut_i=ref['i']+1, cut_j=ref['j']+1)
        for key, bias in biases.items():
            values = []
            for a in alphas:
                Kb = tilt(K0, a*bias)
                lb = law(Kb, boltzmann(F.ravel()+a*bias), A, B, q0)
                assert np.array_equal(lb['i'], ref['i']) and np.array_equal(lb['j'], ref['j'])
                values.append(dict(alpha=float(a), C_TS=float(np.minimum(ref['probability'], lb['probability']).sum()),
                                   Omega_J=overlap_current(ref['current'], lb['current']),
                                   mfpt_speedup=t0/mfpt(Kb, A, B),
                                   capacity_relative_residual=lb['capacity_relative_residual']))
            row['protocols'][key] = values
            arrays[key+'_probability'] = lb['probability']
            arrays[key+'_C_TS'] = np.array([v['C_TS'] for v in values])
            arrays[key+'_Omega_J'] = np.array([v['Omega_J'] for v in values])
            arrays[key+'_speedup'] = np.array([v['mfpt_speedup'] for v in values])
            print(name, key, values[-1], flush=True)
        if name == 'nine_well':
            old = loadmat(DATA/'mechanism_data.mat')
            residuals = {}
            for key, prefix in [('MFPT_U6','m'),('spectral_U6','g')]:
                for newfield, oldfield in [('Omega_J','OmegaJ'),('mfpt_speedup','mfpt_su')]:
                    new = np.array([v[newfield] for v in row['protocols'][key]])
                    previous = old[prefix+'_'+oldfield].ravel()
                    residuals[key+'_'+newfield] = float(np.max(abs(new-previous)/np.maximum(abs(previous),1e-12)))
            row['published_reproduction_relative_errors'] = residuals
        result['cases'][name] = row
        row['cut_level_sensitivity'] = {}
        for level in [.4, .45, .5, .55, .6]:
            reference_level = law(K0, pi0, A, B, q0, level)
            level_rows = {}
            for key, bias in biases.items():
                biased_level = law(tilt(K0,bias),boltzmann(F.ravel()+bias),A,B,q0,level)
                level_rows[key] = float(np.minimum(reference_level['probability'],biased_level['probability']).sum())
            row['cut_level_sensitivity'][str(level)] = level_rows
        savemat(OUT/(name+'_last_crossing.mat'), arrays)
    # Analytically specified triangle in the submitted manuscript, A=0 B=1 D=2.
    a, c, d, U = .05, 1., .02, 3.
    K0 = csr_matrix([[-a-c,a,c],[a,-a-d,d],[c,d,-c-d]])
    pi0 = np.ones(3)/3
    A, B = np.array([0]), np.array([1])
    q0 = harmonic(K0,A,B); ref = law(K0,pi0,A,B,q0); t0=mfpt(K0,A,B)
    decoy = np.clip(2*np.log((a+np.sqrt(a*a+c*d*d/(c+np.exp(U/2)*d)))/d),-U,U)
    tri = dict(q0=q0.tolist(), reference_cut_edges=np.c_[ref['i'],ref['j']].tolist(),
               reference_probability=ref['probability'].tolist(), protocols={})
    for key, bias in [('MFPT_U3',np.array([0.,-U,decoy])),('spectral_U3',np.array([0.,U,-U]))]:
        kb=tilt(K0,bias); lb=law(kb,boltzmann(bias),A,B,q0)
        tri['protocols'][key]=dict(bias=bias.tolist(), C_TS=float(np.minimum(ref['probability'],lb['probability']).sum()),
                                    Omega_J=overlap_current(ref['current'],lb['current']),
                                    mfpt_speedup=t0/mfpt(kb,A,B), probability=lb['probability'].tolist())
    result['cases']['triangle']=tri
    (OUT/'exact_last_crossing_results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
