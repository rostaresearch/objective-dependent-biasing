"""Shared loading and exact-model metrics for the pathway-overlap reference values and statistical cost.

Imports the network construction and the last-crossing law from the TS audit
(common/exact_last_crossing.py) without modifying it.
All quantities here are exact finite-network (model-level) results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

sys.dont_write_bytecode = True          # do not write .pyc files into the audit folder
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repository root
AUDIT = ROOT/'common'
sys.path.insert(0, str(AUDIT))
import exact_last_crossing as elc       # noqa: E402

harmonic, law, tilt, boltzmann, mfpt, generator_grid = (
    elc.harmonic, elc.law, elc.tilt, elc.boltzmann, elc.mfpt, elc.generator_grid)


class System:
    """One analytic network with its saved optimized biases."""

    def __init__(self, name, F, coords, A, B, biases, source):
        self.name = name
        self.shape = F.shape
        self.F = np.asarray(F, float).ravel()
        self.coords = coords
        self.A, self.B = np.asarray(A), np.asarray(B)
        self.biases = {k: np.asarray(v, float) for k, v in biases.items()}
        self.source = str(source)
        self.n = self.F.size
        self.K0 = generator_grid(np.asarray(F))
        self.pi0 = boltzmann(self.F)
        self.q0 = harmonic(self.K0, self.A, self.B)
        self.S = self.q0 < .5
        self.t0 = mfpt(self.K0, self.A, self.B)
        self.gap0 = spectral_gap(self.K0, self.pi0)
        self.ref = law(self.K0, self.pi0, self.A, self.B, self.q0)
        # undirected edge list (a<b) of the grid
        coo = self.K0.tocoo()
        m = coo.row < coo.col
        self.ea, self.eb = coo.row[m], coo.col[m]
        self.n_edges = len(self.ea)
        self.edge_index = {(int(a), int(b)): e for e, (a, b) in enumerate(zip(self.ea, self.eb))}
        self.J0 = signed_current(self.ref, self.ea, self.eb)
        self.p0 = self.ref['probability']

    def bias_pi(self, b):
        return boltzmann(self.F + b)


def systems():
    out = {}
    for name, F, coords, A, B, biases, source in elc.cases():
        out[name] = System(name, F, coords, A, B, biases, source)
    return out


def spectral_gap(K, pi):
    """-lambda_2 of a reversible generator via the symmetrized matrix D^1/2 K D^-1/2."""
    d = np.sqrt(pi)
    Kd = K.toarray() if hasattr(K, 'toarray') else np.asarray(K)
    Sm = (d[:, None]*Kd)/d[None, :]
    Sm = .5*(Sm+Sm.T)
    n = Sm.shape[0]
    vals = eigh(Sm, eigvals_only=True, subset_by_index=[n-2, n-1])
    return float(-vals[0])


def signed_current(lw, ea, eb):
    """Normalized net reactive current on undirected edges (a<b), sign = direction.

    sum(|x|) = 1; the positive part on the directed edges is the J-hat of the paper.
    """
    cur = lw['current']                  # normalized positive part, directed
    x = np.asarray(cur[ea, eb]).ravel() - np.asarray(cur[eb, ea]).ravel()
    return x/np.abs(x).sum()


def overlap_signed(x, y):
    """Omega_J between two signed edge vectors, each normalized to sum|.|=1."""
    return float(1 - .5*np.abs(x - y).sum())


def overlap_prob(p, r):
    return float(np.minimum(p, r).sum())


def ess_stats(pi0, pib):
    """Kish ESS/N for reweighting i.i.d. samples of pib to pi0, plus weight statistics."""
    w = pi0/pib
    second = float(np.sum(pib*w*w))     # = 1 + chi^2(pi0 || pib)
    logw = np.log(pi0) - np.log(pib)
    mlog = float(np.sum(pib*logw))
    return dict(ess_frac=1/second, chi2=second-1, max_w=float(w.max()),
                var_logw=float(np.sum(pib*(logw-mlog)**2)))


def metrics(sys_, b, with_law=True):
    """All exact-model metrics for bias b on system sys_."""
    Kb = tilt(sys_.K0, b)
    pib = sys_.bias_pi(b)
    row = dict(mfpt_speedup=sys_.t0/mfpt(Kb, sys_.A, sys_.B),
               gap_speedup=spectral_gap(Kb, pib)/sys_.gap0)
    if with_law:
        lb = law(Kb, pib, sys_.A, sys_.B, sys_.q0)
        assert np.array_equal(lb['i'], sys_.ref['i']) and np.array_equal(lb['j'], sys_.ref['j'])
        row['C_TS'] = overlap_prob(sys_.p0, lb['probability'])
        row['Omega_J'] = overlap_signed(sys_.J0, signed_current(lb, sys_.ea, sys_.eb))
        row['Omega_J_sparse'] = float(sys_.ref['current'].minimum(lb['current']).sum())
    row.update(ess_stats(sys_.pi0, pib))
    row['max_abs_b'] = float(np.abs(b).max())
    return row


def ipr(p):
    return float(1/np.sum(np.asarray(p)**2))


def reproduce_audit(systems_):
    """Compare against last_crossing/exact_last_crossing_results.json."""
    saved = json.loads((ROOT/'last_crossing'/'exact_last_crossing_results.json').read_text())
    rows, worst = [], 0.
    for name, s in systems_.items():
        for key, b in s.biases.items():
            m = metrics(s, b)
            ref = saved['cases'][name]['protocols'][key][-1]
            rel = {k: abs(m[k]-ref[k])/abs(ref[k]) for k in ['C_TS', 'Omega_J', 'mfpt_speedup']}
            worst = max(worst, max(rel.values()))
            rows.append(dict(system=name, bias=key, C_TS=m['C_TS'], Omega_J=m['Omega_J'],
                             mfpt_speedup=m['mfpt_speedup'], gap_speedup=m['gap_speedup'],
                             relative_error_vs_audit=rel))
    return rows, worst


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o
