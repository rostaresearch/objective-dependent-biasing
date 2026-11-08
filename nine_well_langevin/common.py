"""Shared definitions for Test E (Langevin validation of the network MFPT predictions).

Everything here is either imported from, or a re-implementation checked against,
common/ (grid_sensitivity.py, exact_last_crossing.py). Nothing in
that folder is modified.

Units: F and b in kBT, D = 1, lengths in the units of the potential (domain [0, 2*pi]^2).
Network time with the code's k0 = 1 is converted to physical time with k0 = D/h^2,
i.e. t_phys = t_network * h^2 / D.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.io import loadmat
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

HERE = Path(__file__).resolve().parent
# Repository root (holds common/ and data/); can be overridden with E_LANGEVIN_ROOT on a cluster.
ROOT = Path(os.environ.get('E_LANGEVIN_ROOT', str(HERE.parent)))
AUDIT = ROOT / 'common'
sys.path.insert(0, str(AUDIT))
from grid_sensitivity import potential as audit_potential  # noqa: E402
from exact_last_crossing import DATA, generator_grid, tilt, mfpt as audit_mfpt  # noqa: E402

L_BOX = 2 * np.pi
D_COEF = 1.0
# Basin squares, as in grid_sensitivity.json (coarse-cell centres, half-width = half the 20x20 spacing)
CENTRE_A = np.array([4.960409453036515, 3.968327562429212])
CENTRE_B = np.array([0.992081890607303, 1.6534698176788385])
HALF = 0.16534698176788384

BIAS_KEYS = {'MFPT_U3': 'u_mfpt_1', 'spectral_U3': 'u_spec_1',
             'MFPT_U6': 'u_mfpt_2', 'spectral_U6': 'u_spec_2'}

# Potential parameters (copied from grid_sensitivity.potential)
_AMP = 6.
P_AMPL = np.array([.9, .3, .5, 1., .2, .4, .9, .9, .9]) * _AMP
P_X0 = np.array([1.12, 1., 3., 4.15, 4., 5.27, 5.5, 6., 1.])
P_Y0 = np.array([1.34, 2.25, 2.31, 3.62, 5., 4.14, 4.5, 1.52, 5.])
P_SX = np.array([.5, .3, .4, 2., .9, 1., .3, .5, .5])
P_SY = np.array([.5, .3, 1., .8, .2, .3, 1., .6, .7])
KCAL_PER_KT = .5981


def load_data():
    return loadmat(DATA / 'rkhs_2d_data.mat')


def coarse_axis():
    return load_data()['xs'].ravel()


def load_bias_table(name):
    """20x20 per-state bias, indexed [iy, ix] (row = y), as used in grid_sensitivity.py."""
    return np.ascontiguousarray(load_data()[BIAS_KEYS[name]], dtype=float)


# ---------------------------------------------------------------- continuous potential
@njit(cache=True)
def potential_grad(x, y):
    """Return F(x, y) (up to an additive constant) and its gradient, in kBT."""
    z = 0.
    gx = 0.
    gy = 0.
    for k in range(9):
        ex = (x - P_X0[k]) / P_SX[k]
        ey = (y - P_Y0[k]) / P_SY[k]
        arg = -.5 * (ex * ex + ey * ey)
        if arg > -60.:
            g = P_AMPL[k] * np.exp(arg)
            z -= g
            gx += g * ex / P_SX[k]
            gy += g * ey / P_SY[k]
    ex = (x - np.pi) / 3.
    ey = (y - np.pi) / .3
    g = .3 * _AMP * np.exp(-.5 * (ex * ex + ey * ey))
    z += g
    gx -= g * ex / 3.
    gy -= g * ey / .3
    # walls 100*expit(-5(x+.7)) etc.; d/dx expit(u) = s(1-s) du/dx
    for c, sign in ((x + .7, -1.), (x - (2 * np.pi + .7), 1.)):
        u = 5. * sign * c
        if u > -60.:
            s = 1. / (1. + np.exp(-u))
            z += 100. * s
            gx += 100. * s * (1. - s) * 5. * sign
    for c, sign in ((y + .7, -1.), (y - (2 * np.pi + .7), 1.)):
        u = 5. * sign * c
        if u > -60.:
            s = 1. / (1. + np.exp(-u))
            z += 100. * s
            gy += 100. * s * (1. - s) * 5. * sign
    return z / KCAL_PER_KT, gx / KCAL_PER_KT, gy / KCAL_PER_KT


@njit(parallel=True, cache=True)
def _fill_table(tg, xs):
    n = xs.size
    for j in prange(n):
        for i in range(n):
            _, gx, gy = potential_grad(xs[i], xs[j])
            tg[j, i, 0] = gx
            tg[j, i, 1] = gy


TABLE_N = 4096


def force_table(n=TABLE_N):
    """Exact gradient of F on an n x n node grid over [0, 2pi]^2, tg[j, i, :] at (xs[i], xs[j])."""
    xs = np.linspace(0, L_BOX, n)
    tg = np.empty((n, n, 2))
    _fill_table(tg, xs)
    return tg


@njit(cache=True)
def table_grad(tg, x, y):
    """Bilinear interpolation of the tabulated gradient of F."""
    n = tg.shape[0]
    fx = x * ((n - 1) / L_BOX)
    fy = y * ((n - 1) / L_BOX)
    i = int(fx)
    j = int(fy)
    if i > n - 2:
        i = n - 2
    if j > n - 2:
        j = n - 2
    if i < 0:
        i = 0
    if j < 0:
        j = 0
    s = fx - i
    r = fy - j
    w00 = (1 - s) * (1 - r)
    w10 = s * (1 - r)
    w01 = (1 - s) * r
    w11 = s * r
    gx = w00 * tg[j, i, 0] + w10 * tg[j, i + 1, 0] + w01 * tg[j + 1, i, 0] + w11 * tg[j + 1, i + 1, 0]
    gy = w00 * tg[j, i, 1] + w10 * tg[j, i + 1, 1] + w01 * tg[j + 1, i, 1] + w11 * tg[j + 1, i + 1, 1]
    return gx, gy


# ---------------------------------------------------------------- continuous bias
COARSE_N = 20
COARSE_H = L_BOX / (COARSE_N - 1)


@njit(cache=True)
def _cell(x, y):
    i = int(x / COARSE_H)
    j = int(y / COARSE_H)
    if i < 0:
        i = 0
    if i > COARSE_N - 2:
        i = COARSE_N - 2
    if j < 0:
        j = 0
    if j > COARSE_N - 2:
        j = COARSE_N - 2
    s = x / COARSE_H - i
    r = y / COARSE_H - j
    return i, j, s, r


@njit(cache=True)
def bias_bilinear(tab, x, y):
    """Bilinear interpolation of tab[iy, ix]; returns value and gradient."""
    i, j, s, r = _cell(x, y)
    b00 = tab[j, i]
    b10 = tab[j, i + 1]
    b01 = tab[j + 1, i]
    b11 = tab[j + 1, i + 1]
    v = (1 - s) * (1 - r) * b00 + s * (1 - r) * b10 + (1 - s) * r * b01 + s * r * b11
    gx = ((1 - r) * (b10 - b00) + r * (b11 - b01)) / COARSE_H
    gy = ((1 - s) * (b01 - b00) + s * (b11 - b10)) / COARSE_H
    return v, gx, gy


@njit(cache=True)
def bias_bicubic(coef, x, y):
    """Piecewise bicubic: coef[j, i, p, q] multiplies s^p r^q in cell (i, j)."""
    i, j, s, r = _cell(x, y)
    v = 0.
    gx = 0.
    gy = 0.
    sp = 1.
    for p in range(4):
        rq = 1.
        for q in range(4):
            c = coef[j, i, p, q]
            v += c * sp * rq
            if p > 0:
                gx += c * p * s ** (p - 1) * rq
            if q > 0:
                gy += c * q * sp * r ** (q - 1)
            rq *= r
        sp *= s
    return v, gx / COARSE_H, gy / COARSE_H


def bicubic_spline(tab):
    """C2 interpolating bicubic spline through the 20x20 values (FITPACK, s = 0)."""
    xs = np.linspace(0, L_BOX, COARSE_N)
    return RectBivariateSpline(xs, xs, tab, kx=3, ky=3, s=0)  # first argument is y (row)


def bicubic_coefficients(tab):
    """Exact per-cell polynomial coefficients of the bicubic spline (it is a bicubic
    polynomial inside each coarse cell, so 16 samples per cell determine it)."""
    spl = bicubic_spline(tab)
    t = np.array([0., 1 / 3, 2 / 3, 1.])
    V = np.vander(t, 4, increasing=True)          # V[a, p] = t_a^p
    M = np.kron(V, V)                             # rows (a_s, a_r), cols (p, q)
    Minv = np.linalg.inv(M)
    coef = np.zeros((COARSE_N - 1, COARSE_N - 1, 4, 4))
    xs = np.linspace(0, L_BOX, COARSE_N)
    for j in range(COARSE_N - 1):
        for i in range(COARSE_N - 1):
            S, R = np.meshgrid(t, t, indexing='ij')   # S varies along first index (a_s)
            xx = xs[i] + S * COARSE_H
            yy = xs[j] + R * COARSE_H
            vals = spl.ev(yy.ravel(), xx.ravel())
            coef[j, i] = (Minv @ vals).reshape(4, 4)
    return coef


def bias_on_points(name, interp, x, y):
    """Evaluate the continuous bias at arbitrary points (vectorised, for networks)."""
    tab = load_bias_table(name)
    xs = np.linspace(0, L_BOX, COARSE_N)
    if interp == 'bilinear':
        return RegularGridInterpolator((xs, xs), tab)(np.c_[y, x])
    if interp == 'bicubic':
        return bicubic_spline(tab).ev(y, x)
    raise ValueError(interp)


# ---------------------------------------------------------------- networks
def grid(n):
    x = np.linspace(0, L_BOX, n)
    X, Y = np.meshgrid(x, x)
    coords = np.c_[X.ravel(), Y.ravel()]
    A, B = [np.flatnonzero(np.max(abs(coords - c), axis=1) <= HALF + 1e-12) for c in (CENTRE_A, CENTRE_B)]
    return x, coords, A, B


def generator_sg(U):
    """Scharfetter-Gummel (exponential-fitting) generator with k0 = 1: rate i->j = B(U_j-U_i),
    B(s) = s/(e^s-1). An independent consistent discretization of the same Smoluchowski operator."""
    U = np.asarray(U)
    idx = np.arange(U.size).reshape(U.shape)
    a, b = [], []
    for axis in range(U.ndim):
        left, right = [slice(None)] * U.ndim, [slice(None)] * U.ndim
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        p, q = idx[tuple(left)].ravel(), idx[tuple(right)].ravel()
        a.extend([p, q]); b.extend([q, p])
    i, j = np.concatenate(a), np.concatenate(b)
    s = U.ravel()[j] - U.ravel()[i]
    with np.errstate(divide='ignore', invalid='ignore'):
        rates = np.where(abs(s) < 1e-8, 1. - s / 2, s / np.expm1(s))
    K = csr_matrix((rates, (i, j)), shape=(U.size, U.size))
    return K - diags(np.asarray(K.sum(axis=1)).ravel())


def mfpt_vector(K, B):
    """Mean first-passage time into B from every state (network units)."""
    n = K.shape[0]
    free = np.setdiff1d(np.arange(n), B)
    t = np.zeros(n)
    t[free] = spsolve(csr_matrix(-K[free][:, free]).tocsc(), np.ones(len(free)))
    if t.min() < -1e-8 or not np.isfinite(t).all():
        raise ValueError('Invalid MFPT solve')
    return t


# ---------------------------------------------------------------- basins
@njit(cache=True)
def in_B(x, y):
    return abs(x - CENTRE_B[0]) <= HALF and abs(y - CENTRE_B[1]) <= HALF
