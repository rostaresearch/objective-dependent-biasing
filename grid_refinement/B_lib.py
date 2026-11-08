"""Shared code for the grid-refinement (discretization robustness) checks.

Systems
-------
* Bottleneck surface: the F of analytic_lib.grid_2d_generator (bottleneck=True),
  node-centred grid on [-1,1]^2, state index ix*ny+iy.  The saved per-state
  optima in grid_2d_data.mat were computed with barrier_height=4.0
  (export_grid_2d_mat.py, per_state_ceiling.py, mfpt_per_state.py); this is
  checked numerically in B0_checks.py.
* Nine-well surface: the F of common/grid_sensitivity.py,
  node-centred grid on [0,2pi]^2, state index iy*n+ix (meshgrid 'xy').

Generators are built sparse with exact_last_crossing.generator_grid (SQRA,
k0 = 1) and tilted with exact_last_crossing.tilt.  B0_checks.py verifies that
they equal analytic_lib.grid_2d_generator / analytic_lib.tilt_generator.

A and B are fixed physical regions: the union of the footprints of the coarse
(reference-grid) basin cells, as in grid_sensitivity.py.  MFPT(A->B) is the
uniform average of the mean first-passage time over the cells of A, as in
analytic_lib.mfpt_K and exact_last_crossing.mfpt.

Objectives and gradients
------------------------
The fast routines below use the symmetrised form of the reversible tilted
generator, S = D^(1/2) K^b D^(-1/2) with D = diag(pi_b).  Its off-diagonal
entries do not depend on b; only the diagonal K^b_ii does.  For a simple
eigenvalue lambda_2 with unit eigenvector phi of S (Hellmann-Feynman):

    d gamma / d b_k = -(1/2) sum_p phi_p^2 K^b_pk ,   gamma = -lambda_2.

This equals the left/right-eigenvector expression in per_state_ceiling.gap_and_grad
(checked numerically in B0_checks.py).  The MFPT gradient is the adjoint formula of
mfpt_per_state.mfpt_and_grad evaluated with a sparse LU factorisation.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh, splu

REPO = Path(__file__).resolve().parents[1]      # repository root
AUDIT = REPO/'common'
DATA = REPO/'data'
ROOT = DATA
OUT = Path(__file__).resolve().parent


def rel(path):
    """Repository-relative path for provenance records."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.name


for p in (str(ROOT), str(AUDIT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import analytic_lib as L                     # noqa: E402
import exact_last_crossing as X              # noqa: E402
import grid_sensitivity as GS                # noqa: E402

BOTTLENECK_BARRIER = 4.0      # barrier_height used for the saved optima (see B0_checks)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def dump(obj, path):
    Path(path).write_text(json.dumps(jsonable(obj), indent=1) + '\n')


# ---------------------------------------------------------------- systems

def bottleneck_F(nx, ny, barrier_height=BOTTLENECK_BARRIER):
    """Same formula and ordering as analytic_lib.grid_2d_generator(bottleneck=True)."""
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    Xg, Yg = np.meshgrid(xs, ys, indexing='ij')
    coords = np.column_stack([Xg.ravel(), Yg.ravel()])
    x, y = coords[:, 0], coords[:, 1]
    F = barrier_height*(x**2 - 1.0)**2 + 2.0*y**2
    F = F + 6.0*np.exp(-(x/0.20)**2)*(1.0 - np.exp(-(y/0.18)**2))
    return xs, ys, coords, F


class System:
    """Container for one discretised surface."""

    def __init__(self, name, nx, ny, xs, ys, coords, F, K0, A, B, shape, interp_axes):
        self.name, self.nx, self.ny = name, nx, ny
        self.xs, self.ys, self.coords, self.F = xs, ys, coords, F
        self.K0, self.A, self.B = K0, np.asarray(A, int), np.asarray(B, int)
        self.n = F.size
        self.shape = shape            # reshape of a state vector to a 2-D map
        self.interp_axes = interp_axes
        self.logpi0 = -(F - F.min())
        self.pi0 = X.boltzmann(F)
        self._q0 = None

    @property
    def label(self):
        return f'{self.nx}x{self.ny}'

    def q0(self):
        if self._q0 is None:
            self._q0 = X.harmonic(self.K0, self.A, self.B)
        return self._q0

    def to_map(self, v):
        return np.asarray(v).reshape(self.shape)


def bottleneck_system(nx, ny):
    xs, ys, coords, F = bottleneck_F(nx, ny)
    K0 = X.generator_grid(F.reshape(nx, ny))
    # A/B: union of footprints of the coarse (20x10) basin cells x < -0.85 / x > 0.85.
    xc = np.linspace(-1, 1, 20)
    hc = xc[1] - xc[0]
    xa_max = xc[xc < -.85].max() + hc/2
    xb_min = xc[xc > .85].min() - hc/2
    A = np.flatnonzero(coords[:, 0] <= xa_max + 1e-12)
    B = np.flatnonzero(coords[:, 0] >= xb_min - 1e-12)
    return System('bottleneck', nx, ny, xs, ys, coords, F, K0, A, B, (nx, ny), 'ij')


def nine_well_system(n):
    x = np.linspace(0, 2*np.pi, n)
    Xg, Yg = np.meshgrid(x, x)
    coords = np.c_[Xg.ravel(), Yg.ravel()]
    F2 = GS.potential(x)
    K0 = X.generator_grid(F2)
    d = loadmat(DATA/'rkhs_2d_data.mat')
    xc = d['xs'].ravel()
    half = (xc[1] - xc[0])/2
    cc = np.c_[d['X'].ravel(), d['Y'].ravel()]
    centres = [cc[np.argmin(np.linalg.norm(cc - d[k].ravel(), axis=1))] for k in ['start_xy', 'end_xy']]
    A, B = [np.flatnonzero(np.max(abs(coords - c), axis=1) <= half + 1e-12) for c in centres]
    return System('nine_well', n, n, x, x, coords, F2.ravel(), K0, A, B, (n, n), 'xy')


def interpolate_bias(b_coarse_map, xs_c, ys_c, sysf, axes):
    """Bilinear interpolation of a coarse per-state bias map onto sysf's nodes.
    axes='ij': map[ix,iy] on (xs,ys);  axes='xy': map[iy,ix] on (ys,xs)."""
    f = RegularGridInterpolator((xs_c, ys_c) if axes == 'ij' else (ys_c, xs_c), b_coarse_map)
    pts = sysf.coords if axes == 'ij' else sysf.coords[:, ::-1]
    return f(pts)


# ---------------------------------------------------------------- kinetics

def tilt(K0, b):
    return X.tilt(K0, np.asarray(b, float))


def sym_operator(Kb, logpi):
    """S = D^(1/2) K D^(-1/2), D = diag(pi); exactly symmetric for reversible K."""
    c = Kb.tocoo()
    v = c.data*np.exp(0.5*(logpi[c.row] - logpi[c.col]))
    S = csr_matrix((v, (c.row, c.col)), shape=Kb.shape)
    return ((S + S.T)*0.5).tocsc()


def top_eigs(Kb, logpi, k=4, sigma=None):
    """Largest k eigenvalues (0 > l2 >= l3 ...) of the reversible generator Kb,
    via shift-invert Lanczos on its symmetrised form.  Returns (vals desc, vecs of S)."""
    S = sym_operator(Kb, logpi)
    if sigma is None:
        sigma = 1e-9*max(1.0, abs(S.diagonal()).max())
    vals, vecs = eigsh(S, k=k, sigma=sigma, which='LM', tol=0, maxiter=10000)
    o = np.argsort(-vals)
    return vals[o], vecs[:, o]


def gap_and_grad_fast(b, sysm, return_extra=False):
    """gamma = -lambda_2 of K^b and its analytic (Hellmann-Feynman) gradient."""
    Kb = tilt(sysm.K0, b)
    logpi = sysm.logpi0 - b
    vals, vecs = top_eigs(Kb, logpi, k=3)
    lam2 = vals[1]
    phi = vecs[:, 1]
    grad = -0.5*np.asarray(Kb.T @ (phi*phi)).ravel()
    if return_extra:
        return -lam2, grad, dict(lams=vals, phi=phi)
    return -lam2, grad


class _MFPTSolver:
    def __init__(self, sysm):
        self.sysm = sysm
        n = sysm.n
        self.N = np.setdiff1d(np.arange(n), sysm.B)
        self.posA = np.searchsorted(self.N, sysm.A)
        assert np.array_equal(self.N[self.posA], sysm.A)


def mfpt_and_grad_fast(b, sysm, solver=None):
    """mean_{i in A} MFPT_i(A->B) of K^b and its adjoint gradient
    (same formula as mfpt_per_state.mfpt_and_grad, sparse LU)."""
    if solver is None:
        solver = _MFPTSolver(sysm)
    n = sysm.n
    Kb = tilt(sysm.K0, b).tocsr()
    N = solver.N
    M = Kb[N][:, N].tocsc()
    lu = splu(M)
    mN = lu.solve(-np.ones(len(N)))
    a = np.zeros(len(N)); a[solver.posA] = 1.0/len(sysm.A)
    lamN = lu.solve(a, trans='T')
    g = float(a @ mN)
    m = np.zeros(n); m[N] = mN
    lam = np.zeros(n); lam[N] = lamN
    alpha = Kb @ m
    beta = Kb.T @ (lam*m)
    gam = Kb.T @ lam
    grad = -0.5*(lam*alpha + beta - m*gam)
    return g, np.asarray(grad).ravel()


def mfpt_value(b, sysm):
    return X.mfpt(tilt(sysm.K0, b), sysm.A, sysm.B)


def gap_value(b, sysm):
    return float(-top_eigs(tilt(sysm.K0, b), sysm.logpi0 - b, k=3)[0][1])


# ---------------------------------------------------------------- mechanism

def mechanism(sysm, b, ref=None):
    """C_TS and Omega_J of bias b against the unbiased network of the same grid
    (exact_last_crossing.law on the fixed unbiased q0 = 1/2 cut)."""
    q0 = sysm.q0()
    if ref is None:
        ref = X.law(sysm.K0, sysm.pi0, sysm.A, sysm.B, q0)
    Kb = tilt(sysm.K0, b)
    lb = X.law(Kb, X.boltzmann(sysm.F + b), sysm.A, sysm.B, q0)
    assert np.array_equal(lb['i'], ref['i']) and np.array_equal(lb['j'], ref['j'])
    return dict(C_TS=float(np.minimum(ref['probability'], lb['probability']).sum()),
                Omega_J=X.overlap_current(ref['current'], lb['current']),
                flux_residual=lb['capacity_relative_residual'],
                n_cut_edges=int(len(ref['i'])),
                cut_effective_number=float(1/np.sum(ref['probability']**2))), ref, lb


# ---------------------------------------------------------------- second routes
# Independent numerical routes used to cross-check every reported number.

def mfpt_route2(Kb, logpi, A, B, dense_limit=3200):
    """(i) occupation (forward) route: MFPT = -sum(lam), M^T lam = a, solved with
    spsolve on the transposed matrix; (ii) for n <= dense_limit, eigendecomposition
    of the symmetrised Dirichlet generator.  Returns dict."""
    from scipy.sparse.linalg import spsolve
    from scipy.linalg import eigh
    n = Kb.shape[0]
    N = np.setdiff1d(np.arange(n), B)
    M = Kb.tocsr()[N][:, N]
    a = np.zeros(len(N)); a[np.searchsorted(N, A)] = 1.0/len(A)
    lam = spsolve(M.T.tocsc(), a)
    out = dict(occupation=float(-lam.sum()))
    if n <= dense_limit:
        lp = logpi[N]
        Md = M.toarray()*np.exp(0.5*(lp[:, None] - lp[None, :]))
        Md = 0.5*(Md + Md.T)
        mu, V = eigh(Md)
        w = np.exp(0.5*(lp - lp.max()))           # D^(1/2) 1, scaled
        m = -(V @ ((V.T @ w)/mu))/w                # D^(-1/2) V mu^-1 V^T D^(1/2) 1
        out['eigendecomposition'] = float(a @ m)
    return out


def gap_route2(Kb, dense_limit=800):
    """Second-largest eigenvalue of the NON-symmetrised generator: sparse
    shift-invert Arnoldi (eigs), and dense analytic_lib.spectral_gap_K for small n."""
    from scipy.sparse.linalg import eigs
    sigma = 1e-9*max(1.0, abs(Kb.diagonal()).max())
    vals = eigs(Kb.tocsc(), k=4, sigma=sigma, which='LM', tol=0, maxiter=20000, return_eigenvectors=False)
    vals = np.sort(vals.real)[::-1]
    out = dict(arnoldi=float(-vals[1]))
    if Kb.shape[0] <= dense_limit:
        out['dense_eig'] = L.spectral_gap_K(Kb.toarray())
    return out


def committor_route2(Kb, pi, A, B):
    """Committor from the symmetric flux matrix W = diag(pi) K (reversible)."""
    from scipy.sparse.linalg import spsolve
    n = Kb.shape[0]
    W = (diags(pi) @ Kb).tocsr()
    W = (W + W.T)*0.5
    free = np.setdiff1d(np.arange(n), np.r_[A, B])
    q = np.zeros(n); q[B] = 1.0
    q[free] = spsolve(W[free][:, free].tocsc(), -np.asarray(W[free][:, B].sum(axis=1)).ravel())
    return q


def omega_bhs(Kb, pi, q):
    """Normalised positive reactive current in the reversible form of
    Berezhkovskii, Hummer and Szabo: J+_ij = max(pi_i K_ij (q_j - q_i), 0)."""
    c = Kb.tocoo()
    off = c.row != c.col
    i, j, k = c.row[off], c.col[off], c.data[off]
    J = np.maximum(pi[i]*k*(q[j] - q[i]), 0.0)
    J = csr_matrix((J, (i, j)), shape=Kb.shape)
    J.eliminate_zeros()
    return J/J.sum()


def last_crossing_route2(Kb, pi, A, B, q0, level=.5):
    """Last-crossing law on the fixed cut S = {q0 < level} from the forward
    occupation of the reactive (q-conditioned, Doob-transformed) jump process.

    crossings(i->j) = (expected time at i) * K_ij q_j / q_i   (launch jumps from A
    counted separately), times P^cond_j(hit B before S).  No backward committor and
    no reversed generator are used.  Returns (i, j, probability) on cut edges."""
    from scipy.sparse.linalg import spsolve
    n = Kb.shape[0]
    q = committor_route2(Kb, pi, A, B)
    K = Kb.tocsr()
    isA = np.zeros(n, bool); isA[A] = True
    isB = np.zeros(n, bool); isB[B] = True
    T = np.flatnonzero(~isA & ~isB)
    c = K.tocoo()
    off = c.row != c.col
    r, cc, k = c.row[off], c.col[off], c.data[off]
    # conditioned rates Q_ij = K_ij q_j / q_i for i in T
    keep = ~isA[r] & ~isB[r] & ~isA[cc]
    Q = csr_matrix((k[keep]*q[cc[keep]]/q[r[keep]], (r[keep], cc[keep])), shape=(n, n))
    Q = Q - diags(np.asarray(Q.sum(axis=1)).ravel())
    # launch: a in A -> j not in A with weight pi_a K_aj q_j
    la = isA[r] & ~isA[cc]
    launch_w = pi[r[la]]*k[la]*q[cc[la]]
    launch_w = launch_w/launch_w.sum()
    rho = np.zeros(n); np.add.at(rho, cc[la], launch_w)
    # occupation times on T: (-Q_TT)^T n_T = rho_T
    QT = Q[T][:, T].tocsc()
    occ = np.zeros(n)
    occ[T] = spsolve((-QT).T.tocsc(), rho[T])
    S = q0 < level
    # P^cond_j(hit B before S) for j outside S
    Uset = np.flatnonzero(~S & ~isB)
    g = np.zeros(n); g[B] = 1.0
    QU = Q[Uset][:, Uset].tocsc()
    g[Uset] = spsolve(QU, -np.asarray(Q[Uset][:, B].sum(axis=1)).ravel())
    cut = S[r] & ~S[cc]
    ii, jj, kk = r[cut], cc[cut], k[cut]
    cross = np.where(isA[ii], 0.0, occ[ii]*kk*q[jj]/np.where(q[ii] > 0, q[ii], 1.0))
    # launch jumps that cross the cut directly
    lmap = {}
    for a_, j_, w_ in zip(r[la], cc[la], launch_w):
        lmap[(a_, j_)] = w_
    cross = cross + np.array([lmap.get((a_, j_), 0.0) for a_, j_ in zip(ii, jj)])
    mass = cross*g[jj]
    return ii, jj, mass/mass.sum(), float(mass.sum())


def evaluate(sysm, biases, dense_mfpt_limit=3200, dense_gap_limit=800, verbose=True):
    """Speedups, mechanism diagnostics and second-route checks for each bias
    on one discretised system.  Conventions follow grid_sensitivity.py."""
    t_start = time.time()
    K0 = sysm.K0
    q0 = sysm.q0()
    ref = X.law(K0, sysm.pi0, sysm.A, sysm.B, q0)
    i0, j0, p0_2, tot0 = last_crossing_route2(K0, sysm.pi0, sysm.A, sysm.B, q0)
    assert np.array_equal(i0, ref['i']) and np.array_equal(j0, ref['j'])
    J0_2 = omega_bhs(K0, sysm.pi0, committor_route2(K0, sysm.pi0, sysm.A, sysm.B))
    t0 = X.mfpt(K0, sysm.A, sysm.B)
    vals0, _ = top_eigs(K0, sysm.logpi0, k=4)
    g0 = float(-vals0[1])
    m2 = mfpt_route2(K0, sysm.logpi0, sysm.A, sysm.B, dense_mfpt_limit)
    gg2 = gap_route2(K0, dense_gap_limit)
    hx = float(sysm.xs[1] - sysm.xs[0]); hy = float(sysm.ys[1] - sysm.ys[0])
    base = dict(system=sysm.name, grid=sysm.label, nx=sysm.nx, ny=sysm.ny, n=sysm.n,
                n_A=len(sysm.A), n_B=len(sysm.B), hx=hx, hy=hy,
                mfpt0=t0, gap0=g0, lam3_over_lam2_0=float(vals0[2]/vals0[1]),
                mfpt0_times_hx2=t0*hx*hx, gap0_over_hx2=g0/(hx*hx),
                n_cut_edges=int(len(ref['i'])),
                cut_effective_number=float(1/np.sum(ref['probability']**2)),
                check_mfpt0_rel={k: v/t0 - 1 for k, v in m2.items()},
                check_gap0_rel={k: v/g0 - 1 for k, v in gg2.items()},
                check_lastcrossing0_maxdiff=float(np.abs(p0_2 - ref['probability']).max()),
                check_lastcrossing0_total=tot0,
                check_current0_maxdiff=float(abs(J0_2 - ref['current']).max()))
    rows = {}
    for name, b in biases.items():
        b = np.asarray(b, float)
        Kb = tilt(K0, b)
        lp = sysm.logpi0 - b
        pib = X.boltzmann(sysm.F + b)
        lb = X.law(Kb, pib, sysm.A, sysm.B, q0)
        assert np.array_equal(lb['i'], ref['i']) and np.array_equal(lb['j'], ref['j'])
        tb = X.mfpt(Kb, sysm.A, sysm.B)
        vals, _ = top_eigs(Kb, lp, k=4)
        gb = float(-vals[1])
        m2 = mfpt_route2(Kb, lp, sysm.A, sysm.B, dense_mfpt_limit)
        gg2 = gap_route2(Kb, dense_gap_limit)
        ii, jj, p2, tot = last_crossing_route2(Kb, pib, sysm.A, sysm.B, q0)
        assert np.array_equal(ii, ref['i']) and np.array_equal(jj, ref['j'])
        Jb2 = omega_bhs(Kb, pib, committor_route2(Kb, pib, sysm.A, sysm.B))
        U = float(np.abs(b).max())
        row = dict(mfpt=tb, gap=gb, mfpt_speedup=t0/tb, gap_speedup=gb/g0,
                   lam2=float(vals[1]), lam3=float(vals[2]), lam3_over_lam2=float(vals[2]/vals[1]),
                   C_TS=float(np.minimum(ref['probability'], lb['probability']).sum()),
                   Omega_J=X.overlap_current(ref['current'], lb['current']),
                   flux_residual=lb['capacity_relative_residual'],
                   bias_max_abs=U, bias_range=float(b.max() - b.min()),
                   frac_at_bound=float(np.mean(np.abs(np.abs(b) - U) < 1e-6)),
                   check_mfpt_rel={k: v/tb - 1 for k, v in m2.items()},
                   check_gap_rel={k: v/gb - 1 for k, v in gg2.items()},
                   check_C_TS_route2=float(np.minimum(p0_2, p2).sum()),
                   check_Omega_J_BHS=float(J0_2.minimum(Jb2).sum()),
                   check_lastcrossing_total=tot,
                   check_lastcrossing_maxdiff=float(np.abs(p2 - lb['probability']).max()))
        rows[name] = row
        if verbose:
            print(f'  {sysm.name} {sysm.label} {name:>22s}: S_MFPT={row["mfpt_speedup"]:.5g} '
                  f'S_gap={row["gap_speedup"]:.5g} l3/l2={row["lam3_over_lam2"]:.4f} '
                  f'Omega_J={row["Omega_J"]:.4f} C_TS={row["C_TS"]:.4f}', flush=True)
    base['seconds'] = time.time() - t_start
    return base, rows


# ---------------------------------------------------------------- optimiser

def optimise(fg, n, U, starts, maxiter=20000, label='', verbose=True, ftol=1e-13, gtol=1e-10):
    """L-BFGS-B under the pure box |b_i| <= U from each start (as in
    per_state_ceiling.optimise_per_state_analytic / mfpt_per_state)."""
    bounds = [(-U, U)]*n
    runs = []
    for name, b0 in starts:
        t0 = time.time()
        nev = [0]

        def f(b):
            nev[0] += 1
            return fg(b)
        res = minimize(f, np.clip(b0, -U, U), jac=True, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=maxiter, maxfun=4*maxiter, gtol=gtol, ftol=ftol, maxcor=30))
        val, grad = fg(res.x)
        pg = projected_gradient(res.x, grad, U)
        runs.append(dict(start=name, fun=float(res.fun), nit=int(res.nit), nfev=nev[0],
                         message=str(res.message), success=bool(res.success),
                         proj_grad_inf=float(np.abs(pg).max()),
                         proj_grad_rel=float(np.abs(pg).max()/max(abs(val), 1e-300)),
                         seconds=time.time() - t0, x=res.x.copy()))
        if verbose:
            print(f'   [{label}] start={name:>12s} f={res.fun:.10g} nit={res.nit} '
                  f'nfev={nev[0]} |pg|/|f|={runs[-1]["proj_grad_rel"]:.2e} '
                  f'{time.time()-t0:.1f}s  {res.message}', flush=True)
    return runs


def projected_gradient(x, g, U, tol=1e-9):
    pg = g.copy()
    at_lo = x <= -U + tol
    at_hi = x >= U - tol
    pg[at_lo & (g > 0)] = 0.0      # minimisation: at lower bound a positive gradient is fine
    pg[at_hi & (g < 0)] = 0.0
    return pg


def random_starts(n, U, k, seed):
    rng = np.random.default_rng(seed)
    return [(f'random{s}', rng.uniform(-U, U, n)) for s in range(k)]


def flatten_start(sysm, U):
    """Budget-clipped flattening, as the second start in the original optimisers."""
    lp = np.log(np.clip(sysm.pi0, 1e-300, None))
    return np.clip(lp - lp.mean(), -U, U)


def mfpt_mask(s):
    """Cells whose bias changes the A->B MFPT: all cells outside B and the B cells
    adjacent to a cell outside B.  Inside B the MFPT does not depend on b."""
    isB = np.zeros(s.n, bool); isB[s.B] = True
    c = s.K0.tocoo()
    off = c.row != c.col
    adj = np.zeros(s.n, bool)
    adj[c.col[off & ~isB[c.row] & isB[c.col]]] = True
    return ~isB | adj
