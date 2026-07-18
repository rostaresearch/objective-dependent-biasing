"""
Mechanism-conservation / pathway-distortion analysis (for the JCTC paper).

Compares three networks  K0,  K^(b_MFPT),  K^(b_gamma)  on the 9-well
landscape, asking whether MFPT optimisation accelerates A->B while
preserving the unbiased transition tube/current better or worse than the
spectral (lambda_2) bias.

Bias-scaling:  b(alpha) = alpha * b*,  alpha in [0,1], same grid for both.
TPT observables (committor, reactive current, transition-path density)
computed for every alpha; mechanism-overlap metrics reported against K0.

Outputs mechanism_data.mat for MATLAB plotting.
"""
from __future__ import annotations
import numpy as np
from scipy.io import loadmat, savemat
import analytic_lib as L

# ----- load the 9-well network data (F, optimal biases, coords) -----
d = loadmat("rkhs_2d_data.mat")
print("mat keys:", [k for k in d.keys() if not k.startswith("__")])
F = np.asarray(d["F"], float)            # dimensionless free energy, 20x20
N = F.shape[0]
xs = np.asarray(d["xs"], float).ravel()
ys = np.asarray(d["ys"], float).ravel()
start_xy = np.asarray(d["start_xy"], float).ravel()
end_xy   = np.asarray(d["end_xy"], float).ravel()
b_gamma = np.asarray(d["u_spec_2"], float).ravel()   # U_max = 6 kT
b_mfpt  = np.asarray(d["u_mfpt_2"], float).ravel()

# ----- rebuild K0 (4-connected, detailed balance wrt exp(-F)) -----
Fflat = F.ravel(order="C")
n = N * N
K0 = np.zeros((n, n))
def idx(ix, iy): return ix * N + iy
for ix in range(N):
    for iy in range(N):
        i = idx(ix, iy)
        for jx, jy in ((ix-1, iy), (ix+1, iy), (ix, iy-1), (ix, iy+1)):
            if 0 <= jx < N and 0 <= jy < N:
                j = idx(jx, jy)
                K0[i, j] = np.exp(-(Fflat[j] - Fflat[i]) / 2.0)
np.fill_diagonal(K0, 0.0); np.fill_diagonal(K0, -K0.sum(1))
L.check_row_generator(K0)

# state coordinates: F[row,col] with row->y, col->x (meshgrid 'xy' in builder);
# state s = row*N + col, so x = xs[col], y = ys[row].
rows = np.arange(n) // N; cols = np.arange(n) % N
cx = xs[cols]; cy = ys[rows]

def coord_to_state(xp, yp):
    col = int(np.argmin(np.abs(xs - xp))); row = int(np.argmin(np.abs(ys - yp)))
    return row * N + col
A = [coord_to_state(start_xy[0], start_xy[1])]
B = [coord_to_state(end_xy[0],   end_xy[1])]

pi0 = L.stationary_distribution_from_K(K0)
gap0 = L.spectral_gap_K(K0)
mfpt0 = L.mfpt_K(K0, A, B)
print(f"n={n}  gap0={gap0:.3e}  mfpt0={mfpt0:.3e}  A={A} B={B}")

# ----- TPT current + transition-path density for a generator -----
def tpt(K, pi):
    q = L.committor_K(K, A, B)
    gross = (pi[:, None] * K) * ((1 - q)[:, None]) * (q[None, :])
    np.fill_diagonal(gross, 0.0)
    net = gross - gross.T
    Jpos = np.maximum(net, 0.0)
    tot = Jpos.sum()
    Jhat = Jpos / tot if tot > 0 else Jpos
    rho = pi * q * (1 - q)
    s = rho.sum(); rho = rho / s if s > 0 else rho
    return q, Jhat, rho

def overlap(p, qd):
    p = p / p.sum() if p.sum() > 0 else p
    qd = qd / qd.sum() if qd.sum() > 0 else qd
    return float(np.minimum(p, qd).sum())

def flux_field(Jhat):
    """per-state net out-current vector (for quiver)."""
    fx = np.zeros(n); fy = np.zeros(n)
    # vector from i to j weighted by Jhat[i,j]
    I, J = np.nonzero(Jhat)
    for i, j in zip(I, J):
        w = Jhat[i, j]
        fx[i] += w * (cx[j] - cx[i]); fy[i] += w * (cy[j] - cy[i])
    return fx, fy

# reference TPT
q0, J0, rho0 = tpt(K0, pi0)

# ----- alpha scan for both biases -----
alphas = np.linspace(0.0, 1.0, 21)
def scan(bstar):
    out = dict(mfpt_su=[], spec_su=[], OmegaJ=[], Omegarho=[], Dq=[], Dedge=[], Dedge95=[])
    for a in alphas:
        b = a * bstar
        Kb = L.tilt_generator(K0, b)
        pib = L.stationary_distribution_from_K(Kb)
        out["mfpt_su"].append(mfpt0 / L.mfpt_K(Kb, A, B))
        out["spec_su"].append(L.spectral_gap_K(Kb) / gap0)
        qb, Jb, rhob = tpt(Kb, pib)
        out["OmegaJ"].append(overlap(J0, Jb))
        out["Omegarho"].append(overlap(rho0, rhob))
        out["Dq"].append(float(np.sqrt(np.sum(pi0 * (qb - q0) ** 2))))
        # current-weighted edge distortion on the UNBIASED current
        absdb = np.abs(b[None, :] - b[:, None]) * 0.5      # |Delta log K|
        w = J0 / J0.sum()
        out["Dedge"].append(float(np.sum(w * absdb)))
        # 95th percentile of |Delta log K| weighted by J0
        vals = absdb[J0 > 0]; wv = (J0[J0 > 0]); wv = wv / wv.sum()
        order = np.argsort(vals); cdf = np.cumsum(wv[order])
        out["Dedge95"].append(float(vals[order][np.searchsorted(cdf, 0.95)]))
    return {k: np.asarray(v) for k, v in out.items()}

print("scanning MFPT bias..."); S_m = scan(b_mfpt)
print("scanning gamma bias..."); S_g = scan(b_gamma)

# full-strength current fields + edge scatter (alpha = 1)
Kb_m = L.tilt_generator(K0, b_mfpt); pim = L.stationary_distribution_from_K(Kb_m)
Kb_g = L.tilt_generator(K0, b_gamma); pig = L.stationary_distribution_from_K(Kb_g)
qm, Jm, rhom = tpt(Kb_m, pim)
qg, Jg, rhog = tpt(Kb_g, pig)
fx0, fy0 = flux_field(J0)
fxm, fym = flux_field(Jm)
fxg, fyg = flux_field(Jg)

# edge scatter: edges present in either J0 or Jb
def scatter_pairs(Jb, b):
    I, J = np.nonzero((J0 > 0) | (Jb > 0))
    j0 = J0[I, J]; jb = Jb[I, J]; db = b[J] - b[I]
    return np.column_stack([j0, jb, db])
sc_m = scatter_pairs(Jm, b_mfpt)
sc_g = scatter_pairs(Jg, b_gamma)

savemat("mechanism_data.mat", dict(
    alphas=alphas, gap0=gap0, mfpt0=mfpt0,
    cx=cx, cy=cy, xs=xs, ys=ys, F=F, start_xy=start_xy, end_xy=end_xy,
    # scan curves
    m_mfpt_su=S_m["mfpt_su"], m_spec_su=S_m["spec_su"], m_OmegaJ=S_m["OmegaJ"],
    m_Omegarho=S_m["Omegarho"], m_Dq=S_m["Dq"], m_Dedge=S_m["Dedge"], m_Dedge95=S_m["Dedge95"],
    g_mfpt_su=S_g["mfpt_su"], g_spec_su=S_g["spec_su"], g_OmegaJ=S_g["OmegaJ"],
    g_Omegarho=S_g["Omegarho"], g_Dq=S_g["Dq"], g_Dedge=S_g["Dedge"], g_Dedge95=S_g["Dedge95"],
    # full-strength fields
    fx0=fx0, fy0=fy0, fxm=fxm, fym=fym, fxg=fxg, fyg=fyg,
    q0=q0, qm=qm, qg=qg, rho0=rho0,
    scatter_mfpt=sc_m, scatter_gamma=sc_g,
))
print("saved mechanism_data.mat")
print(f"\nAt full strength (alpha=1, U_max=6):")
print(f"  MFPT bias : MFPT x{S_m['mfpt_su'][-1]:.3g}  gamma x{S_m['spec_su'][-1]:.3g}  "
      f"OmegaJ={S_m['OmegaJ'][-1]:.2f}  Omega_rho={S_m['Omegarho'][-1]:.2f}  Dedge={S_m['Dedge'][-1]:.2f}")
print(f"  gamma bias: MFPT x{S_g['mfpt_su'][-1]:.3g}  gamma x{S_g['spec_su'][-1]:.3g}  "
      f"OmegaJ={S_g['OmegaJ'][-1]:.2f}  Omega_rho={S_g['Omegarho'][-1]:.2f}  Dedge={S_g['Dedge'][-1]:.2f}")
