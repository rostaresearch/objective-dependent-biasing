"""Regenerate the 6-panel spectral-preconditioning figure (Fig. 3, fig:spectral).

Authoritative fix for the v4 bundle: the manuscript's Figure 3 caption describes
a six-panel figure (a: unbiased F, b: budget-clipped flattening, c: smooth
polynomial optimum, d: biased FE, e: per-state ceiling, f: reactive-current
ratio) but the \\includegraphics was pointed at fig_grid2d_bias_profiles (the
4-panel Fig. 4 image).

This script rebuilds the intended 6-panel figure entirely from authoritative
stored data on the n=200 (20x10) grid, so the plotted speedups match the
manuscript exactly:
  - F, pi0, u_poly_spec (69.0x), u_ps_spec (91.3x)  : grid_2d_data.mat
  - budget-clipped flattening speedup (35.3x)        : grid_2d_data.mat budget sweep @ U=3
  - reactive currents (flux under poly vs unbiased)  : spectral_results.json
No re-optimisation is performed (the archived spectral_results.json per-state
run reached only 55.4x, a worse local optimum than the archived .mat ceiling of
91.3x used in the manuscript), so the figure is guaranteed consistent with the
text.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat

import os
# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
d = loadmat(f"{PATH}/grid_2d_data.mat")
j = json.load(open(f"{PATH}/spectral_results.json"))

nx, ny = int(d["nx"].ravel()[0]), int(d["ny"].ravel()[0])   # 20 x 10
U_max = float(d["U_max"].ravel()[0])                         # 3.0
pi0 = np.asarray(d["pi0"], float)
F = np.asarray(d["F"], float); F = F - F.min()
u_poly = np.asarray(d["u_poly_spec"], float)
u_ps = np.asarray(d["u_ps_spec"], float)
gap0 = float(d["gap0"].ravel()[0])
tau0 = 1.0 / gap0

# authoritative speedups (match manuscript 35.3 / 69.0 / 91.3)
budget_U = d["budget_U"].ravel()
i3 = int(np.argmin(np.abs(budget_U - 3.0)))
sp_flat = float(d["budget_gap_flat"].ravel()[i3])           # 35.27
sp_poly = float(d["speedup_poly_spec"].ravel()[0])          # 68.98
sp_ps = float(d["speedup_ps_spec"].ravel()[0])              # 91.30

# budget-clipped flattening bias (centred then clipped) -> reshape to grid
logpi = np.log(np.clip(pi0.ravel(order="C"), 1e-300, None))
u_flat = np.clip(logpi - logpi.mean(), -U_max, U_max).reshape(nx, ny)

# biased/effective FE under the polynomial optimum.
# Convention: pi^(b) ∝ pi^(0) e^{-b}  =>  F^(b) = F^(0) + b  (up to an additive const).
# (F - u would DEEPEN the basins; F + u fills them, matching the caption.)
F_poly = F + u_poly; F_poly = F_poly - F_poly.min()

# reactive current ratio (per state) under polynomial optimum vs unbiased
flux_unb = np.asarray(j["flux_unb"], float)
flux_poly = np.asarray(j["flux_poly"], float)
ratio = (flux_poly + 1e-30) / (flux_unb + 1e-30)
log_ratio = np.log10(ratio).reshape(nx, ny)

# ---- coords for pcolor axes (match grid_2d_generator convention) ----
xg = np.asarray(d["xs"], float).ravel()   # length nx
yg = np.asarray(d["ys"], float).ravel()   # length ny
X, Y = np.meshgrid(xg, yg, indexing="ij")

fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.6), constrained_layout=True)

ax = axes[0, 0]
c = ax.contourf(X, Y, F, levels=18, cmap="viridis")
ax.set_title(f"(a) Unbiased free energy $F$\n"
             f"spec gap = {gap0:.2e}, $\\tau_0$ = {tau0:.0f}")
plt.colorbar(c, ax=ax, label=r"$F\;[k_BT]$")

ax = axes[0, 1]
c = ax.contourf(X, Y, u_flat, levels=18, cmap="coolwarm", vmin=-U_max, vmax=U_max)
ax.set_title(f"(b) Budget-clipped flattening\n"
             fr"$u=\mathrm{{clip}}_{{[-{U_max:.0f},{U_max:.0f}]}}(\log\pi)$,"
             f" speedup = {sp_flat:.1f}x")
plt.colorbar(c, ax=ax, label=r"$u\;[k_BT]$")

ax = axes[0, 2]
c = ax.contourf(X, Y, u_poly, levels=18, cmap="coolwarm", vmin=-U_max, vmax=U_max)
ax.set_title(f"(c) Smooth polynomial optimum (deg 4)\n"
             fr"$|u|\leq{U_max:.0f}$, speedup = {sp_poly:.1f}x")
plt.colorbar(c, ax=ax, label=r"$u\;[k_BT]$")

ax = axes[1, 0]
c = ax.contourf(X, Y, F_poly, levels=18, cmap="viridis")
ax.set_title(r"(d) Biased free energy $F^{b}=F+u$ (polynomial opt)")
plt.colorbar(c, ax=ax, label=r"$F^b\;[k_BT]$")

ax = axes[1, 1]
c = ax.contourf(X, Y, u_ps, levels=18, cmap="coolwarm", vmin=-U_max, vmax=U_max)
ax.set_title(f"(e) Per-state ceiling (graph-theoretic)\n"
             fr"$n$ params, speedup = {sp_ps:.1f}x")
plt.colorbar(c, ax=ax, label=r"$u\;[k_BT]$")

ax = axes[1, 2]
c = ax.contourf(X, Y, log_ratio, levels=18, cmap="RdBu_r", vmin=-2, vmax=2)
ax.set_title(r"(f) Reactive current ratio"
             "\n" r"$\log_{10}(\Phi^{\rm poly}/\Phi^{\rm unb})$ per state")
plt.colorbar(c, ax=ax, label=r"$\log_{10}$ ratio")

fig.suptitle(r"Optimal one-shot spectral preconditioning bias on the 2D grid "
             r"$|u|\leq 3\,k_BT$ (symmetric budget, zero-mean)", fontsize=11)
fig.savefig(f"{PATH}/fig_spectral_bias.png", dpi=170, bbox_inches="tight")
fig.savefig(f"{PATH}/fig_spectral_bias.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved fig_spectral_bias.png/.pdf  (n={nx*ny}, "
      f"flat={sp_flat:.1f}x poly={sp_poly:.1f}x ceiling={sp_ps:.1f}x)")
