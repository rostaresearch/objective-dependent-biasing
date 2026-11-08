"""Bottleneck grid checks.

1. A/B masks: A and B are the footprints of the 20x10 cells with x < -0.85 / x > 0.85
   (x <= -0.842 / x >= 0.842). The script prints the number of columns and states on each grid,
   and compares them with a plain x < -0.85 threshold.
2. SI Table S1: Kish effective sample fraction, reweighting cost, spectral and MFPT speedups for
   the flattening bias, the saved 40x20 optima (best runs), and the optima scaled to Neff/N = 0.5.
3. Non-uniqueness: the MFPT optimum leaves the B cells without neighbours outside B undetermined.
   The script reports S_MFPT, S_gamma and Neff/N for every saved start.

Inputs: B_lib.py and B2_bottleneck_40x20_U3_{gap,mfpt}.npz in this folder.
Run: python bottleneck_tableS1_and_masks.py
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

BDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BDIR))
import B_lib as L  # noqa: E402


def masks():
    xc = np.linspace(-1, 1, 20); hc = xc[1] - xc[0]
    xa, xb = xc[xc < -.85].max() + hc / 2, xc[xc > .85].min() - hc / 2
    print(f'A/B thresholds: x <= {xa:.4f}, x >= {xb:.4f}')
    for nx, ny in [(20, 10), (40, 20), (80, 40), (160, 80)]:
        xs = np.linspace(-1, 1, nx); k = int((xs <= xa + 1e-12).sum())
        print(f'{nx}x{ny}: A columns {k}, n_A = {k*ny} (plain x < -0.85 would give {int((xs < -.85).sum())} columns)')


def table_s1():
    s = L.bottleneck_system(40, 20)
    g0 = L.gap_value(np.zeros(s.n), s); m0 = L.mfpt_value(np.zeros(s.n), s)

    def ess(b):
        p = s.pi0 * np.exp(-(b - b.min())); p /= p.sum(); w = s.pi0 / p
        return (p @ w) ** 2 / (p @ (w * w))

    def row(name, b):
        e = ess(b)
        print(f'{name:22s} Ueff {(b.max()-b.min())/2:5.2f}  Neff/N {e:.4f}  Crw {1/e:6.2f}  '
              f'S_gamma {L.gap_value(b, s)/g0:6.2f}  S_MFPT {m0/L.mfpt_value(b, s):6.2f}')

    print(f'\nTable S1 (40x20, n_A = {len(s.A)}, n_B = {len(s.B)})')
    row('flattening', L.flatten_start(s, 3))
    for o, lab in [('gap', 'spectral'), ('mfpt', 'MFPT')]:
        z = np.load(BDIR / f'B2_bottleneck_40x20_U3_{o}.npz'); b = z['best']
        row(lab, b)
        a = brentq(lambda a: ess(a * b) - 0.5, 1e-3, 1)
        row(f'{a:.3f} x {lab}', a * b)
    print('\nMFPT optimum, all saved starts (identical MFPT, different undetermined B cells)')
    z = np.load(BDIR / 'B2_bottleneck_40x20_U3_mfpt.npz')
    for k in z.files: row(k, z[k])


if __name__ == '__main__':
    masks(); table_s1()
