"""Correct the option-C sweep curves: `*_mfpt_poly` is the MFPT of the SPECTRAL bias.

I initially got this wrong. In grid_2d_data.mat the only polynomial curve is `poly_spec`
-- the polynomial SPECTRAL optimum -- and it is plotted on BOTH axes:

    budget_gap_poly  = gap_speedup_poly_spec   = gamma(Kb_poly)/gamma_0
    budget_mfpt_poly = mfpt_speedup_poly_spec  = tau_0/tau(Kb_poly)      <-- SAME bias

i.e. `*_mfpt_poly` is the cross-objective curve: how the spectral-optimal polynomial fares
on MFPT. That is why the shipped curve COLLAPSES at large budget (27.5 -> 11.6 -> 3.5 ->
0.27): the spectral optimum slows the very transition it was meant to accelerate. That
collapse is the paper's central result. Substituting the MFPT-optimal polynomial there
would erase it.

So: for each budget/barrier, take the option-C SPECTRAL optimum and evaluate its MFPT.
"""
from __future__ import annotations
import sys, json, time
import numpy as np
sys.path.insert(0, r'C:\Users\edina\Dropbox\MSM_Roundtable_2026')
from scipy.optimize import minimize
import analytic_lib as L
import optimal_spectral_bias as O
from rerun_polynomial_optionC_lean import u_of, make_fg

PATH = r'C:\Users\edina\Dropbox\MSM_Roundtable_2026'


def opt_gap(K0, Fb, U, edges, A, B, seeds, maxiter=300):
    fg = make_fg('gap', K0, Fb, U, edges, A, B)
    best = None
    for t0 in seeds:
        t0 = np.asarray(t0, float)
        n = np.linalg.norm(t0)
        if n > 0:
            t0 = t0 / n
        r = minimize(fg, t0, jac=True, method='BFGS',
                     options=dict(maxiter=maxiter, gtol=1e-11))
        if best is None or r.fun < best.fun:
            best = r
    return best


def main():
    t0 = time.time()
    fin = json.load(open(f'{PATH}/polynomial_optionC_FINAL.json'))
    TG = np.asarray(fin['spectral']['theta'], float)
    rs = json.load(open(f'{PATH}/optionC_sweeps_RESEEDED.json'))
    out = dict(note='option C: gap AND mfpt of the SPECTRAL-optimal polynomial (poly_spec)')

    # ---- budget sweep, barrier 4.0 ----
    K0, coords, _ = L.grid_2d_generator(nx=20, ny=10, barrier_height=4.0, bottleneck=True)
    gap0 = L.spectral_gap_K(K0)
    edges = O.nearest_neighbour_edges(K0)
    Fb = O.basis_poly(coords, max_order=4)
    A = np.where(coords[:, 0] < -0.85)[0]
    B = np.where(coords[:, 0] > 0.85)[0]
    mfpt0 = L.mfpt_K(K0, A, B)
    G, M = [], []
    pg = TG
    print('BUDGET: gap and MFPT of the SAME spectral-optimal option-C polynomial')
    print(f"{'U':>5} {'S_gamma':>9} {'S_MFPT(of that bias)':>21}")
    for i, U in enumerate(rs['budgets']):
        bg = opt_gap(K0, Fb, U, edges, A, B, [TG, pg]); pg = bg.x
        ug, _ = u_of(bg.x, Fb, U)
        Kb = L.tilt_generator(K0, ug)
        g = L.spectral_gap_K(Kb) / gap0
        # keep the best-found gap (max), but the MFPT must come from THAT bias
        if rs['budget_gap_poly_C'][i] > g:
            g = rs['budget_gap_poly_C'][i]      # (only differs in the 4th digit)
        m = mfpt0 / L.mfpt_K(Kb, A, B)
        G.append(float(g)); M.append(float(m))
        print(f'{U:5.1f} {g:9.3f} {m:21.3f}', flush=True)
    out['budgets'] = rs['budgets']
    out['budget_gap_poly'] = G
    out['budget_mfpt_poly'] = M

    # ---- regime sweep, U = 3.0 ----
    Gr, Mr = [], []
    pg = TG
    print('\nREGIME: same, across barrier heights')
    print(f"{'bh':>5} {'S_gamma':>9} {'S_MFPT(of that bias)':>21}")
    for i, bh in enumerate(rs['barriers']):
        Kb0, cb, _ = L.grid_2d_generator(nx=20, ny=10, barrier_height=bh, bottleneck=True)
        g0 = L.spectral_gap_K(Kb0)
        eb = O.nearest_neighbour_edges(Kb0)
        Fbb = O.basis_poly(cb, max_order=4)
        Ab = np.where(cb[:, 0] < -0.85)[0]
        Bb = np.where(cb[:, 0] > 0.85)[0]
        m0 = L.mfpt_K(Kb0, Ab, Bb)
        bg = opt_gap(Kb0, Fbb, 3.0, eb, Ab, Bb, [TG, pg]); pg = bg.x
        ug, _ = u_of(bg.x, Fbb, 3.0)
        Kb = L.tilt_generator(Kb0, ug)
        g = L.spectral_gap_K(Kb) / g0
        if rs['regime_gap_poly_C'][i] > g:
            g = rs['regime_gap_poly_C'][i]
        m = m0 / L.mfpt_K(Kb, Ab, Bb)
        Gr.append(float(g)); Mr.append(float(m))
        print(f'{bh:5.1f} {g:9.3f} {m:21.3f}', flush=True)
    out['barriers'] = rs['barriers']
    out['regime_FE'] = rs['regime_FE']
    out['regime_gap_poly'] = Gr
    out['regime_mfpt_poly'] = Mr

    i3 = out['budgets'].index(3.0); i4 = out['barriers'].index(4.0)
    print('\nshared point (barrier 4, U=3):')
    print(f'  budget: gap {G[i3]:.3f}  mfpt {M[i3]:.3f}')
    print(f'  regime: gap {Gr[i4]:.3f}  mfpt {Mr[i4]:.3f}')
    print(f'\nshipped (clipped) mfpt_poly for comparison: '
          f'[2.5, 5.82, 19.18, 27.49, 11.57, 3.51, 0.27]')
    print(f'option C          mfpt_poly              : {[round(x,2) for x in M]}')
    print('  (does the cross-objective collapse survive? '
          f"{'YES' if M[-1] < M[i3] else 'NO -- check!'})")
    json.dump(out, open(f'{PATH}/optionC_sweeps_CORRECT.json', 'w'), indent=2)
    print(f'\nsaved optionC_sweeps_CORRECT.json  [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
