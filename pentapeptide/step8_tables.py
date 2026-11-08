"""Build the README tables (tables_C.md) from the JSON outputs, including the
worst-case agreement between each primary route and its independent check."""
from __future__ import annotations

import json

import numpy as np

import C_lib as C

R = json.loads((C.HERE / 'results_C_pentapeptide.json').read_text())
S1 = json.loads((C.HERE / 'step1_T_check.json').read_text())
MC = json.loads((C.HERE / 'sampling_check.json').read_text())
ST = json.loads((C.HERE / 'structure_C.json').read_text())


def row(U, name, m):
    return (f"| {U} | {name} | {m['gap_speedup']:.2f} | {m['mfpt_AB_speedup']:.2f} | "
            f"{m['mfpt_BA_speedup']:.3g} | {m['Omega_J']:.3f} | {m['C_TS']:.3f} | "
            f"{m['IPR_biased_last_crossing']:.1f} | {m['ESS_over_N']:.3f} |")


HDR = ("| U | Bias | Gap speedup | A→B MFPT speedup | B→A MFPT speedup | Ω_J | C_TS | "
       "IPR of biased last crossings | ESS/N |\n|---|---|---:|---:|---:|---:|---:|---:|---:|")


def all_metric_dicts():
    out = []
    for U in ('U1', 'U2', 'U3'):
        out += list(R['cross_objective'][U].values())
    rb = R['robustness_U2']
    out += [v for v in rb['alternative_AB'].values() if isinstance(v, dict)]
    out += [rb['direction_BA']['mfpt_BA_optimum']]
    out += [rb['generator_K2']['spectral_optimum'], rb['generator_K2']['mfpt_AB_optimum']]
    out += list(rb['generator_K2']['K_optima_evaluated_on_K2'].values())
    return out


def main():
    L = []
    L.append('### Cross-objective table, primary A/B (A = state 189, B = state 219)\n')
    L.append(HDR)
    for U in (1, 2, 3):
        co = R['cross_objective'][f'U{U}']
        L.append(row(U, 'spectral optimum', co['spectral_optimum']))
        L.append(row(U, 'A→B MFPT optimum', co['mfpt_AB_optimum']))
    L.append('')
    L.append('Fraction of the attainable speedup that each optimum reaches on the other objective:\n')
    L.append('| U | spectral optimum: A→B MFPT speedup / best A→B MFPT speedup | '
             'MFPT optimum: gap speedup / best gap speedup |\n|---|---:|---:|')
    for U in (1, 2, 3):
        co = R['cross_objective'][f'U{U}']
        s, m = co['spectral_optimum'], co['mfpt_AB_optimum']
        L.append(f"| {U} | {s['mfpt_AB_speedup']:.2f} / {m['mfpt_AB_speedup']:.2f} = "
                 f"{s['mfpt_AB_speedup'] / m['mfpt_AB_speedup']:.2f} | "
                 f"{m['gap_speedup']:.2f} / {s['gap_speedup']:.2f} = {m['gap_speedup'] / s['gap_speedup']:.2f} |")
    L.append('')
    rb = R['robustness_U2']
    L.append('### Robustness at U = 2\n')
    L.append(HDR)
    co = R['cross_objective']['U2']
    L.append(row(2, 'primary A/B: spectral optimum', co['spectral_optimum']))
    L.append(row(2, 'primary A/B: A→B MFPT optimum', co['mfpt_AB_optimum']))
    a = rb['alternative_AB']
    L.append(row(2, 'PCCA+ A/B: spectral optimum (same bias)', a['spectral_optimum']))
    L.append(row(2, 'PCCA+ A/B: A→B MFPT optimum', a['mfpt_AB_optimum']))
    L.append(row(2, 'PCCA+ A/B: primary MFPT optimum applied', a['primary_mfpt_optimum_evaluated_on_alternative']))
    L.append('')
    L.append('### Extra checks outside the requested scope (U = 2)\n')
    L.append(HDR)
    L.append(row(2, 'primary A/B: B→A MFPT optimum', rb['direction_BA']['mfpt_BA_optimum']))
    k = rb['generator_K2']
    L.append(row(2, 'K2: spectral optimum', k['spectral_optimum']))
    L.append(row(2, 'K2: A→B MFPT optimum', k['mfpt_AB_optimum']))
    L.append(row(2, 'K2: K spectral optimum applied', k['K_optima_evaluated_on_K2']['spectral']))
    L.append(row(2, 'K2: K MFPT optimum applied', k['K_optima_evaluated_on_K2']['mfpt_AB']))
    L.append('')

    g = ST['last_crossing_by_destination_macrostate']
    L.append('### Last crossings grouped by the macrostate entered (primary A/B)\n')
    hdr = ' | '.join(f"into PCCA+ {d} ({n} edges)" for d, n in zip(g['destination_macrostates'],
                                                              g['n_cut_edges_per_destination']))
    L.append(f'| Bias | {hdr} | overlap with unbiased after grouping | C_TS (43 edges) |')
    L.append('|---|' + '---:|' * (len(g['destination_macrostates']) + 2))
    L.append('| unbiased | ' + ' | '.join(f'{w:.3f}' for w in g['unbiased']) + ' | 1 | 1 |')
    for U in (1, 2, 3):
        for key, name in ((f'spectral_U{U}', 'spectral_optimum'), (f'mfpt_AB_U{U}', 'mfpt_AB_optimum')):
            w = g[key]
            label = f"U = {U}, {'spectral' if 'spectral' in key else 'A→B MFPT'} optimum"
            L.append(f'| {label} | ' + ' | '.join(f'{x:.3f}' for x in w['weights']) +
                     f" | {w['overlap_with_unbiased']:.3f} | {R['cross_objective'][f'U{U}'][name]['C_TS']:.3f} |")
    L.append('')

    # ---------------- cross-check agreement
    ms = all_metric_dicts()
    rel = lambda x, y: abs(x - y) / max(abs(y), 1e-300)
    gap = max(rel(m['check']['gap_sym'], m['gap']) for m in ms)
    mab = max(rel(m['check']['mfpt_AB_fundamental'], m['mfpt_AB']) for m in ms)
    mba = max(rel(m['check']['mfpt_BA_fundamental'], m['mfpt_BA']) for m in ms)
    oj1 = max(abs(m['check']['Omega_J_bhs'] - m['Omega_J']) for m in ms)
    oj2 = max(abs(m['check']['Omega_J_law_current'] - m['Omega_J']) for m in ms)
    e1 = max(abs(m['check']['ESS_closed_form'] - m['ESS_over_N']) for m in ms)
    e2 = max(abs(m['check']['ESS_analytic_lib'] - m['ESS_over_N']) for m in ms)
    pib = max(m['pib_route_difference'] for m in ms)
    cap = max(m['capacity_relative_residual'] for m in ms)
    sp = R['optimizer_spread']
    gc = R['gradient_checks']
    kk = [m['kkt']['rel_projected_grad'] for m in ms if 'kkt' in m]
    prim = {k: v for k, v in MC['primary'].items() if k != 'two_sample_noise_unbiased'}
    allmc = list(prim.values()) + list(MC['alternative'].values())
    L.append('### Cross-checks: worst case over every bias reported above\n')
    L.append('| Quantity | Primary route | Independent route | Worst disagreement |\n|---|---|---|---:|')
    L.append(f'| Spectral gap | `analytic_lib.spectral_gap_K` (eig of K^b) | eigh of D^½ K^b D^-½ | {gap:.1e} (relative) |')
    L.append(f'| A→B and B→A MFPT | `analytic_lib.mfpt_K` (linear solve) | fundamental matrix / eigendecomposition of the symmetrised block | {max(mab, mba):.1e} (relative) |')
    L.append(f"| A→B MFPT | as above | Monte Carlo, {MC['primary']['unbiased']['n_paths']} first-passage trajectories per bias | "
             f"max abs z = {max(abs(m['mfpt_z']) for m in allmc):.2f} |")
    L.append(f'| Biased stationary distribution | π⁰e^(−b) normalised | eigenvector of K^b | {pib:.1e} |')
    L.append(f'| Ω_J | `positive_net_current` (π_i K_ij (1−q_i) q_j form) | BHS form max(π_i K_ij (q_j−q_i), 0); `exact_last_crossing.law` current (backward committor from the time-reversed generator) | {max(oj1, oj2):.1e} |')
    L.append(f"| Edge currents behind Ω_J | J_ij = π_i K_ij (q_j−q_i) | Monte Carlo net traversals per reactive path, top {allmc[0]['edge_current_test']['n_edges']} edges, {allmc[0]['edge_current_test']['n_batches']} batches | "
             f"max abs z = {max(m['edge_current_test']['max_abs_z'] for m in allmc):.2f}, rms z ≤ {max(m['edge_current_test']['rms_z'] for m in allmc):.2f} |")
    biased = [m for m in allmc if m['C_TS_exact'] < 0.999]
    L.append(f"| C_TS | `exact_last_crossing.law` | Monte Carlo last crossings (C_TS of the sampled biased distribution against the exact unbiased one) | "
             f"max abs difference {max(abs(m['C_TS_exact_ref_vs_sampled_bias'] - m['C_TS_exact']) for m in biased):.4f} over the biased networks; "
             f"sampled-vs-exact TV ≤ {max(m['last_crossing_TV_sampled_vs_exact'] for m in allmc):.3f} for all networks (sampling noise) |")
    L.append(f'| Last-crossing total | Σ L_ij | reactive capacity from the committor | {cap:.1e} (relative) |')
    L.append(f'| ESS/N | 1/Σ π^b (π⁰/π^b)² | 1/(⟨e^−b⟩⟨e^b⟩) over π⁰; `analytic_lib.reweighting_ess` | {max(e1, e2):.1e} |')
    gs = max(v['spectral']['max_rel_err_fd'] for v in gc.values())
    gm = max(v['mfpt']['max_rel_err_fd'] for v in gc.values())
    gsym = max(v['spectral']['max_rel_diff_symmetric_route_all_components'] for v in gc.values())
    L.append(f'| Spectral-gap gradient | `per_state_ceiling.gap_and_grad` | central finite differences (ε = 1e-5, 24 components, 5 points); symmetric-route formula, all components | FD {gs:.1e}; symmetric {gsym:.1e} (relative) |')
    L.append(f'| MFPT gradient | `mfpt_per_state.mfpt_and_grad` | central finite differences (ε = 1e-5, all 250 components, 5 points) | {gm:.1e} (relative) |')
    L.append(f'| First-order optimality | — | projected gradient on the box | ≤ {max(kk):.1e} of the largest gradient component |')
    ssp = max(sp[k]['spread_rel'] for k in sp if 'spectral' in k)
    msp = max(sp[k]['spread_rel'] for k in sp if 'mfpt' in k)
    L.append(f'| Spectral optimum value | imported `optimise_per_state_analytic` (flatten and zero starts) | symmetric-route engine: 6 random starts, lower-U warm start, soft-min continuation | spread {ssp:.1e} (relative) |')
    L.append(f'| MFPT optimum value | imported `optimise_mfpt_per_state` (three starts) | 6 random starts with L-BFGS-B; TNC from zero and from the best point | spread {msp:.1e} (relative) |')
    L.append('')
    nz = MC['primary']['two_sample_noise_unbiased']
    L.append(f"Two independent unbiased Monte Carlo samples of {MC['primary']['unbiased']['n_paths']} paths overlap at "
             f"C_TS = {nz['C_TS_two_samples']:.3f} and Ω_J = {nz['Omega_J_two_samples']:.3f}. Edge-level currents over "
             f"{S1['T']['nonzero_offdiagonal']} directed edges are not resolved by this many paths, "
             "so the check of Ω_J rests on the three formulas and the edge test above.")
    (C.HERE / 'tables_C.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
