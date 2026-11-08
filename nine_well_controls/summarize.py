"""Merge parts 1-4 into results.json and write the README tables (tables.md)."""
from __future__ import annotations

import json

import numpy as np

from common import HERE, systems, reproduce_audit, to_jsonable

SYS = {'nine_well': 'Nine-well', 'bottleneck_grid': 'Bottleneck'}
BL = {'MFPT_U3': 'MFPT, U = 3', 'spectral_U3': 'spectral, U = 3', 'MFPT_U6': 'MFPT, U = 6',
      'spectral_U6': 'spectral, U = 6', 'polynomial_spectral_U3': 'polynomial spectral, U = 3'}
CL = {'top50': '+U on top-50% last-crossing cells', 'channel': '+U on main channel region',
      'channel_pushpull': '+U on channel, -U on flanking rows'}


def f(x, d=3):
    if x is None:
        return '-'
    if isinstance(x, str):
        return x
    if abs(x) >= 1000:
        return f'{x:,.0f}'
    if abs(x) >= 100:
        return f'{x:.0f}'
    if abs(x) >= 10:
        return f'{x:.1f}'
    return f'{x:.{d}g}' if abs(x) < .01 else f'{x:.{d}f}'


def main():
    S = systems()
    rows, worst = reproduce_audit(S)
    p1 = json.loads((HERE/'results_part1.json').read_text())
    p2 = json.loads((HERE/'results_part2.json').read_text())
    p3 = json.loads((HERE/'results_part3.json').read_text())
    p4 = json.loads((HERE/'results_part4.json').read_text())
    p1b = json.loads((HERE/'results_part1b.json').read_text())
    out = dict(
        description='Calibration of C_TS and Omega_J (noise ceiling, random-bias reference, positive '
                    'controls) and reweighting cost of each bias. All values are exact or sampled results on '
                    'the analytic network models; none are MD measurements.',
        reproduction_of_TS_audit=dict(rows=rows, worst_relative_error=worst),
        part1_noise_ceiling=p1, part1b_biased_own_noise=p1b, part2_random_biases=p2, part3_positive_controls=p3, part4_ess=p4,
        extra_checks=json.loads((HERE/'results_checks.json').read_text()),
        small_N_check=json.loads((HERE/'check_small_N.json').read_text()),
        part5_random_routes=json.loads((HERE/'results_part5.json').read_text()))
    p5 = out['part5_random_routes']
    (HERE/'results.json').write_text(json.dumps(to_jsonable(out), indent=1)+'\n')

    L = []
    L.append('### Reproduction of the TS-audit table\n')
    L.append('| System | Bias | Omega_J | C_TS | MFPT speedup | gap speedup | max rel. error vs audit |')
    L.append('|---|---|---:|---:|---:|---:|---:|')
    for r in rows:
        L.append(f"| {SYS[r['system']]} | {BL[r['bias']]} | {r['Omega_J']:.5f} | {r['C_TS']:.5f} | "
                 f"{f(r['mfpt_speedup'])} | {f(r['gap_speedup'])} | {max(r['relative_error_vs_audit'].values()):.1e} |")
    L.append('\n### Part 1: two-sample noise band (unbiased vs unbiased), median [5%, 95%]\n')
    L.append('| System | N | C_TS | C_TS, multinomial route | C_TS, large-N formula (mean) | Omega_J | Omega_J, large-N formula (mean) | Omega_J sampled mean |')
    L.append('|---|---:|---|---|---:|---|---:|---:|')
    for name, res in p1['systems'].items():
        for N in p1['N_values']:
            n = res['noise'][str(N)]
            L.append(f"| {SYS[name]} | {N:,} | {n['C_TS']['median']:.4f} [{n['C_TS']['p05']:.4f}, {n['C_TS']['p95']:.4f}] | "
                     f"{n['C_TS_multinomial']['median']:.4f} [{n['C_TS_multinomial']['p05']:.4f}, {n['C_TS_multinomial']['p95']:.4f}] | "
                     f"{n['C_TS_clt_mean']:.4f} | {n['Omega_J']['median']:.4f} [{n['Omega_J']['p05']:.4f}, {n['Omega_J']['p95']:.4f}] | "
                     f"{n['Omega_J_clt_mean']:.4f} | {n['Omega_J']['mean']:.4f} |")
    L.append('\n### Part 1: sampled biased vs sampled unbiased, median [5%, 95%] at N = 10,000 and 100,000, and the exact-law value\n')
    L.append('| System | Bias | C_TS exact | C_TS, N=1e4 | C_TS, N=1e5 | Omega_J exact | Omega_J, N=1e4 | Omega_J, N=1e5 |')
    L.append('|---|---|---:|---|---|---:|---|---|')
    for name, res in p1['systems'].items():
        for key, rb in res['biased'].items():
            a, b = rb['by_N']['10000'], rb['by_N']['100000']
            L.append(f"| {SYS[name]} | {BL[key]} | {rb['exact']['C_TS']:.4f} | {a['C_TS']['median']:.4f} [{a['C_TS']['p05']:.4f}, {a['C_TS']['p95']:.4f}] | "
                     f"{b['C_TS']['median']:.4f} [{b['C_TS']['p05']:.4f}, {b['C_TS']['p95']:.4f}] | "
                     f"{rb['exact']['Omega_J']:.4f} | {a['Omega_J']['median']:.4f} [{a['Omega_J']['p05']:.4f}, {a['Omega_J']['p95']:.4f}] | "
                     f"{b['Omega_J']['median']:.4f} [{b['Omega_J']['p05']:.4f}, {b['Omega_J']['p95']:.4f}] |")
    L.append('\n### Part 1b: own noise of each biased ensemble (biased vs biased) at N = 10,000, and first N at which the biased-vs-unbiased band lies below the noise bands\n')
    L.append('| System | Bias | exact mean jumps per path | C_TS own noise, N=1e4 | Omega_J own noise, N=1e4 | Omega_J own noise, large-N formula | C_TS separates at N (unbiased band / both bands) | Omega_J separates at N (unbiased band / both bands) |')
    L.append('|---|---|---:|---|---|---:|---|---|')
    sep = lambda x: f(x) if x else 'not by 1e5'
    for name, res in p1b['systems'].items():
        for key, rb in res.items():
            a = rb['by_N']['10000']; b1 = p1['systems'][name]['biased'][key]
            L.append(f"| {SYS[name]} | {BL[key]} | {rb['exact_mean_jumps']:.0f} | "
                     f"{a['C_TS_own_noise']['median']:.4f} [{a['C_TS_own_noise']['p05']:.4f}, {a['C_TS_own_noise']['p95']:.4f}] | "
                     f"{a['Omega_J_own_noise']['median']:.4f} [{a['Omega_J_own_noise']['p05']:.4f}, {a['Omega_J_own_noise']['p95']:.4f}] | "
                     f"{a['Omega_J_own_noise_clt_mean']:.4f} | {sep(b1['C_TS_first_separated_N'])} / {sep(rb['C_TS_first_separated_N_both_bands'])} | "
                     f"{sep(b1['Omega_J_first_separated_N'])} / {sep(rb['Omega_J_first_separated_N_both_bands'])} |")
    L.append(f"\n### Part 2: random biases, median [5%, 95%] over M = {p2['M']} draws\n")
    L.append('| System, U | Kind | MFPT speedup | gap speedup | Omega_J | C_TS | ESS/N | fraction with MFPT speedup > 1 |')
    L.append('|---|---|---|---|---|---|---|---:|')
    for ck, cfg in p2['configs'].items():
        for kind, kd in cfg['kinds'].items():
            pc = kd['percentiles']
            cells = ' | '.join(f"{f(pc[m]['p50'])} [{f(pc[m]['p05'])}, {f(pc[m]['p95'])}]"
                               for m in ['mfpt_speedup', 'gap_speedup', 'Omega_J', 'C_TS', 'ess_frac'])
            L.append(f"| {SYS[cfg['system']]}, {cfg['U']:g} | {kind} | {cells} | {kd['fraction_mfpt_speedup_gt1']:.3f} |")
    L.append('\n### Part 2: percentile rank of each optimized bias within the random distributions (i.i.d. / smooth l = spacing / smooth l = spacing/3)\n')
    L.append('| System | Bias | MFPT speedup | gap speedup | Omega_J | C_TS | ESS/N |')
    L.append('|---|---|---|---|---|---|---|')
    for ck, cfg in p2['configs'].items():
        for key, o in cfg['optimized'].items():
            pr = o['percentile_rank']
            cells = ' | '.join(f"{f(o['metrics'][m])}: " + ' / '.join(f"{pr[k][m]:.1f}" for k in ['iid', 'grf_long', 'grf_short'])
                               for m in ['mfpt_speedup', 'gap_speedup', 'Omega_J', 'C_TS', 'ess_frac'])
            L.append(f"| {SYS[cfg['system']]} | {BL[key]} | {cells} |")
    L.append('\n### Part 3: positive controls (exact network values; sampled overlap at N = 1,000 as median [5%, 95%])\n')
    L.append('| System | Control | cells | last-crossing mass on cells | C_TS | Omega_J | MFPT speedup | gap speedup | ESS/N | sampled C_TS, N=1e3 | sampled Omega_J, N=1e3 | separates from noise at N (C_TS / Omega_J) |')
    L.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|')
    for name, row in p3['systems'].items():
        for key, c in row['controls'].items():
            base, U = key.rsplit('_U', 1)
            e = c['exact']; sm = c['sampled']['1000']
            L.append(f"| {SYS[name]} | {CL[base]}, U = {U} | {len(c['cells'])} | {c['last_crossing_mass_of_cells']:.3f} | "
                     f"{e['C_TS']:.3f} | {e['Omega_J']:.3f} | {f(e['mfpt_speedup'])} | {f(e['gap_speedup'])} | {f(e['ess_frac'])} | "
                     f"{sm['C_TS']['median']:.3f} [{sm['C_TS']['p05']:.3f}, {sm['C_TS']['p95']:.3f}] | "
                     f"{sm['Omega_J']['median']:.3f} [{sm['Omega_J']['p05']:.3f}, {sm['Omega_J']['p95']:.3f}] | "
                     f"{c['C_TS_first_separated_N'] or 'not by 1e4'} / {c['Omega_J_first_separated_N'] or 'not by 1e4'} |")
    L.append('\n### Part 4: reweighting cost of the optimized biases\n')
    L.append('| System | Bias | MFPT speedup | ESS/N | chi^2 | max w | Var log w | repo function, rel. diff. |')
    L.append('|---|---|---:|---:|---:|---:|---:|---:|')
    for k, e in p4['optimized'].items():
        name, key = k.split('/')
        su = next(r['mfpt_speedup'] for r in rows if r['system'] == name and r['bias'] == key)
        L.append(f"| {SYS[name]} | {BL[key]} | {f(su)} | {f(e['ess_frac'])} | {f(e['chi2'])} | {f(e['max_w'])} | "
                 f"{f(e['var_logw'])} | {e['relative_difference_vs_repo']:.1e} |")
    L.append('\nDirect Monte Carlo check (Kish ESS from n i.i.d. draws of pib, 40 repeats):\n')
    L.append('| Case | n | exact ESS/N | MC median [5%, 95%] | MC mean +- s.e. |')
    L.append('|---|---:|---:|---|---|')
    for k, m in p4['checks']['monte_carlo'].items():
        L.append(f"| {k} | {m['n_samples']:,} | {m['exact']:.5f} | {m['mc_median']:.5f} [{m['mc_p05']:.5f}, {m['mc_p95']:.5f}] | "
                 f"{m['mc_mean']:.5f} +- {m['mc_sem']:.5f} |")
    L.append('\n### Part 5: routes that ignore the landscape, against the true unbiased laws\n')
    L.append('| System | Route | N | C_TS | Omega_J |')
    L.append('|---|---|---:|---|---|')
    for name, r5 in p5['systems'].items():
        fl = r5['flat']
        L.append(f"| {SYS[name]} | flat landscape, F = 0 (exact) | - | {fl['C_TS']:.3f} | {fl['Omega_J']:.3f} |")
        for N in ['1', '10', '100', '1000', '10000', '30000']:
            r = r5['lerw'][N]
            L.append(f"| {SYS[name]} | random lattice routes | {int(N):,} | {r['C_TS']['median']:.3f} [{r['C_TS']['p05']:.3f}, {r['C_TS']['p95']:.3f}] | "
                     f"{r['Omega_J']['median']:.3f} [{r['Omega_J']['p05']:.3f}, {r['Omega_J']['p95']:.3f}] |")
    L.append('\nChecks for part 5:\n')
    L.append('| System | flat via tilt b = -F (C_TS / Omega_J) | Kirchhoff current vs flat law (max abs) | pooled routes vs Kirchhoff (Omega_J) | pooled routes vs flat law (Omega_J) | Wilson vs loop-erased walks, last crossing (C_TS; chi2/dof) | Wilson routes vs Kirchhoff (Omega_J) | edges per route (loop-erased / Wilson) |')
    L.append('|---|---|---:|---:|---:|---|---:|---|')
    for name, r5 in p5['systems'].items():
        fl, po, w = r5['flat'], r5['lerw_pooled'], r5['wilson_check']
        L.append(f"| {SYS[name]} | {fl['via_tilt']['C_TS']:.6f} / {fl['via_tilt']['Omega_J']:.6f} | {fl['kirchhoff_vs_flat_law_max_abs']:.1e} | "
                 f"{po['Omega_J_vs_kirchhoff']:.4f} | {po['Omega_J_vs_flat']:.4f} | {w['C_TS_wilson_vs_lerw']:.4f}; {w['chi2']:.1f}/{w['dof']} | "
                 f"{w['Omega_J_wilson_vs_kirchhoff']:.4f} | {w['mean_route_edges_lerw']:.1f} / {w['mean_route_edges_wilson']:.1f} |")
    (HERE/'tables.md').write_text('\n'.join(L)+'\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
