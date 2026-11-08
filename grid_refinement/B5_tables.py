"""Print the README tables (markdown) from the result JSONs, and write the
combined results file B_results.json."""
import json
import numpy as np

import B_lib as BL

OUT = BL.OUT
B0 = json.loads((OUT/'B0_checks.json').read_text())
B1b = json.loads((OUT/'B1_bottleneck_interpolated.json').read_text())
B1n = json.loads((OUT/'B1_nine_well_interpolated.json').read_text())
B3 = json.loads((OUT/'B3_reoptimized.json').read_text())
X = '×'


def g(lv):
    return lv['grid'].replace('x', X)


def f(v, d=3):
    if v is None:
        return '-'
    if abs(v) >= 1000:
        return f'{v:,.0f}'
    return f'{v:.{d}g}' if abs(v) >= 100 else f'{v:.{d}g}'


def sp(v):
    return f'{v:.1f}' if v < 1000 else f'{v:,.0f}'


lines = []
P = lines.append

# ---------------------------------------------------------------- table 1
P('### Table 1. Bottleneck, U = 3, coarse (20×10) optima interpolated onto each grid\n')
P('| Grid | n | cells in A (= B) | MFPT₀·h_x² | γ₀/h_x² | MFPT opt.: S_MFPT | S_γ | Ω_J | C_TS '
  '| Spectral opt.: S_MFPT | S_γ | Ω_J | C_TS |')
P('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for lv in B1b['levels']:
    m, s = lv['protocols']['MFPT_opt_U3'], lv['protocols']['spectral_opt_U3']
    P(f'| {g(lv)} | {lv["n"]} | {lv["n_A"]} | {lv["mfpt0_times_hx2"]:.1f} | {lv["gap0_over_hx2"]:.4f} | '
      f'{m["mfpt_speedup"]:.1f} | {m["gap_speedup"]:.1f} | {m["Omega_J"]:.3f} | {m["C_TS"]:.3f} | '
      f'{s["mfpt_speedup"]:.1f} | {s["gap_speedup"]:.1f} | {s["Omega_J"]:.3f} | {s["C_TS"]:.3f} |')
P('')

# ---------------------------------------------------------------- table 2
P('### Table 2. Nine-well, coarse (20×20) optima interpolated onto each grid\n')
P('| U | Grid | cells in A / B | MFPT opt.: S_MFPT | S_γ | Ω_J | C_TS | Spectral opt.: S_MFPT | S_γ | Ω_J | C_TS |')
P('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for U in [3, 6]:
    for lv in B1n['levels']:
        m, s = lv['protocols'][f'MFPT_opt_U{U}'], lv['protocols'][f'spectral_opt_U{U}']
        P(f'| {U} | {g(lv)} | {lv["n_A"]} / {lv["n_B"]} | {sp(m["mfpt_speedup"])} | {m["gap_speedup"]:.2f} | '
          f'{m["Omega_J"]:.3f} | {m["C_TS"]:.3f} | {s["mfpt_speedup"]:.2f} | {s["gap_speedup"]:.1f} | '
          f'{s["Omega_J"]:.3f} | {s["C_TS"]:.3f} |')
P('')


# ---------------------------------------------------------------- table 3/4
def reopt_rows(entry, U, label):
    rows = []
    for obj, key, ikey, job in [('MFPT', f'MFPT_opt_U{U}', f'MFPT_coarse_interp_U{U}', 'job_mfpt'),
                                ('spectral', f'spectral_opt_U{U}', f'spectral_coarse_interp_U{U}', 'job_gap')]:
        r, i, j = entry['protocols'][key], entry['protocols'][ikey], entry[job]
        own = 'mfpt_speedup' if obj == 'MFPT' else 'gap_speedup'
        other = 'gap_speedup' if obj == 'MFPT' else 'mfpt_speedup'
        st = j['stationarity']
        stat = st['proj_grad_rel'] if obj == 'MFPT' else st['clarke_proj_grad_rel']
        rows.append(f'| {label} | {obj} | {sp(i[own])} | **{sp(r[own])}** | {sp(r[other]) if r[other] >= 1000 else f"{r[other]:.2f}"} | '
                    f'{r["Omega_J"]:.3f} | {r["C_TS"]:.3f} | {r["lam3_over_lam2"]:.3f} | '
                    f'{j["value_spread_rel"]:.1e} | {stat:.1e} | {len(j["runs"])} |')
    return rows


hdr = ('| {0} | Optimum | own objective, coarse optimum interpolated | own objective, re-optimized | '
       'other objective (cross value) | Ω_J | C_TS | λ₃/λ₂ | spread over starts (rel.) | stationarity | starts |')
P('### Table 3. Bottleneck, U = 3: per-state optima re-optimized on each grid\n')
P(hdr.format('Grid'))
P('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for grid, e in B3['bottleneck'].items():
    lines.extend(reopt_rows(e, 3, grid.replace('x', X)))
P('')
P('### Table 4. Nine-well: per-state optima re-optimized on each grid\n')
P(hdr.format('U, grid'))
P('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
p7 = OUT/'B7_ninewell_80.json'          # 80x80 re-optimization
B7 = json.loads(p7.read_text()) if p7.exists() else {}
nine = dict(B3['nine_well']); nine.update(B7)
for k in sorted(nine, key=lambda s: (int(s.split('_')[0][1:]), int(s.split('_')[1]))):
    e = nine[k]
    U = int(k.split('_')[0][1:]); grid = k.split('_')[1]
    lines.extend(reopt_rows(e, U, f'{U}, {grid}{X}{grid}'))
P('')

# ---------------------------------------------------------------- table 5: series 2
if B3.get('bottleneck_reopt_interpolated'):
    P('### Table 5. Bottleneck: re-optimized optima interpolated onto finer grids\n')
    P('| From → to | MFPT opt.: S_MFPT | S_γ | Ω_J | C_TS | Spectral opt.: S_MFPT | S_γ | Ω_J | C_TS |')
    P('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for k, e in B3['bottleneck_reopt_interpolated'].items():
        pr = e['protocols']
        m = [v for kk, v in pr.items() if kk.startswith('MFPT')][0]
        s = [v for kk, v in pr.items() if kk.startswith('spectral')][0]
        P(f'| {k.replace("x", X).replace("->", " → ")} | {m["mfpt_speedup"]:.1f} | {m["gap_speedup"]:.1f} | {m["Omega_J"]:.3f} | '
          f'{m["C_TS"]:.3f} | {s["mfpt_speedup"]:.1f} | {s["gap_speedup"]:.1f} | {s["Omega_J"]:.3f} | {s["C_TS"]:.3f} |')
    P('')

# ---------------------------------------------------------------- table 6: profiles
B3c = json.loads((OUT/'B3c_profile_weighted.json').read_text())
P('### Table 6. Bias profiles: re-optimized optimum minus a coarser optimum interpolated onto the same grid\n')
P('Weighted by where the objective is sensitive (MFPT optimum: occupation time on the way from A to B; '
  'spectral optimum: slow-mode density φ₂²), computed on the finer grid.\n')
P('| System | Optimum | Comparison | weighted RMS / U | weight in cells where abs(Δb) > U/2 | unweighted RMS / U |')
P('|---|---|---|---:|---:|---:|')
names = {'saved20x10': f'saved 20{X}10', '40x20': f're-optimized 40{X}20', 'saved20x20': f'saved 20{X}20',
         'reopt20x20': f're-optimized 20{X}20'}
for k, v in B3c.items():
    sysn, obj, comp = k.split('/')
    a, b = comp.split('_vs_')
    a = a.replace('_reopt', ' re-optimized').replace('x', X)
    sysl = 'bottleneck, U = 3' if sysn == 'bottleneck' else f'nine-well, U = {sysn[-1]}'
    P(f'| {sysl} | {"spectral" if obj == "gap" else "MFPT"} | {a} vs {names.get(b, b)} | '
      f'{v["weighted_rms_over_U"]:.4f} | {v["weight_in_cells_diff_gt_half_U"]:.1e} | {v["unweighted_rms_over_U"]:.3f} |')
for k, e in B7.items():
    U = k.split('_')[0][1:]
    for obj, nm in [('gap', 'spectral'), ('mfpt', 'MFPT')]:
        v = e['profile_weighted'][f'{obj}_80_vs_reopt40']
        P(f'| nine-well, U = {U} | {nm} | 80{X}80 vs re-optimized 40{X}40 | '
          f'{v["weighted_rms_over_U"]:.4f} | {v["weight_in_cells_diff_gt_half_U"]:.1e} | {v["unweighted_rms_over_U"]:.3f} |')
P('')

# ---------------------------------------------------------------- checks summary
worst = dict(mfpt=0.0, gap=0.0, C_TS=0.0, Omega=0.0, lc_total=0.0, lc_law=0.0, flux=0.0)


def scan(prot):
    for r in prot.values():
        worst['mfpt'] = max(worst['mfpt'], *[abs(v) for v in r['check_mfpt_rel'].values()])
        worst['gap'] = max(worst['gap'], *[abs(v) for v in r['check_gap_rel'].values()])
        worst['C_TS'] = max(worst['C_TS'], abs(r['check_C_TS_route2'] - r['C_TS']))
        worst['Omega'] = max(worst['Omega'], abs(r['check_Omega_J_BHS'] - r['Omega_J']))
        worst['lc_total'] = max(worst['lc_total'], abs(r['check_lastcrossing_total'] - 1))
        worst['lc_law'] = max(worst['lc_law'], r['check_lastcrossing_maxdiff'])
        worst['flux'] = max(worst['flux'], r['flux_residual'])


for lv in B1b['levels'] + B1n['levels']:
    scan(lv['protocols'])
for e in list(B3['bottleneck'].values()) + list(B3['nine_well'].values()) + \
        list(B3.get('bottleneck_reopt_interpolated', {}).values()) + list(B7.values()):
    scan(e['protocols'])
P('### Second-route checks (largest deviation over every evaluated bias)\n')
P(f'- MFPT, occupation solve and eigendecomposition vs LU solve: {worst["mfpt"]:.1e} (relative)')
P(f'- γ, Arnoldi on the non-symmetrised generator and dense eig vs Lanczos: {worst["gap"]:.1e} (relative)')
P(f'- C_TS, forward-occupation route vs exact_last_crossing.law: {worst["C_TS"]:.1e}; '
  f'largest per-edge probability difference {worst["lc_law"]:.1e}; total last-crossing mass minus 1: {worst["lc_total"]:.1e}')
P(f'- Ω_J, reversible BHS form vs general form: {worst["Omega"]:.1e}')
P(f'- last-crossing flux vs reactive capacity (relative): {worst["flux"]:.1e}')
print('\n'.join(lines))
(OUT/'B5_tables.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

# combined results
BL.dump(dict(step0_checks=B0, step1_bottleneck_interpolated=B1b, step1_nine_well_interpolated=B1n,
             step2_reoptimized=B3, step2_profile_weighted=B3c, step2_ninewell_80x80_markov=B7,
             step2_mfpt_cross_value_spread=json.loads((OUT/'B3b_mfpt_cross_spread.json').read_text()),
             second_route_worst_deviation=worst), OUT/'B_results.json')
print('saved B_results.json')

# fill the tables into README.md between <!-- BEGIN Tn --> and <!-- END Tn --> markers
import re
md = '\n'.join(lines)
blocks = {}
for blk in re.split(r'(?=^### )', md, flags=re.M):
    m = re.match(r'### Table (\d+)\.', blk)
    if m:
        blocks[f'T{m.group(1)}'] = blk.strip()
    elif blk.startswith('### Second-route'):
        blocks['CHECKS'] = blk.strip()
readme = OUT/'README.md'
if readme.exists():
    txt = readme.read_text(encoding='utf-8')
    for k, v in blocks.items():
        txt = re.sub(rf'<!-- BEGIN {k} -->.*?<!-- END {k} -->',
                     lambda _m: f'<!-- BEGIN {k} -->\n{v}\n<!-- END {k} -->', txt, flags=re.S)
    readme.write_text(txt, encoding='utf-8')
    print('README tables filled:', sorted(blocks))
