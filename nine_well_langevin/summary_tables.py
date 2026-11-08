"""Markdown tables for the README, generated from results.json and the check files (no hand copying)."""
import json

from common import HERE

LBL = {'unbiased|none': 'unbiased', 'MFPT_U3|bilinear': 'MFPT bias, U = 3', 'spectral_U3|bilinear': 'spectral bias, U = 3',
       'MFPT_U6|bilinear': 'MFPT bias, U = 6', 'spectral_U6|bilinear': 'spectral bias, U = 6',
       'MFPT_U3|bicubic': 'MFPT bias, U = 3 (bicubic)', 'spectral_U3|bicubic': 'spectral bias, U = 3 (bicubic)',
       'MFPT_U6|bicubic': 'MFPT bias, U = 6 (bicubic)', 'spectral_U6|bicubic': 'spectral bias, U = 6 (bicubic)'}
ORDER = list(LBL)


def g(x, d=4):
    if abs(x) >= 1e5:
        return f'{x:.3e}'.replace('e+0', 'e').replace('e+', 'e')
    if abs(x) >= 1e3:
        return f'{x:.0f}'
    return f'{x:.{d}g}'


def ci(c, d=4):
    return f'[{g(c[0], d)}, {g(c[1], d)}]'


def main():
    r = json.loads((HERE / 'results.json').read_text())
    out = []
    out.append('### Network predictions (A -> B MFPT, physical time, D = 1)\n')
    out.append('| Case | vertex grid 120 x 120 (audit grid) | aligned grids n = 114 / 228 / 456 | h -> 0, SQRA | h -> 0, Scharfetter-Gummel | h -> 0, central differences |')
    out.append('|---|---:|---:|---:|---:|---:|')
    for k in ORDER:
        n = r['network_reference'].get(k)
        if not n:
            continue
        fd = n.get('central_fd')
        out.append(f"| {LBL[k]} | {g(n['vertex_120'])} | {' / '.join(g(v) for v in n['sqra']['values'])} | "
                   f"{g(n['continuum'])} | {g(n['continuum_sg'])} | {g(fd['extrapolated']) if fd else '-'} |")
    out.append('\n### Langevin against network\n')
    out.append('| Case | Langevin MFPT (95% CI) | per time step: dt: walkers or transitions (by host), MFPT | network, h -> 0 | Langevin / network (95% CI) |')
    out.append('|---|---:|---|---:|---:|')
    for k in ORDER:
        c = r['comparison'].get(k)
        if not c:
            continue
        def hosts(v):
            h = v.get('n_by_host', {})
            return ' (' + ', '.join(f"{k} {h[k]}" for k in sorted(h)) + ')' if len(h) > 1 else f" ({next(iter(h))})" if h else ''
        per = '; '.join(f"{d}: {v['n']}{'+' + str(v['censored']) + ' running' if v['censored'] else ''}{hosts(v)}, {g(v['mfpt'])}"
                        for d, v in sorted(c['per_dt'].items(), key=lambda kv: -float(kv[0])))
        extra = (f"; widened x{c['birge_ratio']:.2f} for run scatter: {ci(c['measured_over_network_ci95_scaled'], 3)}"
                 if 'birge_ratio' in c else '')
        out.append(f"| {LBL[k]} | {g(c['measured'])} {ci(c['measured_ci95'])} | {per} | {g(c['network_continuum'])} | "
                   f"{c['measured_over_network']:.3f} {ci(c['measured_over_network_ci95'], 3)}{extra} |")
    out.append('\n### Speedups (unbiased MFPT / biased MFPT)\n')
    out.append('| Bias | network | Langevin / Langevin (95% CI) | network unbiased / Langevin biased (95% CI) |')
    out.append('|---|---:|---:|---:|')
    for k in ORDER:
        s = r['speedups'].get(k)
        if not s:
            continue
        meas = f"{g(s['measured'], 3)} {ci(s['measured_ci95'], 3)}" if 'measured' in s else '-'
        out.append(f"| {LBL[k]} | {g(s['network'], 4)} | {meas} | {g(s['with_network_T0'], 4)} {ci(s['with_network_T0_ci95'], 4)} |")
    out.append('\n### Ranking: T(spectral bias) / T(MFPT bias) = speedup(MFPT bias) / speedup(spectral bias)\n')
    out.append('| U, interpolation | network | Langevin (95% CI) |')
    out.append('|---|---:|---:|')
    for k, v in r['ranking'].items():
        out.append(f"| {k.replace('|', ', ')} | {g(v['network'], 4)} | {g(v['measured'], 4)} {ci(v['measured_ci95'], 4)} |")
    if r['dt_extrapolation']:
        out.append('\n### Time-step trend (diagnostic): weighted line through the per-dt means\n')
        out.append('| Case | dt values | relative change from dt -> 0 to the largest dt (95% CI) |')
        out.append('|---|---|---:|')
        for k, v in r['dt_extrapolation'].items():
            out.append(f"| {LBL[k]} | {', '.join(f'{d:g}' for d in v['dts'])} | "
                       f"{100 * v['relative_dt_error_at_largest_dt']:+.1f}% "
                       f"[{100 * v['relative_dt_error_at_largest_dt_ci95'][0]:+.1f}%, {100 * v['relative_dt_error_at_largest_dt_ci95'][1]:+.1f}%] |")
    text = '\n'.join(out) + '\n'
    (HERE / 'results_tables.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
