"""Figure 7 -- umbrella coverage-gain per CV (faded variant).

Reads filtered_results.json (per-objective validity, from filtered_best.py).
Solid bar         = best protocol passing that objective's coverage screen.
Faded + hatched   = best protocol found (W>=3) that FAILS the screen, annotated
                    with its coverage-failure fraction Phi.
W=1 "mirage" protocols are not plotted (they go in the companion table).

Two panels: (a) 2D grid, (b) pentapeptide negative control.  One shared legend,
upper-right of panel (b).  Output overwrites the manuscript figure fig2_notitle
(legacy name) as .pdf + .png in rosta_jctc_v5/figures/.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # bundle root; override via MSM_ROOT
F = json.load(open(f'{ROOT}/filtered_results.json'))

ORDER = {'grid': ['Committor', 'EigVec2', 'MFPTcv', 'x'],
         'pentapeptide': ['Basic', 'Committor', 'EigVec2', 'MFPTcv']}
LABEL = {'grid': '(a) 2D grid ($n=800$, deep barrier)',
         'pentapeptide': '(b) Pentapeptide MSM ($n=250$)'}
col = {'barrier': (0.12, 0.47, 0.71), 'rate': (0.84, 0.15, 0.16)}
W = 0.38


def fmt3(v):
    if abs(v) >= 1e4:
        m, e = f'{v:.2e}'.split('e')
        return rf'${float(m):.2f}{{\times}}10^{{{int(e)}}}$'
    return f'{v:.3g}'


fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
for ax, net in zip(axes, ['grid', 'pentapeptide']):
    cvs = ORDER[net]; xs = np.arange(len(cvs))
    for i, cv in enumerate(cvs):
        for obj, dx in (('barrier', -W / 2), ('rate', +W / 2)):
            o = F[net]['per_cv'][cv][obj]
            c = col[obj]
            if o['valid'] is not None:                       # solid: passes screen
                v = o['valid']['score']
                ax.bar(i + dx, v, W, facecolor=c, edgecolor='k', lw=1.1, zorder=3)
                ax.annotate(fmt3(v), (i + dx, v), xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8.5, color=c)
            elif o['best_found'] is not None:                # faded+hatched: fails screen
                v = o['best_found']['score']; phi = o['best_found']['coverage_failure']
                ax.bar(i + dx, v, W, facecolor=(*c, 0.26), edgecolor=c, lw=1.2,
                       hatch='///', zorder=3)
                ax.annotate(f'{fmt3(v)}\n$\\Phi$={phi:.3f}', (i + dx, v),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8, color='0.35')
    ax.set_yscale('log'); ax.set_xticks(xs); ax.set_xticklabels(cvs, rotation=12)
    ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.6, zorder=1)
    ax.set_ylabel('valid coverage gain (log scale)')
    ax.set_title(LABEL[net], fontsize=12)
    ax.margins(y=0.18)

axes[1].legend(handles=[
    Patch(facecolor=col['barrier'], edgecolor='k', label='barrier objective (passes screen)'),
    Patch(facecolor=col['rate'], edgecolor='k', label='rate objective (passes screen)'),
    Patch(facecolor='0.85', edgecolor='0.4', hatch='///', label='best found, fails screen')],
    fontsize=8.5, loc='upper right', framealpha=0.95)

for ext in ('pdf', 'png'):
    fig.savefig(f'{ROOT}/rosta_jctc_v5/figures/fig2_notitle.{ext}',
                dpi=200, bbox_inches='tight')
print('wrote rosta_jctc_v5/figures/fig2_notitle.pdf/.png')
