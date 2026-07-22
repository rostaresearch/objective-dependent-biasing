"""Assemble Figure 13 (fig_ras_pathway): the two-panel Ras phosphate-release figure.

Panel (a) is the molecular schematic of the phosphate leaving the Mg(2+)/GDP
pocket, coloured by the P-Mg distance. It is an author-rendered PyMOL image that
cannot be regenerated from the deposited processed inputs (it needs the raw
~224 GB trajectories and a molecular-graphics session), so it is deposited as a
static asset, ``figures/fig_ras_pathway_panelA.png``, and simply placed here.

Panel (b) -- the mean-field edge-rate distortion D_edge across three grid
resolutions and three auxiliary features, with 90% block-bootstrap intervals --
IS reproduced from the deposited data (``data/ras_pi_audit.json``,
``grid_feature_sweep``). Running this script rebuilds panel (b) from those
numbers and composites it with panel (a) to write ``figures/fig_ras_pathway.png``
(and ``.pdf``), the exact manuscript Figure 13.

Usage:
    export MSM_ROOT=/path/to/objective-dependent-biasing   # or auto-detected
    python code/assemble_figure13.py
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # <root>/code/ -> <root>
DATA = os.path.join(PATH, 'data')
FIGURES = os.path.join(PATH, 'figures')

CRIMSON = (0.84, 0.15, 0.16)


def prettify(label):
    """Turn a code-style 'grid / feature' label into a readable TeX tick label."""
    for a, b in (('10x7', r'10$\times$7'), ('12x8', r'12$\times$8'),
                 ('14x9', r'14$\times$9')):
        label = label.replace(a, b)
    label = label.replace('Pi_Q61', r'P$_\mathrm{i}$-Gln61')      # phosphate-Gln61
    label = label.replace('Pi_wat_4.5', r'P$_\mathrm{i}$-water')  # phosphate solvation
    label = label.replace('Mg_O_coord', 'Mg-O coord')            # Mg(2+)-O coordination
    return label


def main():
    # ---- panel (b) data: D_edge grid x feature sweep ------------------------
    J = json.load(open(os.path.join(DATA, 'ras_pi_audit.json')))
    gf = J['grid_feature_sweep']
    labels = [prettify(g['label']) for g in gf]
    med = np.array([g['D_edge']['median'] for g in gf])
    lo = np.array([g['D_edge']['p5'] for g in gf])
    hi = np.array([g['D_edge']['p95'] for g in gf])
    x = np.arange(len(gf))

    # ---- panel (a) image: author-rendered molecular schematic ---------------
    panelA_path = os.path.join(FIGURES, 'fig_ras_pathway_panelA.png')
    if not os.path.exists(panelA_path):
        raise FileNotFoundError(
            f"missing molecular panel {panelA_path}; it is a deposited static "
            "asset (see module docstring) and cannot be regenerated from data.")
    imgA = mpimg.imread(panelA_path)

    # ---- composite ----------------------------------------------------------
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 20})
    fig = plt.figure(figsize=(20.0, 8.72))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.22],
                          left=0.005, right=0.985, top=0.90, bottom=0.20,
                          wspace=0.06)

    axA = fig.add_subplot(gs[0, 0])
    axA.imshow(imgA)
    axA.axis('off')
    axA.set_title('(a) Phosphate-release pathway', fontsize=27,
                  fontweight='bold', loc='left', pad=10)

    axB = fig.add_subplot(gs[0, 1])
    axB.bar(x, med, width=0.68, color=CRIMSON, edgecolor='k', linewidth=1.1)
    axB.errorbar(x, med, yerr=[med - lo, hi - med], fmt='none', ecolor='k',
                 elinewidth=2.0, capsize=7, capthick=2.0)
    axB.set_ylim(0, 0.09)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, rotation=32, ha='right', fontsize=18)
    axB.set_ylabel(r'$D_\mathrm{edge}$', fontsize=26)
    axB.set_title(r'(b) Edge-rate distortion across grid $\times$ feature',
                  fontsize=27, fontweight='bold', pad=10)
    axB.tick_params(axis='y', labelsize=20)
    for s in ('top', 'right'):
        axB.spines[s].set_visible(False)

    out_png = os.path.join(FIGURES, 'fig_ras_pathway.png')
    out_pdf = os.path.join(FIGURES, 'fig_ras_pathway.pdf')
    fig.savefig(out_png, dpi=200, facecolor='white')
    fig.savefig(out_pdf, facecolor='white')
    plt.close(fig)
    print(f'D_edge band across 9 grid x feature combinations: '
          f'[{med.min():.4f}, {med.max():.4f}]')
    print(f'wrote {out_png}  ({imgA.shape[1]}x{imgA.shape[0]} panel (a) + '
          f'{len(gf)}-bar panel (b))')
    print(f'wrote {out_pdf}')


if __name__ == '__main__':
    main()
