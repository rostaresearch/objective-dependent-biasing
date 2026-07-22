"""
Final figures for the MSM Roundtable contribution.

Three figures:
  fig_landscape.png    Visualises the 2D grid, basins, committor, bias windows
  fig_heatmap_WK.png   (W, kappa) heatmap of barrier & rate speedups per CV
  fig_best_per_cv.png  Bar chart: best barrier & rate speedup per CV
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analytic_lib as L
from run_barrier_study import (make_grid_network, make_pentapeptide_network,
                                _mfpt_field)

import os
# Bundle root: override with the MSM_ROOT environment variable.
PATH = os.environ.get('MSM_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(PATH, 'data')
FIGURES = os.path.join(PATH, 'figures')
def fig_landscape():
    net = make_grid_network()
    K0, pi0, A, B = net['K0'], net['pi0'], net['A'], net['B']
    coords, F = net['coords'], net['F']
    q = L.committor_K(K0, A, B)
    nx, ny = 40, 20

    F_ref = -np.log(np.clip(pi0, 1e-300, None))
    F_ref = F_ref.reshape(nx, ny)
    q_grid = q.reshape(nx, ny)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    x = coords[:, 0].reshape(nx, ny); y = coords[:, 1].reshape(nx, ny)

    # 1) Free energy
    c1 = axes[0].contourf(x, y, F_ref, levels=20, cmap='viridis')
    axes[0].set_title(r'(a) Free energy $F = -k_BT\log\pi$')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    plt.colorbar(c1, ax=axes[0], label=r'$F\,[k_BT]$')
    axes[0].plot(x.flatten()[A], y.flatten()[A], 'wo', ms=2, alpha=0.7)
    axes[0].plot(x.flatten()[B], y.flatten()[B], 'wx', ms=4, alpha=0.7)
    axes[0].text(-0.93, 0.93, 'A', color='white', fontsize=12, fontweight='bold')
    axes[0].text( 0.85, 0.93, 'B', color='white', fontsize=12, fontweight='bold')

    # 2) Committor
    c2 = axes[1].contourf(x, y, q_grid, levels=20, cmap='RdBu_r', vmin=0, vmax=1)
    axes[1].contour(x, y, q_grid, levels=[0.4, 0.5, 0.6], colors='k', linewidths=1)
    axes[1].set_title(r'(b) Committor $q(x,y)$ with band $0.4 < q < 0.6$')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    plt.colorbar(c2, ax=axes[1], label=r'$q$')

    # 3) EigVec2 with umbrella centres
    v2 = L.right_eigvec2_K(K0)
    v2 = (v2 - v2.min()) / (v2.max() - v2.min())
    v2_grid = v2.reshape(nx, ny)
    c3 = axes[2].contourf(x, y, v2_grid, levels=20, cmap='coolwarm')
    axes[2].set_title(r'(c) Slow eigenvector $v_2$ with $W=4$ umbrella centers')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
    plt.colorbar(c3, ax=axes[2], label=r'$v_2$')
    # show 4 windows: contours where EigVec2 ≈ centre value
    for c_val in np.linspace(0.05, 0.95, 4):
        axes[2].contour(x, y, v2_grid, levels=[c_val], colors='k',
                        linewidths=0.8, alpha=0.6, linestyles='--')

    for _out in (PATH, os.path.join(PATH, 'figures')):
        if os.path.isdir(_out):
            fig.savefig(f'{_out}/fig_landscape.png', dpi=160, bbox_inches='tight')
            fig.savefig(f'{_out}/fig_landscape.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_landscape.png/.pdf')


def fig_heatmap_WK():
    with open(f'{DATA}/barrier_study.json') as f:
        results = json.load(f)
    res_grid = results['grid']
    rows = [r for r in res_grid['rows'] if 'error' not in r]
    cvs = sorted({r['cv'] for r in rows})
    Ws = sorted({int(r['W']) for r in rows})
    Ks = sorted({float(r['kappa']) for r in rows})

    fig, axes = plt.subplots(2, len(cvs), figsize=(3.0*len(cvs), 6.0),
                             constrained_layout=True)
    for c, cv in enumerate(cvs):
        # build grids
        bar = np.full((len(Ks), len(Ws)), np.nan)
        rat = np.full((len(Ks), len(Ws)), np.nan)
        for r in rows:
            if r['cv'] != cv: continue
            ki = Ks.index(float(r['kappa']))
            wi = Ws.index(int(r['W']))
            bar[ki, wi] = r['barrier_speedup']
            rat[ki, wi] = r['rate_speedup']
        im0 = axes[0, c].imshow(np.log10(np.maximum(bar, 1e-3)), origin='lower',
                                aspect='auto', cmap='RdYlBu_r',
                                vmin=-0.5, vmax=3.5,
                                extent=[-0.5, len(Ws)-0.5, -0.5, len(Ks)-0.5])
        axes[0, c].set_xticks(range(len(Ws))); axes[0, c].set_xticklabels(Ws)
        axes[0, c].set_yticks(range(len(Ks))); axes[0, c].set_yticklabels([f'{k:.0f}' for k in Ks])
        axes[0, c].set_xlabel('W'); axes[0, c].set_ylabel(r'$\kappa$')
        axes[0, c].set_title(f'{cv}\nlog$_{{10}}$(barrier speedup)', fontsize=9)
        plt.colorbar(im0, ax=axes[0, c], fraction=0.06)

        im1 = axes[1, c].imshow(np.log10(np.maximum(rat, 1e-3)), origin='lower',
                                aspect='auto', cmap='RdYlBu_r',
                                vmin=-0.5, vmax=1.5,
                                extent=[-0.5, len(Ws)-0.5, -0.5, len(Ks)-0.5])
        axes[1, c].set_xticks(range(len(Ws))); axes[1, c].set_xticklabels(Ws)
        axes[1, c].set_yticks(range(len(Ks))); axes[1, c].set_yticklabels([f'{k:.0f}' for k in Ks])
        axes[1, c].set_xlabel('W'); axes[1, c].set_ylabel(r'$\kappa$')
        axes[1, c].set_title(f'log$_{{10}}$(rate speedup)', fontsize=9)
        plt.colorbar(im1, ax=axes[1, c], fraction=0.06)

    fig.suptitle('2D grid (n=800): umbrella sampling speedup vs (W, κ)  '
                 r'[$N_{tot}=10^5$ lagtimes]', fontsize=11)
    fig.savefig(f'{FIGURES}/fig_heatmap_WK.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_heatmap_WK.png')


def fig_best_per_cv():
    with open(f'{DATA}/barrier_study.json') as f:
        results = json.load(f)

    def best(rows, target):
        bests = {}
        for r in rows:
            if 'error' in r: continue
            cv = r['cv']
            if cv not in bests or r[target] > bests[cv][target]:
                bests[cv] = r
        return bests

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    nets = [('grid', '2D grid (n=800, deep barrier)'),
            ('pentapeptide', 'Pentapeptide MSM (n=250)')]
    width = 0.35

    for ax, (key, label) in zip(axes, nets):
        R = results[key]
        rows = R['rows']
        bestB = best(rows, 'barrier_speedup')
        bestR = best(rows, 'rate_speedup')
        cvs = sorted(bestB.keys())
        xs = np.arange(len(cvs))
        bar_speedups = [bestB[cv]['barrier_speedup'] for cv in cvs]
        rate_speedups = [bestR[cv]['rate_speedup'] for cv in cvs]
        ax.bar(xs - width/2, bar_speedups,  width, label='barrier (best W,κ)', color='#1f77b4', edgecolor='k')
        ax.bar(xs + width/2, rate_speedups, width, label='rate (best W,κ)',    color='#d62728', edgecolor='k')
        ax.set_xticks(xs); ax.set_xticklabels(cvs, rotation=12)
        ax.set_ylabel('speedup vs unbiased')
        ax.set_title(label, fontsize=10)
        ax.axhline(1.0, color='gray', lw=0.8, ls='--')
        ax.set_yscale('log')
        ax.grid(True, axis='y', ls=':', alpha=0.4)
        ax.legend(fontsize=8, loc='upper left')
        for i, cv in enumerate(cvs):
            r = bestB[cv]
            ax.annotate(f"W={r['W']},κ={r['kappa']:.0f}",
                        (xs[i] - width/2, bar_speedups[i]),
                        xytext=(0, 3), textcoords='offset points',
                        fontsize=7, ha='center')
            r = bestR[cv]
            ax.annotate(f"W={r['W']},κ={r['kappa']:.0f}",
                        (xs[i] + width/2, max(rate_speedups[i], 1.001)),
                        xytext=(0, 3), textcoords='offset points',
                        fontsize=7, ha='center', color='#d62728')
    fig.suptitle('Best analytic speedup per CV: barrier-probability vs rate-relevance-weighted',
                 fontsize=11)
    fig.savefig(f'{FIGURES}/fig_best_per_cv.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_best_per_cv.png')


if __name__ == '__main__':
    fig_landscape()
    fig_heatmap_WK()
    fig_best_per_cv()
