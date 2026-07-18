# Objective-dependent optimal biasing of Markov state networks

Code and processed data to reproduce every figure and table in

> C. Wang, A. Fadhluddin, E. Rosta,
> *Objective-dependent optimal biasing of Markov state networks*,
> Journal of Chemical Theory and Computation (submitted).
> Corresponding author: Edina Rosta (`e.rosta@ucl.ac.uk`), University College London.

The analysis is closed-form linear algebra on transition-rate matrices — no
molecular dynamics or Monte-Carlo is run here. The only external simulation input
is the set of pre-computed Ras phosphate-unbinding propagations (see
[Data availability](#data-availability)); everything else is generated from the
deposited processed inputs.

---

## What the paper is about

Given a reversible Markov state model with generator `K⁰`, we ask what it means to
add a single static bias `b` (with `Kᵇ` the locally-detailed-balance-tilted
generator) that is *optimal*. The central point is that "optimal" is
objective-dependent: three natural objectives have three genuinely different
optima under the same budget.

1. **Umbrella-sampling coverage** — which multi-window protocol best covers a
   reaction coordinate for free-energy/rate estimation. Formulated as a coverage
   problem with a Poisson local-coverage approximation of the estimator variance,
   plus a validity screen (window overlap and coverage-failure thresholds).
2. **Spectral preconditioning** — which single bias maximizes the spectral gap
   `γ = −maxᵢ Re λᵢ(Kᵇ)` over the non-stationary spectrum (fastest global
   relaxation). Solved with a closed-form Hellmann–Feynman eigenvalue gradient.
3. **Mean-first-passage-time (MFPT) optimization** — which single bias minimizes
   `MFPT(A→B)` for a specific transition. Solved with a closed-form adjoint
   gradient.

The headline result is the **cross-objective collapse**: at large bias budgets the
spectral-gap-optimal bias actively *slows* the A→B transition it should help
(`S_MFPT < 1`, i.e. slower than unbiased), because maximizing global mixing
deepens the inter-basin barriers that gate the specific transition. Because the
budget is an upper bound (`b = 0` is always feasible), the MFPT-optimal speedup is
monotone and never drops below one — so the slowdown is a property of applying a
*spectral*-optimal bias to a directed transition, not of the budget.

A second contribution is a pair of **mechanism-conservation diagnostics** for a
bias that has actually been applied to real data:

- **`Ω_J`** — reactive-current overlap between the biased and unbiased
  transition-path-theory currents (1 = mechanism preserved).
- **`D_edge`** — edge-rate distortion, the relative change in per-edge rates.

These are applied to real Ras phosphate-unbinding trajectories to show that a
production bias can be strong enough to accelerate sampling while still conserving
the release mechanism, and to define a decision curve (`Ω_J` vs model MFPT
speedup) for choosing a production bias amplitude.

---

## Repository layout

```
objective-dependent-biasing/
├── README.md
├── code/            analysis + plotting scripts (Python 3 and MATLAB)
├── data/            processed inputs and numerical outputs (see manifest)
└── figures/         reference copies of the manuscript figures (regenerable)
```

All scripts resolve paths through the `MSM_ROOT` environment variable and contain
no machine-specific locations. Set it once to the repository root:

```bash
export MSM_ROOT=/path/to/objective-dependent-biasing      # bash
$env:MSM_ROOT = "C:\path\to\objective-dependent-biasing"  # PowerShell
```

If `MSM_ROOT` is unset, the scripts fall back to auto-detecting the repository
root from their own location.

---

## Requirements

- **Python ≥ 3.9** with `numpy`, `scipy`, `matplotlib` (figure generation),
  and `MDAnalysis` (only for re-projecting raw Ras trajectories, not needed for
  the deposited processed inputs).
- **MATLAB ≥ R2021b** for the `plot_*.m` figure scripts (they use `tiledlayout`
  and `exportgraphics`). The `.fig` sources are also provided for hand-editing.

No compilation is required.

---

## How the pipeline is organized

The computation is two-tier: **Python compute scripts** read the processed inputs
and write numerical results (`data/*.json`, `data/*.mat`); **MATLAB/Python plot
scripts** read those and render the figures. To reproduce a figure you can either
rerun its compute script (regenerates the numbers) or go straight to the plot
script (uses the deposited numbers).

### Core library

| File | Contents |
|---|---|
| `code/analytic_lib.py` | Reference implementations: generator tilting `Kᵇ`, stationary distribution, committor, MFPT, spectral gap, umbrella coverage/variance scores, oracle flattening bias. |
| `code/mechanism_audit_ras_proper.py` | Mechanism-conservation diagnostics (`Ω_J`, `D_edge`), 2D count-matrix builder, bootstrap driver. Sets the thermodynamic temperature (`T = 300 K`, matching the KOMBI runs). Imported by all Ras analysis scripts. |

### Objective 1 — umbrella-sampling coverage and validity screen

| File | Role |
|---|---|
| `code/network_us.py` | Umbrella family on a network; coverage and Poisson-variance scores. |
| `code/run_barrier_study.py` | Protocol sweep over windows `W`, force constant `κ`, and CV choice. |
| `code/filtered_best.py` | Applies the validity screen (overlap ≥ `O_min`, coverage-failure ≤ `Φ_max`). |

### Objective 2 — spectral preconditioning

| File | Role |
|---|---|
| `code/optimal_spectral_bias.py` | Polynomial spectral optimum with edge-smoothness regularization. |
| `code/per_state_ceiling.py` | Per-state spectral ceiling via the Hellmann–Feynman gradient (verified against finite differences). |
| `code/optimal_equilibration_bias.py` | Flattening/equilibration bias reference. |

### Objective 3 — MFPT optimization

| File | Role |
|---|---|
| `code/optimal_mfpt_bias.py` | Polynomial MFPT optimum. |
| `code/mfpt_per_state.py` | Per-state MFPT ceiling via the adjoint gradient (verified against finite differences). |

### Cross-objective sweeps (the collapse)

| File | Role |
|---|---|
| `code/regime_and_budget_sweep.py` | Budget sweep (`U_max`) and barrier-height regime sweep; powers the collapse figures. |
| `code/optionC_sweeps_specbias_mfpt.py`, `code/optionC_consolidate_headline.py` | Smooth degree-4 polynomial bias (Option C) sweeps and consolidated headline values. |

### Mechanism-conservation analysis on Ras

| File | Role |
|---|---|
| `code/ras_pi_audit.py` | Canonical Ras applied-bias analysis: builds the 2D MSM on the phosphate-release CV, computes `Ω_J` and `D_edge`. |
| `code/ras_pi_audit_boot.py`, `code/ras_pi_block_boot.py` | Bootstrap confidence intervals (block bootstrap over propagations for correct dependence). |
| `code/ras_pi_jensen_Rij.py` | Jensen-convexity correction term `R_ij` for the reweighting. |
| `code/ras_pi_resolved_dham.py`, `code/ras_pi_dedge_resolved_boot.py`, `code/ras_pi_dedge95.py` | Propagation-resolved (vs mean-field) bias treatment and its edge-distortion. |
| `code/sensitivity_alpha_sweep.py` | Pseudocount-`α` regularization sensitivity. |
| `code/sensitivity_grid_feature_sweep.py` | Grid-resolution × orthogonal-feature robustness sweep. |
| `code/export_ras_pi_mat.py` | Bias-scaling (Pareto) sweep; exports `data/ras_pi_fig_data.mat` for the MATLAB figures. |

### Figure scripts

Python (matplotlib): `code/make_figures.py`, `code/make_figures_v2.py`,
`code/make_final_figure.py`, `code/regen_fig_spectral_bias.py`.
MATLAB: `code/plot_grid_2d.m`, `code/plot_rkhs_2d.m`, `code/plot_asymmetric_1d.m`,
`code/plot_coordfree_bias.m`, `code/plot_mechanism_v2.m`, `code/plot_fig2_us.m`,
`code/plot_ras_pi_figs.m` (the four Ras figures). `code/generate_all_figs.m` runs
the MATLAB set. Editable `.fig` sources accompany each MATLAB figure.

---

## Protocol details

- **Bias budget.** The dimensionless bias `bᵢ = β Uᵢ` is constrained by a box
  `|bᵢ| ≤ U_max` (default `U_max = 3 k_BT`). Because `Kᵇ` depends only on the
  differences `b_j − b_i`, the box is stated in its shift-invariant form
  `maxᵢ bᵢ − minᵢ bᵢ ≤ 2 U_max`, with the gauge fixed by midrange centering.
- **Temperature.** `T = 300 K`, `k_BT = 8.31446261815·10⁻³ × T` kJ/mol, matching
  the KOMBI Ras propagations.
- **Spectral gap at the optimum.** `γ` is taken as `−maxᵢ Re λᵢ` over the
  non-stationary spectrum (a maximum over eigenvalues, not a labeled branch), so
  it stays continuous where the two slowest modes merge — which is exactly what a
  strong spectral optimum drives toward. The Hellmann–Feynman gradient is applied
  only away from such crossings.
- **Ras analysis.** Diagnostics use the canonical 12×8 grid on the
  phosphate-release CV, with the source set `A` and target set `B` connected by
  observed counts; confidence intervals are 200-resample block bootstraps over
  propagations. The nearest-neighbor pseudocount `α` never activates on this data
  (`A`, `B` are dynamically connected), so the reported diagnostics are properties
  of the data rather than the regularization.

---

## Data availability

The `data/` directory contains the processed transition-count inputs,
propagation-resolved bias metadata, state assignments, and numerical outputs
needed to regenerate every figure and table.

The **adaptive biasing algorithm** that generated the Ras propagations is KOMBI,
available separately at <https://github.com/rostaresearch/KOMBI>.

The **raw Ras phosphate-unbinding trajectories** (~224 GB of biased propagations)
are too large for public deposition and are available from the corresponding
author on reasonable request. The projected per-frame collective variable, applied
bias, and orthogonal protein features used here — sufficient to reproduce every
Ras number in the paper — are included in `data/` (`pi_features.npz`).

---

## Citation

If you use this code, please cite the paper above. The Ras trajectory-generation
method should additionally cite the KOMBI reference given in the manuscript.
