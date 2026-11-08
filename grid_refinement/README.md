# grid_refinement

Entropic-bottleneck (and nine-well) grids: fixed-bias refinement versus reoptimization on each grid.

Run order: `B0_checks.py` → `B1_refine_interpolated.py` → `B2_reoptimize.py` (+ `B2_merge.py`) → `B3_evaluate.py`, `B3b_mfpt_cross_spread.py`, `B3c_profile_weighted.py` → `B7_eval_ninewell_80.py` → `B5_tables.py`.

- `bottleneck_tableS1_and_masks.py`: Table S1 (reweighting cost) from the saved 40 × 20 optima, and the A/B masks
- `B2_*_{gap,mfpt}.npz`: reoptimized optima; `B5_tables.md`: Table S2 values
