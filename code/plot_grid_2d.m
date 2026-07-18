% Paper-quality MATLAB plots for the 2D-grid analyses
% (per-state spectral ceiling, MFPT optimum, budget sweep, regime sweep).
% Replaces fig_per_state_ceiling.png, fig_mfpt_vs_spectral.png,
% fig_budget_sweep.png, fig_regime_sweep.png, fig_landscape.png.
%
% Typography: Arial 18 pt, no top/right spine, ticks outward, no grid.

clear; close all;
PATH = getenv('MSM_ROOT');
if isempty(PATH), PATH = fileparts(fileparts(mfilename('fullpath'))); end
load(fullfile(PATH, 'grid_2d_data.mat'));
hires_path = fullfile(PATH, 'grid_2d_hires.mat');
if exist(hires_path, 'file')
    H = load(hires_path);
    fprintf('Using high-resolution grid (nx=%d, ny=%d) for heatmaps\n', H.nx, H.ny);
else
    H = struct('xs', xs, 'ys', ys, 'F', F, 'pi0', pi0, ...
               'u_ps_spec', u_ps_spec, 'u_ps_mfpt', u_ps_mfpt, ...
               'speedup_ps_spec', speedup_ps_spec, ...
               'speedup_ps_mfpt', speedup_ps_mfpt, ...
               'U_max', U_max);
    fprintf('grid_2d_hires.mat not found — using low-res for heatmaps\n');
end

set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultAxesFontSize', 18);
set(groot, 'defaultAxesLineWidth', 1.8);
set(groot, 'defaultAxesTickDir', 'out');
set(groot, 'defaultAxesBox', 'off');
set(groot, 'defaultTextFontName', 'Arial');
set(groot, 'defaultTextFontSize', 18);
set(groot, 'defaultLineLineWidth', 2.2);
set(groot, 'defaultFigureColor', 'w');

clean_ax = @(ax) set(ax, 'TickDir', 'out', 'Box', 'off', ...
                    'XGrid', 'off', 'YGrid', 'off', ...
                    'LineWidth', 1.8, 'FontSize', 18);
exportPNG = @(fname) exportgraphics(gcf, fullfile(PATH, [fname '.png']), ...
                                    'Resolution', 600, 'BackgroundColor', 'white');
exportPDF = @(fname) try_pdf(PATH, fname);

% Common consistent colors
col_flat = [0.50 0.50 0.50];
col_poly = [0.55 0.35 0.75];
col_ps   = [0.10 0.45 0.75];
col_mfpt = [0.85 0.30 0.10];

% =====================================================================
% FIG 1: landscape (V and pi0 heatmaps)
% =====================================================================
fig = figure('Position', [100 100 1100 460]);
tl = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax = nexttile;
contourf(ax, H.ys, H.xs, H.F, 24, 'LineColor', 'none');
colormap(ax, parula(64));
c = colorbar(ax);
c.Label.String = 'V(x, y)  [ k_B T ]';
c.Label.FontSize = 18;
xlabel(ax, 'y', 'FontSize', 18);
ylabel(ax, 'x', 'FontSize', 18);
clean_ax(ax);
title(ax, '(a) Free-energy landscape', 'FontWeight', 'normal');

ax = nexttile;
contourf(ax, H.ys, H.xs, H.pi0, 24, 'LineColor', 'none');
colormap(ax, parula(64));
c = colorbar(ax);
c.Label.String = '\pi_0(x, y)';
c.Label.FontSize = 18;
xlabel(ax, 'y', 'FontSize', 18);
ylabel(ax, 'x', 'FontSize', 18);
clean_ax(ax);
title(ax, '(b) Equilibrium distribution', 'FontWeight', 'normal');

exportPNG('fig_grid2d_landscape');
exportPDF('fig_grid2d_landscape');
fprintf('Saved fig_grid2d_landscape\n');

% =====================================================================
% FIG 2: per-state bias profiles (polynomial vs per-state, both objectives)
% =====================================================================
fig = figure('Position', [100 100 1400 920]);
tl = tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

Umax = H.U_max;
Nlev = 24;
lev_sym = linspace(-Umax, Umax, Nlev);

ax = nexttile;
contourf(ax, ys, xs, u_poly_spec, lev_sym, 'LineColor', 'none');
clim(ax, [-Umax, Umax]);
colormap(ax, redblue(64)); c = colorbar(ax);
c.Label.String = 'u  [ k_B T ]'; c.Label.FontSize = 18;
xlabel(ax, 'y'); ylabel(ax, 'x');
title(ax, sprintf('(a) Polynomial-spec bias  %.1f\\times', speedup_poly_spec), ...
      'FontWeight', 'normal');
clean_ax(ax);

ax = nexttile;
contourf(ax, H.ys, H.xs, H.u_ps_spec, lev_sym, 'LineColor', 'none');
clim(ax, [-Umax, Umax]);
colormap(ax, redblue(64)); c = colorbar(ax);
c.Label.String = 'u  [ k_B T ]'; c.Label.FontSize = 18;
xlabel(ax, 'y'); ylabel(ax, 'x');
title(ax, sprintf('(b) Per-state spectral  %.1f\\times', H.speedup_ps_spec), ...
      'FontWeight', 'normal');
clean_ax(ax);

ax = nexttile;
contourf(ax, H.ys, H.xs, H.u_ps_mfpt, lev_sym, 'LineColor', 'none');
clim(ax, [-Umax, Umax]);
colormap(ax, redblue(64)); c = colorbar(ax);
c.Label.String = 'u  [ k_B T ]'; c.Label.FontSize = 18;
xlabel(ax, 'y'); ylabel(ax, 'x');
title(ax, sprintf('(c) Per-state MFPT  %.1f\\times', H.speedup_ps_mfpt), ...
      'FontWeight', 'normal');
clean_ax(ax);

ax = nexttile;
diff = H.u_ps_spec - H.u_ps_mfpt;
mx = max(abs(diff(:))); if mx == 0; mx = 1; end
contourf(ax, H.ys, H.xs, diff, linspace(-mx, mx, Nlev), 'LineColor', 'none');
clim(ax, [-mx, mx]);
colormap(ax, redblue(64)); c = colorbar(ax);
c.Label.String = '\Delta u  [ k_B T ]'; c.Label.FontSize = 18;
xlabel(ax, 'y'); ylabel(ax, 'x');
title(ax, '(d) Per-state spectral - MFPT', 'FontWeight', 'normal');
clean_ax(ax);

exportPNG('fig_grid2d_bias_profiles');
exportPDF('fig_grid2d_bias_profiles');
fprintf('Saved fig_grid2d_bias_profiles\n');

% =====================================================================
% FIG 3: budget sweep (gap and MFPT speedup vs U_max)
% =====================================================================
fig = figure('Position', [100 100 1300 500]);
tl = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax = nexttile; hold(ax, 'on');
loglog(ax, budget_U, budget_gap_flat,  '-o', 'Color', col_flat, ...
       'MarkerFaceColor', col_flat, 'MarkerSize', 12);
loglog(ax, budget_U, budget_gap_poly,  '-d', 'Color', col_poly, ...
       'MarkerFaceColor', col_poly, 'MarkerSize', 12);
loglog(ax, budget_U, budget_gap_ps,    '-^', 'Color', col_ps,   ...
       'MarkerFaceColor', col_ps,   'MarkerSize', 12);
loglog(ax, budget_U, budget_gap_mfpt,  '-s', 'Color', col_mfpt, ...
       'MarkerFaceColor', col_mfpt, 'MarkerSize', 12);
set(ax, 'XScale', 'log', 'YScale', 'log');
xlabel(ax, 'U_{max}  [ k_B T ]'); ylabel(ax, '\gamma / \gamma_0');
legend(ax, {'flatten', 'polynomial-spec', 'per-state spec', 'per-state MFPT'}, ...
       'Location', 'northwest', 'Box', 'off', 'FontSize', 16);
title(ax, '(a) Spectral-gap speedup', 'FontWeight', 'normal');
clean_ax(ax);

ax = nexttile; hold(ax, 'on');
loglog(ax, budget_U, budget_mfpt_flat,  '-o', 'Color', col_flat, ...
       'MarkerFaceColor', col_flat, 'MarkerSize', 12);
loglog(ax, budget_U, budget_mfpt_poly,  '-d', 'Color', col_poly, ...
       'MarkerFaceColor', col_poly, 'MarkerSize', 12);
loglog(ax, budget_U, budget_mfpt_ps,    '-^', 'Color', col_ps,   ...
       'MarkerFaceColor', col_ps,   'MarkerSize', 12);
loglog(ax, budget_U, budget_mfpt_mfpt,  '-s', 'Color', col_mfpt, ...
       'MarkerFaceColor', col_mfpt, 'MarkerSize', 12);
yline(ax, 1, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
set(ax, 'XScale', 'log', 'YScale', 'log');
xlabel(ax, 'U_{max}  [ k_B T ]'); ylabel(ax, 'MFPT_0 / MFPT');
legend(ax, {'flatten', 'polynomial-spec', 'per-state spec', 'per-state MFPT'}, ...
       'Location', 'northwest', 'Box', 'off', 'FontSize', 16);
title(ax, '(b) MFPT speedup', 'FontWeight', 'normal');
clean_ax(ax);

exportPNG('fig_grid2d_budget_sweep');
exportPDF('fig_grid2d_budget_sweep');
fprintf('Saved fig_grid2d_budget_sweep\n');

% =====================================================================
% FIG 4: regime sweep (vs FE_range / barrier height)
% =====================================================================
fig = figure('Position', [100 100 1300 500]);
tl = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax = nexttile; hold(ax, 'on');
loglog(ax, regime_FE, regime_gap_flat,  '-o', 'Color', col_flat, ...
       'MarkerFaceColor', col_flat, 'MarkerSize', 12);
loglog(ax, regime_FE, regime_gap_poly,  '-d', 'Color', col_poly, ...
       'MarkerFaceColor', col_poly, 'MarkerSize', 12);
loglog(ax, regime_FE, regime_gap_ps,    '-^', 'Color', col_ps,   ...
       'MarkerFaceColor', col_ps,   'MarkerSize', 12);
loglog(ax, regime_FE, regime_gap_mfpt,  '-s', 'Color', col_mfpt, ...
       'MarkerFaceColor', col_mfpt, 'MarkerSize', 12);
set(ax, 'XScale', 'log', 'YScale', 'log');
xlabel(ax, 'FE range  [ k_B T ]'); ylabel(ax, '\gamma / \gamma_0');
legend(ax, {'flatten', 'polynomial-spec', 'per-state spec', 'per-state MFPT'}, ...
       'Location', 'best', 'Box', 'off', 'FontSize', 16);
title(ax, '(a) Spectral-gap speedup', 'FontWeight', 'normal');
clean_ax(ax);

ax = nexttile; hold(ax, 'on');
loglog(ax, regime_FE, regime_mfpt_flat,  '-o', 'Color', col_flat, ...
       'MarkerFaceColor', col_flat, 'MarkerSize', 12);
loglog(ax, regime_FE, regime_mfpt_poly,  '-d', 'Color', col_poly, ...
       'MarkerFaceColor', col_poly, 'MarkerSize', 12);
loglog(ax, regime_FE, regime_mfpt_ps,    '-^', 'Color', col_ps,   ...
       'MarkerFaceColor', col_ps,   'MarkerSize', 12);
loglog(ax, regime_FE, regime_mfpt_mfpt,  '-s', 'Color', col_mfpt, ...
       'MarkerFaceColor', col_mfpt, 'MarkerSize', 12);
yline(ax, 1, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
set(ax, 'XScale', 'log', 'YScale', 'log');
xlabel(ax, 'FE range  [ k_B T ]'); ylabel(ax, 'MFPT_0 / MFPT');
legend(ax, {'flatten', 'polynomial-spec', 'per-state spec', 'per-state MFPT'}, ...
       'Location', 'best', 'Box', 'off', 'FontSize', 16);
title(ax, '(b) MFPT speedup', 'FontWeight', 'normal');
clean_ax(ax);

exportPNG('fig_grid2d_regime_sweep');
exportPDF('fig_grid2d_regime_sweep');
fprintf('Saved fig_grid2d_regime_sweep\n');

fprintf('\nAll 4 2D-grid figures written to %s\n', PATH);

% ---------------------------------------------------------------------
%  Helpers
% ---------------------------------------------------------------------
function try_pdf(PATH, fname)
    try
        exportgraphics(gcf, fullfile(PATH, [fname '.pdf']), ...
                       'ContentType', 'vector', 'BackgroundColor', 'white');
    catch err
        warning('PDF export skipped for %s: %s', fname, err.message);
    end
end

function cmap = redblue(n)
    if nargin < 1; n = 64; end
    half = floor(n/2);
    blue = [linspace(0.05, 1, half).', linspace(0.30, 1, half).', linspace(0.65, 1, half).'];
    red  = [linspace(1, 0.85, n-half).', linspace(1, 0.20, n-half).', linspace(1, 0.10, n-half).'];
    cmap = [blue; red];
end
