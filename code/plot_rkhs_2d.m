% Plot the 9-well 2D landscape from RKHS/DHAM2D and the per-state
% spectral/MFPT-optimal biases applied to it.
%
% Loads rkhs_2d_data.mat produced by rkhs_2d_check.py.

clear; close all;
PATH = getenv('MSM_ROOT');
if isempty(PATH), PATH = fileparts(fileparts(mfilename('fullpath'))); end
load(fullfile(PATH, 'rkhs_2d_data.mat'));

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

% =====================================================================
% FIG 1: landscape (Z, kcal/mol) with start/end markers
% =====================================================================
fig = figure('Position', [100 100 720 580]);
ax = axes;
contourf(ax, xs, ys, Z, 24, 'LineColor', 'none');
colormap(ax, parula(64));
hold(ax, 'on');
plot(ax, start_xy(1), start_xy(2), 'o', 'MarkerSize', 20, ...
     'MarkerFaceColor', [0.10 0.45 0.75], 'MarkerEdgeColor', 'w', 'LineWidth', 2);
plot(ax, end_xy(1), end_xy(2), 's', 'MarkerSize', 20, ...
     'MarkerFaceColor', [0.85 0.30 0.10], 'MarkerEdgeColor', 'w', 'LineWidth', 2);
text(ax, start_xy(1) + 0.25, start_xy(2) + 0.30, 'start', ...
     'Color', [0.10 0.45 0.75], 'FontSize', 18, 'FontWeight', 'normal');
text(ax, end_xy(1) + 0.25, end_xy(2) + 0.30, 'end', ...
     'Color', [0.85 0.30 0.10], 'FontSize', 18, 'FontWeight', 'normal');
c = colorbar(ax);
c.Label.String = 'Z  [ kcal/mol ]';
c.Label.FontSize = 18;
xlabel(ax, 'x  [ nm ]'); ylabel(ax, 'y  [ nm ]');
xlim(ax, [0, 2*pi]); ylim(ax, [0, 2*pi]);
axis(ax, 'square');
clean_ax(ax);
title(ax, '9-well landscape', 'FontWeight', 'normal');

exportPNG('fig_rkhs_landscape');
exportPDF('fig_rkhs_landscape');
fprintf('Saved fig_rkhs_landscape\n');

% =====================================================================
% FIG 2: per-state optimal bias for each U_max, spectral vs MFPT
% =====================================================================
nU = numel(U_max_list_kT);
fig = figure('Position', [100 100 1500 380*nU]);
tl = tiledlayout(nU, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
panels = 'abcdefghij';
pidx = 1;

for k = 1:nU
    Umax = U_max_list_kT(k);
    levs = linspace(-Umax, Umax, 24);
    Usp = eval(sprintf('u_spec_%d', k));
    Umf = eval(sprintf('u_mfpt_%d', k));
    sg = eval(sprintf('speedup_gap_spec_%d', k));
    sm = eval(sprintf('speedup_mfpt_mfpt_%d', k));

    ax = nexttile;
    contourf(ax, xs, ys, Usp, levs, 'LineColor', 'none');
    clim(ax, [-Umax, Umax]);
    colormap(ax, redblue(64));
    c = colorbar(ax); c.Label.String = 'u  [ k_B T ]'; c.Label.FontSize = 18;
    hold(ax, 'on');
    plot(ax, start_xy(1), start_xy(2), 'o', 'MarkerSize', 18, ...
         'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w', 'LineWidth', 2);
    plot(ax, end_xy(1), end_xy(2), 's', 'MarkerSize', 18, ...
         'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w', 'LineWidth', 2);
    xlabel(ax, 'x  [ nm ]'); ylabel(ax, 'y  [ nm ]');
    xlim(ax, [0, 2*pi]); ylim(ax, [0, 2*pi]);
    axis(ax, 'square'); clean_ax(ax);
    title(ax, sprintf('(%s) Spectral-opt  U_{max}=%g k_BT  \\gamma %.1f\\times', ...
                       panels(pidx), Umax, sg), 'FontWeight', 'normal');
    pidx = pidx + 1;

    ax = nexttile;
    contourf(ax, xs, ys, Umf, levs, 'LineColor', 'none');
    clim(ax, [-Umax, Umax]);
    colormap(ax, redblue(64));
    c = colorbar(ax); c.Label.String = 'u  [ k_B T ]'; c.Label.FontSize = 18;
    hold(ax, 'on');
    plot(ax, start_xy(1), start_xy(2), 'o', 'MarkerSize', 18, ...
         'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w', 'LineWidth', 2);
    plot(ax, end_xy(1), end_xy(2), 's', 'MarkerSize', 18, ...
         'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w', 'LineWidth', 2);
    xlabel(ax, 'x  [ nm ]'); ylabel(ax, 'y  [ nm ]');
    xlim(ax, [0, 2*pi]); ylim(ax, [0, 2*pi]);
    axis(ax, 'square'); clean_ax(ax);
    title(ax, sprintf('(%s) MFPT-opt  U_{max}=%g k_BT  MFPT %.1f\\times', ...
                       panels(pidx), Umax, sm), 'FontWeight', 'normal');
    pidx = pidx + 1;
end

exportPNG('fig_rkhs_bias_profiles');
exportPDF('fig_rkhs_bias_profiles');
fprintf('Saved fig_rkhs_bias_profiles\n');

% =====================================================================
% FIG 3: effective potential Z + u*kBT for each (U_max, objective)
% =====================================================================
fig = figure('Position', [100 100 1500 380*nU]);
tl = tiledlayout(nU, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
pidx = 1;

% Use a common Z range across all panels for fair comparison
zmin_all = inf; zmax_all = -inf;
for k = 1:nU
    Umax = U_max_list_kT(k);
    Usp = eval(sprintf('u_spec_%d', k));
    Umf = eval(sprintf('u_mfpt_%d', k));
    Z_eff_sp = Z + Usp * kBT;
    Z_eff_mf = Z + Umf * kBT;
    % Clip the extreme boundary-wall values for visualisation
    cap = quantile(Z(:), 0.95);
    Z_eff_sp(Z_eff_sp > cap) = cap;
    Z_eff_mf(Z_eff_mf > cap) = cap;
    zmin_all = min([zmin_all, min(Z_eff_sp(:)), min(Z_eff_mf(:))]);
    zmax_all = max([zmax_all, max(Z_eff_sp(:)), max(Z_eff_mf(:))]);
end

for k = 1:nU
    Umax = U_max_list_kT(k);
    Usp = eval(sprintf('u_spec_%d', k));
    Umf = eval(sprintf('u_mfpt_%d', k));
    cap = quantile(Z(:), 0.95);
    Z_eff_sp = Z + Usp * kBT;
    Z_eff_mf = Z + Umf * kBT;
    Z_eff_sp(Z_eff_sp > cap) = cap;
    Z_eff_mf(Z_eff_mf > cap) = cap;

    levs = linspace(zmin_all, zmax_all, 24);

    ax = nexttile;
    contourf(ax, xs, ys, Z_eff_sp, levs, 'LineColor', 'none');
    clim(ax, [zmin_all, zmax_all]);
    colormap(ax, parula(64));
    c = colorbar(ax); c.Label.String = 'Z+u  [ kcal/mol ]'; c.Label.FontSize = 18;
    hold(ax, 'on');
    plot(ax, start_xy(1), start_xy(2), 'o', 'MarkerSize', 18, ...
         'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    plot(ax, end_xy(1), end_xy(2), 's', 'MarkerSize', 18, ...
         'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    xlabel(ax, 'x'); ylabel(ax, 'y');
    xlim(ax, [0, 2*pi]); ylim(ax, [0, 2*pi]); axis(ax, 'square'); clean_ax(ax);
    title(ax, sprintf('Spectral-opt  U_{max}=%g k_BT', Umax), 'FontWeight', 'normal');

    ax = nexttile;
    contourf(ax, xs, ys, Z_eff_mf, levs, 'LineColor', 'none');
    clim(ax, [zmin_all, zmax_all]);
    colormap(ax, parula(64));
    c = colorbar(ax); c.Label.String = 'Z+u  [ kcal/mol ]'; c.Label.FontSize = 18;
    hold(ax, 'on');
    plot(ax, start_xy(1), start_xy(2), 'o', 'MarkerSize', 18, ...
         'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    plot(ax, end_xy(1), end_xy(2), 's', 'MarkerSize', 18, ...
         'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
    xlabel(ax, 'x'); ylabel(ax, 'y');
    xlim(ax, [0, 2*pi]); ylim(ax, [0, 2*pi]); axis(ax, 'square'); clean_ax(ax);
    title(ax, sprintf('MFPT-opt  U_{max}=%g k_BT', Umax), 'FontWeight', 'normal');
end

exportPNG('fig_rkhs_effective');
exportPDF('fig_rkhs_effective');
fprintf('Saved fig_rkhs_effective\n');
fprintf('\nAll RKHS-2D figures written to %s\n', PATH);

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
