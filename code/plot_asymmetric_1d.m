% Paper-quality figures for the 1D asymmetric double-well study.
% Loads asymmetric_1d.mat (produced by asymmetric_1d.py).
%
% Typography: Arial 24 pt for everything, no top/right spines, ticks outward,
% no grid, white background. Saves PNG (600 dpi) and PDF (vector) for each.

clear; close all;
PATH = getenv('MSM_ROOT');
if isempty(PATH), PATH = fileparts(fileparts(mfilename('fullpath'))); end
load(fullfile(PATH, 'asymmetric_1d.mat'));

% ---- Global defaults (apply for every axes we create) ----------------
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultAxesFontSize', 18);
set(groot, 'defaultAxesLineWidth', 1.8);
set(groot, 'defaultAxesTickDir', 'out');
set(groot, 'defaultAxesBox', 'off');
set(groot, 'defaultAxesXGrid', 'off');
set(groot, 'defaultAxesYGrid', 'off');
set(groot, 'defaultTextFontName', 'Arial');
set(groot, 'defaultTextFontSize', 18);
set(groot, 'defaultLineLineWidth', 2.2);
set(groot, 'defaultFigureColor', 'w');

% Convenient handle for axis cleanup (remove top/right spines).
clean_ax = @(ax) set(ax, 'TickDir', 'out', 'Box', 'off', ...
                    'XGrid', 'off', 'YGrid', 'off', ...
                    'LineWidth', 1.8, 'FontSize', 18);

% Make sure exports use high quality
exportPNG = @(fname) exportgraphics(gcf, fullfile(PATH, [fname '.png']), ...
                                     'Resolution', 600, 'BackgroundColor', 'white');
% PDF export wrapper: a one-liner try-catch using cellfun would be opaque, so
% we use a small helper closure with arrayfun to swallow exceptions.
exportPDF = @(fname) try_export_pdf(PATH, fname);

% Pretty colors (consistent with the rest of the paper)
col_landscape = [0.20 0.20 0.20];     % near-black
col_basin_A   = [0.10 0.45 0.75];     % blue
col_basin_B   = [0.85 0.30 0.10];     % orange-red
col_spec      = [0.10 0.45 0.75];     % blue for spectral-opt bias
col_mfpt      = [0.85 0.30 0.10];     % orange-red for MFPT-opt bias
col_neutral   = [0.50 0.50 0.50];     % gray reference

% =====================================================================
% FIGURE 1: landscape + stationary
% =====================================================================
fig = figure('Position', [100 100 1100 500]);
tl = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile;
plot(ax1, x, F, 'Color', col_landscape, 'LineWidth', 2.5); hold(ax1, 'on');
plot(ax1, x(iA), F(iA), 'o', 'MarkerSize', 14, ...
     'MarkerFaceColor', col_basin_A, 'MarkerEdgeColor', col_basin_A);
plot(ax1, x(iB), F(iB), 'o', 'MarkerSize', 14, ...
     'MarkerFaceColor', col_basin_B, 'MarkerEdgeColor', col_basin_B);
text(ax1, x(iA), F(iA) + 2.8, 'A', 'FontSize', 18, 'FontWeight', 'normal', ...
     'Color', col_basin_A, 'HorizontalAlignment', 'center');
text(ax1, x(iB), F(iB) + 2.8, 'B', 'FontSize', 18, 'FontWeight', 'normal', ...
     'Color', col_basin_B, 'HorizontalAlignment', 'center');
xlabel(ax1, 'x', 'FontSize', 18);
ylabel(ax1, 'V(x)  [ k_B T ]', 'FontSize', 18);
title(ax1, '(a) Asymmetric landscape', 'FontSize', 18, 'FontWeight', 'normal');
clean_ax(ax1);

ax2 = nexttile;
plot(ax2, x, pi0, 'Color', col_landscape, 'LineWidth', 2.5);
hold(ax2, 'on');
plot(ax2, [x(iA) x(iA)], [0 max(pi0)*1.05], '--', 'Color', col_basin_A, ...
     'LineWidth', 1.5);
plot(ax2, [x(iB) x(iB)], [0 max(pi0)*1.05], '--', 'Color', col_basin_B, ...
     'LineWidth', 1.5);
xlabel(ax2, 'x', 'FontSize', 18);
ylabel(ax2, '\pi_0(x)', 'FontSize', 18);
title(ax2, '(b) Equilibrium distribution', 'FontSize', 18, ...
      'FontWeight', 'normal');
clean_ax(ax2);

exportPNG('fig_asym1d_landscape');
exportPDF('fig_asym1d_landscape');
fprintf('Saved fig_asym1d_landscape.png/.pdf\n');

% =====================================================================
% FIGURE 2: optimal bias profiles, all U_max values, 2 columns
% (left col: spectral-opt;  right col: MFPT-opt)
% =====================================================================
nU = numel(U_max_list);
fig = figure('Position', [100 100 1500 380*nU]);
tl = tiledlayout(nU, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

panels = 'abcdefghij';
panel_idx = 1;
for k = 1:nU
    Umax = U_max_list(k);

    % spectral-opt
    ax = nexttile;
    plot(ax, x, u_spectral(:, k), 'Color', col_spec, 'LineWidth', 2.5);
    hold(ax, 'on');
    yline(ax, +Umax, ':', 'Color', col_neutral, 'LineWidth', 1.2);
    yline(ax, -Umax, ':', 'Color', col_neutral, 'LineWidth', 1.2);
    yline(ax, 0, '-',  'Color', col_neutral, 'LineWidth', 0.8);
    xlabel(ax, 'x', 'FontSize', 18);
    ylabel(ax, 'u(x)  [ k_B T ]', 'FontSize', 18);
    title(ax, sprintf('(%s) Spectral-opt  U_{max}=%g   %.1f\\times', ...
                      panels(panel_idx), Umax, speedup_gap_spectral(k)), ...
          'FontSize', 18, 'FontWeight', 'normal');
    clean_ax(ax);
    panel_idx = panel_idx + 1;

    % MFPT-opt
    ax = nexttile;
    plot(ax, x, u_mfpt(:, k), 'Color', col_mfpt, 'LineWidth', 2.5);
    hold(ax, 'on');
    yline(ax, +Umax, ':', 'Color', col_neutral, 'LineWidth', 1.2);
    yline(ax, -Umax, ':', 'Color', col_neutral, 'LineWidth', 1.2);
    yline(ax, 0, '-',  'Color', col_neutral, 'LineWidth', 0.8);
    xlabel(ax, 'x', 'FontSize', 18);
    ylabel(ax, 'u(x)  [ k_B T ]', 'FontSize', 18);
    title(ax, sprintf('(%s) MFPT-opt  U_{max}=%g   %.1f\\times', ...
                      panels(panel_idx), Umax, speedup_mfpt_mfpt(k)), ...
          'FontSize', 18, 'FontWeight', 'normal');
    clean_ax(ax);
    panel_idx = panel_idx + 1;
end

exportPNG('fig_asym1d_bias_profiles');
exportPDF('fig_asym1d_bias_profiles');
fprintf('Saved fig_asym1d_bias_profiles.png/.pdf\n');

% =====================================================================
% FIGURE 3: speedup vs budget, 2 panels (gap and MFPT)
% =====================================================================
fig = figure('Position', [100 100 1100 500]);
tl = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% gamma speedup
ax = nexttile;
loglog(ax, U_max_list, speedup_gap_spectral, '-o', ...
       'Color', col_spec, 'MarkerFaceColor', col_spec, ...
       'MarkerSize', 14, 'LineWidth', 2.5);
hold(ax, 'on');
loglog(ax, U_max_list, speedup_gap_mfpt, '-s', ...
       'Color', col_mfpt, 'MarkerFaceColor', col_mfpt, ...
       'MarkerSize', 14, 'LineWidth', 2.5);
xlabel(ax, 'U_{max}  [ k_B T ]', 'FontSize', 18);
ylabel(ax, '\gamma / \gamma_0', 'FontSize', 18);
title(ax, '(a) Spectral-gap speedup', 'FontSize', 18, 'FontWeight', 'normal');
lgd = legend(ax, {'spectral-opt', 'MFPT-opt'}, 'Location', 'northwest', ...
             'FontSize', 18, 'Box', 'off');
clean_ax(ax);

% MFPT speedup
ax = nexttile;
loglog(ax, U_max_list, speedup_mfpt_spectral, '-o', ...
       'Color', col_spec, 'MarkerFaceColor', col_spec, ...
       'MarkerSize', 14, 'LineWidth', 2.5);
hold(ax, 'on');
loglog(ax, U_max_list, speedup_mfpt_mfpt, '-s', ...
       'Color', col_mfpt, 'MarkerFaceColor', col_mfpt, ...
       'MarkerSize', 14, 'LineWidth', 2.5);
yline(ax, 1.0, ':', 'Color', col_neutral, 'LineWidth', 1.5);
xlabel(ax, 'U_{max}  [ k_B T ]', 'FontSize', 18);
ylabel(ax, 'MFPT_0 / MFPT', 'FontSize', 18);
title(ax, '(b) MFPT speedup', 'FontSize', 18, 'FontWeight', 'normal');
lgd = legend(ax, {'spectral-opt', 'MFPT-opt'}, 'Location', 'northwest', ...
             'FontSize', 18, 'Box', 'off');
clean_ax(ax);

exportPNG('fig_asym1d_speedups');
exportPDF('fig_asym1d_speedups');
fprintf('Saved fig_asym1d_speedups.png/.pdf\n');

% =====================================================================
% FIGURE 4: effective profiles V(x) + u(x) for each (U_max, objective)
% Baseline V(x) as dashed gray reference.
% =====================================================================
fig = figure('Position', [100 100 1500 380*nU]);
tl = tiledlayout(nU, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

panels = 'abcdefghij';
panel_idx = 1;
for k = 1:nU
    Umax = U_max_list(k);

    % spectral-opt effective profile
    ax = nexttile;
    plot(ax, x, F, '--', 'Color', col_neutral, 'LineWidth', 1.8); hold(ax, 'on');
    plot(ax, x, F(:) + u_spectral(:, k), 'Color', col_spec, 'LineWidth', 2.5);
    yline(ax, 0, '-', 'Color', col_neutral, 'LineWidth', 0.6);
    xlabel(ax, 'x', 'FontSize', 18);
    ylabel(ax, 'V(x) + u(x)  [ k_B T ]', 'FontSize', 18);
    title(ax, sprintf('(%s) Spectral-opt  U_{max}=%g', panels(panel_idx), Umax), ...
          'FontSize', 18, 'FontWeight', 'normal');
    if k == 1
        legend(ax, {'V(x)', 'V(x) + u_{spec}(x)'}, 'Location', 'north', ...
               'FontSize', 16, 'Box', 'off');
    end
    clean_ax(ax);
    panel_idx = panel_idx + 1;

    % MFPT-opt effective profile
    ax = nexttile;
    plot(ax, x, F, '--', 'Color', col_neutral, 'LineWidth', 1.8); hold(ax, 'on');
    plot(ax, x, F(:) + u_mfpt(:, k), 'Color', col_mfpt, 'LineWidth', 2.5);
    yline(ax, 0, '-', 'Color', col_neutral, 'LineWidth', 0.6);
    xlabel(ax, 'x', 'FontSize', 18);
    ylabel(ax, 'V(x) + u(x)  [ k_B T ]', 'FontSize', 18);
    title(ax, sprintf('(%s) MFPT-opt  U_{max}=%g', panels(panel_idx), Umax), ...
          'FontSize', 18, 'FontWeight', 'normal');
    if k == 1
        legend(ax, {'V(x)', 'V(x) + u_{mfpt}(x)'}, 'Location', 'north', ...
               'FontSize', 16, 'Box', 'off');
    end
    clean_ax(ax);
    panel_idx = panel_idx + 1;
end

exportPNG('fig_asym1d_effective');
exportPDF('fig_asym1d_effective');
fprintf('Saved fig_asym1d_effective.png/.pdf\n');

fprintf('\nAll 4 asymmetric-1D figures written to %s\n', PATH);

function try_export_pdf(PATH, fname)
    try
        exportgraphics(gcf, fullfile(PATH, [fname '.pdf']), ...
                       'ContentType', 'vector', 'BackgroundColor', 'white');
    catch err
        warning('PDF export skipped for %s: %s', fname, err.message);
    end
end
