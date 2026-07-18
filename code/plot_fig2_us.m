% Manuscript Figure 2 -- "Analytic coverage speedup per CV" -- in MATLAB,
% saved as an EDITABLE fig2.fig (open in MATLAB, restyle anything you like).
%
% Replaces the matplotlib fig_final.png. Data comes from export_fig2_mat.py
% (fig2_data.mat), which mirrors make_final_figure.py / filtered_results.json.
%
% Typography: Arial, LARGE fonts (axes 20, labels 22, annotations 15).
% To make letters even bigger, edit FS_* below and re-run, or open fig2.fig
% and use the Property Inspector.

clear; close all;
PATH = getenv('MSM_ROOT');
if isempty(PATH), PATH = fileparts(fileparts(mfilename('fullpath'))); end
S = load(fullfile(PATH, 'fig2_data.mat'));

% ---- font sizes (bump these if you want them larger still) ----
FS_AX    = 20;   % tick labels
FS_LAB   = 22;   % axis labels
FS_TITLE = 22;   % panel titles
FS_ANNOT = 15;   % the W/kappa/O annotations
FS_LEG   = 16;   % legend

set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');
set(groot, 'defaultAxesFontSize', FS_AX);
set(groot, 'defaultAxesLineWidth', 1.8);
set(groot, 'defaultAxesTickDir', 'out');
set(groot, 'defaultAxesBox', 'off');
set(groot, 'defaultFigureColor', 'w');

col_b      = [0.12 0.47 0.71];    % barrier (blue)
col_r      = [0.84 0.15 0.16];    % rate (red)
col_b_soft = [0.81 0.89 1.00];
col_r_soft = [1.00 0.84 0.84];

panels = {S.grid, S.penta};
titles = {S.title_grid, S.title_penta};

fig = figure('Position', [80 80 1700 700]);
tl = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for ip = 1:2
    D = panels{ip};
    cvs = cellstr(string(D.cvs));      % robust to cell / string array
    n = numel(cvs);
    x = 1:n;
    w = 0.36;

    ub_barr = double(D.ub_barr(:)).';
    ub_rate = double(D.ub_rate(:)).';
    v_barr  = double(D.v_barr(:)).';
    v_rate  = double(D.v_rate(:)).';

    ax = nexttile; hold(ax, 'on');

    % --- unconstrained (light fill + strong edge) = "may be invalid" ---
    hub_b = bar(ax, x - w/2, ub_barr, w, 'FaceColor', col_b_soft, ...
                'EdgeColor', col_b, 'LineWidth', 2.0);
    hub_r = bar(ax, x + w/2, ub_rate, w, 'FaceColor', col_r_soft, ...
                'EdgeColor', col_r, 'LineWidth', 2.0);

    % --- valid (solid) ---
    hv_b = bar(ax, x - w/2, v_barr, w, 'FaceColor', col_b, ...
               'EdgeColor', 'k', 'LineWidth', 1.2);
    hv_r = bar(ax, x + w/2, v_rate, w, 'FaceColor', col_r, ...
               'EdgeColor', 'k', 'LineWidth', 1.2);

    % --- oracle flattening reference lines ---
    ob = double(D.oracle_b); orr = double(D.oracle_r);
    hob = yline(ax, ob, ':', 'Color', col_b, 'LineWidth', 2.2);
    hor = yline(ax, orr, ':', 'Color', col_r, 'LineWidth', 2.2);
    yline(ax, 1, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);

    set(ax, 'YScale', 'log');

    % --- annotations (W, kappa, overlap) ---
    lb = cellstr(string(D.lab_barr));
    lr = cellstr(string(D.lab_rate));
    for i = 1:n
        if isfinite(ub_barr(i)) && ~isempty(lb{i})
            text(ax, x(i)-w/2, ub_barr(i)*1.30, lb{i}, 'Rotation', 12, ...
                 'HorizontalAlignment', 'center', 'FontSize', FS_ANNOT, ...
                 'Color', col_b, 'FontName', 'Arial');
        end
        if isfinite(ub_rate(i)) && ~isempty(lr{i})
            text(ax, x(i)+w/2, ub_rate(i)*1.30, lr{i}, 'Rotation', 12, ...
                 'HorizontalAlignment', 'center', 'FontSize', FS_ANNOT, ...
                 'Color', col_r, 'FontName', 'Arial');
        end
    end

    % --- limits with headroom for the rotated annotations ---
    allv = [ub_barr ub_rate v_barr v_rate ob orr 1];
    allv = allv(isfinite(allv) & allv > 0);
    set(ax, 'XLim', [0.4 n+0.6], 'XTick', x, 'XTickLabel', cvs, ...
        'YLim', [10^(floor(log10(min(allv)))-0.15), 10^(ceil(log10(max(allv)))+0.55)]);

    xlabel(ax, 'collective variable', 'FontSize', FS_LAB);
    ylabel(ax, 'speedup vs unbiased', 'FontSize', FS_LAB);
    title(ax, titles{ip}, 'FontSize', FS_TITLE, 'FontWeight', 'normal');
    set(ax, 'FontSize', FS_AX, 'TickDir', 'out', 'Box', 'off', 'LineWidth', 1.8);
    grid(ax, 'off');

    if ip == 1
        lgd = legend(ax, [hub_b hub_r hv_b hv_r hob hor], ...
            {'unconstrained barrier', 'unconstrained rate', ...
             'valid barrier', 'valid rate', ...
             'oracle flatten (barrier)', 'oracle flatten (rate)'}, ...
            'Box', 'off', 'FontSize', FS_LEG, 'NumColumns', 3);
        lgd.Layout.Tile = 'south';   % shared legend, clear of the bar annotations
    end
end

title(tl, ['Analytic umbrella speedup per CV.  Light+outlined = mathematical optimum ' ...
           '(may fail overlap/coverage).  Solid = valid protocol (O\geq0.05, \Phi<0.01, W\geq2).'], ...
      'FontSize', 17, 'FontName', 'Arial');

% ---- save EDITABLE .fig + publication rasters ----
savefig(fig, fullfile(PATH, 'fig2.fig'));
exportgraphics(fig, fullfile(PATH, 'fig2.png'), 'Resolution', 600, 'BackgroundColor', 'white');
try
    exportgraphics(fig, fullfile(PATH, 'fig2.pdf'), 'ContentType', 'vector', 'BackgroundColor', 'white');
catch err
    warning('PDF export skipped: %s', err.message);
end
fprintf('Saved fig2.fig (editable), fig2.png, fig2.pdf to %s\n', PATH);
