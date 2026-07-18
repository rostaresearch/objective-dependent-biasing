% 5-panel summary figure for the LiveCoMS perspective.
%   (a) landscape Z          (b) u_spec      (c) u_MFPT
%   (d) Z + u_spec           (e) Z + u_MFPT
% All at U_max = 6 k_BT.  Two shared vertical colorbars only:
%   left  = u [k_B T]   (redblue, panels b,c)
%   right = Z, Z+u [kcal/mol] (parula, panels a,d,e)
% Square panels, packed close, start (blue circle) / target (red square)
% labelled inside panel (a).

clear; close all;
PATH = getenv('MSM_ROOT');
if isempty(PATH), PATH = fileparts(fileparts(mfilename('fullpath'))); end
load(fullfile(PATH, 'rkhs_2d_data.mat'));

set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultAxesFontSize', 14);
set(groot, 'defaultAxesLineWidth', 1.4);
set(groot, 'defaultAxesTickDir', 'out');
set(groot, 'defaultTextFontName', 'Arial');
set(groot, 'defaultFigureColor', 'w');

k = 2;                       % U_max = 6 k_BT
Umax = U_max_list_kT(k);
Usp = eval(sprintf('u_spec_%d', k));
Umf = eval(sprintf('u_mfpt_%d', k));
sg  = eval(sprintf('speedup_gap_spec_%d', k));
sm  = eval(sprintf('speedup_mfpt_mfpt_%d', k));

cap = quantile(Z(:), 0.95);
Z_disp   = Z;        Z_disp(Z_disp > cap)     = cap;
Z_eff_sp = Z + Usp*kBT; Z_eff_sp(Z_eff_sp>cap)= cap;
Z_eff_mf = Z + Umf*kBT; Z_eff_mf(Z_eff_mf>cap)= cap;
zmin = min([Z_disp(:); Z_eff_sp(:); Z_eff_mf(:)]);
zmax = max([Z_disp(:); Z_eff_sp(:); Z_eff_mf(:)]);

scol = [0.10 0.45 0.85];     % start  -> blue circle
tcol = [0.85 0.20 0.15];     % target -> red square

fig = figure('Position', [40 40 1500 330]);

% --- panel geometry (normalised) ---
xL   = 0.105;                % first panel left (room for left colorbar)
pW   = 0.150;                % panel width
gap  = 0.014;
pB   = 0.205;                % panel bottom (room for x-label)
pH   = pW*1500/330;          % square in pixels
pos  = @(i) [xL+(i-1)*(pW+gap), pB, pW, pH];

titles = {'(a) landscape', ...
          sprintf('(b) u_{spec}  \\gamma %.0f\\times', sg), ...
          sprintf('(c) u_{MFPT}  MFPT %.0f\\times', sm), ...
          '(d) Z + u_{spec}', '(e) Z + u_{MFPT}'};
data    = {Z_disp, Usp, Umf, Z_eff_sp, Z_eff_mf};
cmaps   = {parula(64), redblue(64), redblue(64), parula(64), parula(64)};
isbias  = [false true true false false];

axh = gobjects(1,5);
for i = 1:5
    ax = axes('Position', pos(i)); axh(i) = ax;
    if isbias(i)
        contourf(ax, xs, ys, data{i}, linspace(-Umax,Umax,24), 'LineColor','none');
        clim(ax, [-Umax Umax]);
    else
        contourf(ax, xs, ys, data{i}, 24, 'LineColor','none');
        clim(ax, [zmin zmax]);
    end
    colormap(ax, cmaps{i});
    hold(ax,'on');
    hS = plot(ax, start_xy(1), start_xy(2), 'o', 'MarkerSize',13, ...
        'MarkerFaceColor',scol,'MarkerEdgeColor','w','LineWidth',1.5, ...
        'DisplayName','start');
    hT = plot(ax, end_xy(1), end_xy(2), 's', 'MarkerSize',14, ...
        'MarkerFaceColor',tcol,'MarkerEdgeColor','w','LineWidth',1.5, ...
        'DisplayName','target');
    xlim(ax,[0 2*pi]); ylim(ax,[0 2*pi]);
    set(ax,'XTick',[0 3 6],'YTick',[0 3 6],'FontSize',13, ...
        'TickDir','out','Layer','top','LineWidth',1.4);
    axis(ax,'square');
    xlabel(ax,'x','FontSize',14);
    if i==1, ylabel(ax,'y','FontSize',14); else, set(ax,'YTickLabel',[]); end
    title(ax, titles{i}, 'FontWeight','normal','FontSize',14);
    if i==1
        lg = legend(ax,[hS hT],'Location','northwest','FontSize',11, ...
                    'Color','w','EdgeColor',[0.7 0.7 0.7]);
        lg.ItemTokenSize = [12 12];
    end
end

% --- left colorbar: u [k_B T] (redblue) ---
axHL = axes('Position', pos(1)); set(axHL,'Visible','off');
colormap(axHL, redblue(64)); clim(axHL,[-Umax Umax]);
cbL = colorbar(axHL,'westoutside');
cbL.Position = [0.045 pB 0.013 pH];
cbL.Label.String = 'u  [ k_B T ]'; cbL.FontSize = 13;
cbL.Ticks = [-Umax 0 Umax];

% --- right colorbar: Z, Z+u [kcal/mol] (parula) ---
axHR = axes('Position', pos(5)); set(axHR,'Visible','off');
colormap(axHR, parula(64)); clim(axHR,[zmin zmax]);
cbR = colorbar(axHR,'eastoutside');
cbR.Position = [0.945 pB 0.013 pH];
cbR.Label.String = 'Z, Z+u  [ kcal/mol ]'; cbR.FontSize = 13;

exportgraphics(gcf, fullfile(PATH,'fig_rkhs_5panel.png'), ...
               'Resolution', 600, 'BackgroundColor','white');
try
    exportgraphics(gcf, fullfile(PATH,'fig_rkhs_5panel.pdf'), ...
                   'ContentType','vector','BackgroundColor','white');
catch err
    warning('PDF export skipped: %s', err.message);
end
fprintf('Saved fig_rkhs_5panel.png/.pdf\n');

% ---- helpers ------------------------------------------------------------
function cmap = redblue(n)
    if nargin < 1; n = 64; end
    half = floor(n/2);
    blue = [linspace(0.05,1,half).', linspace(0.30,1,half).', linspace(0.65,1,half).'];
    red  = [linspace(1,0.85,n-half).', linspace(1,0.20,n-half).', linspace(1,0.10,n-half).'];
    cmap = [blue; red];
end
