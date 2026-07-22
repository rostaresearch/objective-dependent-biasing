% Regenerate the four Ras figures from the CORRECTED Pi-unbinding audit
% (Balint's 1D_PMg MFPT runs). Replaces the orchestrator-derived versions.
%
% Data: ras_pi_fig_data.mat  (from export_ras_pi_mat.py)
% Outputs: fig_ras_applied_audit, fig_ras_alpha_sensitivity,
%          fig_ras_grid_feature_sensitivity, fig_ras_pareto_demo  (.png/.pdf/.fig)
%
% Large fonts throughout (edit FS_* to taste; .fig files are editable).

clear; close all;
% Resolve the data root portably (no machine-specific paths, per the DAS):
% honour $MSM_ROOT if set, else auto-detect whether this script sits at the
% data root or inside a code/ subdirectory of it.
PATH = getenv('MSM_ROOT');
if isempty(PATH)
    PATH = fileparts(fileparts(mfilename('fullpath')));   % <root>/code/ -> <root>
end
DATA = fullfile(PATH, 'data');
FIGURES = fullfile(PATH, 'figures');
S = load(fullfile(DATA, 'ras_pi_fig_data.mat'));

FS_AX = 20; FS_LAB = 22; FS_TITLE = 21; FS_LEG = 16;
set(groot,'defaultAxesFontName','Arial'); set(groot,'defaultTextFontName','Arial');
set(groot,'defaultAxesFontSize',FS_AX); set(groot,'defaultAxesLineWidth',1.8);
set(groot,'defaultAxesTickDir','out'); set(groot,'defaultAxesBox','off');
set(groot,'defaultFigureColor','w');
col_o = [0.12 0.47 0.71];   % Omega_J
col_d = [0.84 0.15 0.16];   % D_edge

sv = @(f,n) saveall(f, fullfile(FIGURES,n));

%% ---- FIG 1: applied-bias audit (canonical bars + 90% CI) ----
f1 = figure('Position',[80 80 900 640]); ax = axes(f1); hold(ax,'on');
med = double(S.canon_med(:)).'; lo = double(S.canon_lo(:)).'; hi = double(S.canon_hi(:)).';
x = [1 2];
bar(ax, x(1), med(1), 0.55, 'FaceColor', col_o, 'EdgeColor','k','LineWidth',1.2);
bar(ax, x(2), med(2), 0.55, 'FaceColor', col_d, 'EdgeColor','k','LineWidth',1.2);
errorbar(ax, x, med, med-lo, hi-med, 'k', 'LineStyle','none','LineWidth',2.4,'CapSize',18);
for i=1:2
    text(ax, x(i), hi(i)+0.055, sprintf('%.3f\n[%.3f, %.3f]',med(i),lo(i),hi(i)), ...
        'HorizontalAlignment','center','FontSize',17,'FontName','Arial');
end
set(ax,'XTick',x,'XTickLabel',{'\Omega_J','D_{edge}'},'XLim',[0.4 2.6],'YLim',[0 1.25]);
ylabel(ax,'value','FontSize',FS_LAB);
title(ax, sprintf(['Ras phosphate unbinding at the applied bias\n' ...
      '12\\times8 grid, |A|=%d, |B|=%d, 200 block-bootstrap resamples over runs'], ...
      double(S.canon_A), double(S.canon_B)), 'FontSize',FS_TITLE,'FontWeight','normal');
clean(ax,FS_AX); sv(f1,'fig_ras_applied_audit');

%% ---- FIG 2: alpha (pseudocount) sensitivity ----
f2 = figure('Position',[80 80 1200 560]);
tl = tiledlayout(f2,1,2,'Padding','compact','TileSpacing','compact');
al = double(S.al_alpha(:)); ao = double(S.al_oj(:)); ad = double(S.al_de(:));
ax = nexttile(tl); semilogx(ax, al, ao, '-o','Color',col_o,'MarkerFaceColor',col_o,'MarkerSize',13,'LineWidth',2.6);
set(ax,'YLim',[0 1.1]); xlabel(ax,'pseudocount \alpha','FontSize',FS_LAB);
ylabel(ax,'\Omega_J','FontSize',FS_LAB); title(ax,'(a) current overlap','FontSize',FS_TITLE,'FontWeight','normal'); clean(ax,FS_AX);
ax = nexttile(tl); semilogx(ax, al, ad, '-s','Color',col_d,'MarkerFaceColor',col_d,'MarkerSize',13,'LineWidth',2.6);
set(ax,'YLim',[0 0.12]); xlabel(ax,'pseudocount \alpha','FontSize',FS_LAB);
ylabel(ax,'D_{edge}','FontSize',FS_LAB); title(ax,'(b) edge distortion','FontSize',FS_TITLE,'FontWeight','normal'); clean(ax,FS_AX);
title(tl, 'Regularizer never activates (0/200 resamples)', 'FontSize',17,'FontName','Arial');
sv(f2,'fig_ras_alpha_sensitivity');

%% ---- FIG 3: grid x feature sensitivity ----
f3 = figure('Position',[60 60 1500 640]);
tl = tiledlayout(f3,1,2,'Padding','compact','TileSpacing','compact');
labs = cellstr(string(S.gf_labels));
% Prettify the code-style feature names into readable TeX tick labels.
% (The raw underscores are field separators, not subscripts: MATLAB's default
%  TeX interpreter would subscript the single char after each '_'.)
labs = strrep(labs, '10x7', '10\times7');
labs = strrep(labs, '12x8', '12\times8');
labs = strrep(labs, '14x9', '14\times9');
labs = strrep(labs, 'Pi_Q61',     'P_{i}-Gln61');        % phosphate-Gln61 distance
labs = strrep(labs, 'Pi_wat_4.5', 'P_{i}-water');        % phosphate solvation
labs = strrep(labs, 'Mg_O_coord', 'Mg-O coord');         % Mg(2+)-oxygen coordination
n = numel(labs); xx = 1:n;
oj=double(S.gf_oj(:)).'; ojl=double(S.gf_oj_lo(:)).'; ojh=double(S.gf_oj_hi(:)).';
de=double(S.gf_de(:)).'; del=double(S.gf_de_lo(:)).'; deh=double(S.gf_de_hi(:)).';
ax = nexttile(tl); hold(ax,'on');
bar(ax, xx, oj, 0.6,'FaceColor',col_o,'EdgeColor','k','LineWidth',1.1);
errorbar(ax, xx, oj, oj-ojl, ojh-oj,'k','LineStyle','none','LineWidth',2,'CapSize',12);
set(ax,'XTick',xx,'XTickLabel',labs,'XLim',[0.4 n+0.6],'YLim',[0 1.15]);
xtickangle(ax,32); ylabel(ax,'\Omega_J','FontSize',FS_LAB);
title(ax,'(a) reactive-current overlap','FontSize',FS_TITLE,'FontWeight','normal'); clean(ax,FS_AX);
ax = nexttile(tl); hold(ax,'on');
bar(ax, xx, de, 0.6,'FaceColor',col_d,'EdgeColor','k','LineWidth',1.1);
errorbar(ax, xx, de, de-del, deh-de,'k','LineStyle','none','LineWidth',2,'CapSize',12);
set(ax,'XTick',xx,'XTickLabel',labs,'XLim',[0.4 n+0.6],'YLim',[0 0.09]);
xtickangle(ax,32); ylabel(ax,'D_{edge}','FontSize',FS_LAB);
title(ax,'(b) edge-rate distortion','FontSize',FS_TITLE,'FontWeight','normal'); clean(ax,FS_AX);
title(tl,'Grid and auxiliary-feature sensitivity (90% bootstrap intervals)','FontSize',17,'FontName','Arial');
sv(f3,'fig_ras_grid_feature_sensitivity');

%% ---- FIG 4: bias-scaling / Pareto ----
f4 = figure('Position',[60 60 1500 620]);
tl = tiledlayout(f4,1,2,'Padding','compact','TileSpacing','compact');
a=double(S.sc_alpha(:)); so=double(S.sc_oj(:)); sd=double(S.sc_de(:)); sm=double(S.sc_smfpt(:));
% block-bootstrap-over-runs 5/95 bands (ras_pareto_boot.py)
ojl=double(S.sc_oj_lo(:)); ojh=double(S.sc_oj_hi(:));
del=double(S.sc_de_lo(:)); deh=double(S.sc_de_hi(:));
sml=double(S.sc_sm_lo(:)); smh=double(S.sc_sm_hi(:));
bandfill = @(ax,xx,lo,hi,c) fill(ax,[xx;flipud(xx)],[lo;flipud(hi)],c, ...
    'FaceAlpha',0.18,'EdgeColor','none','HandleVisibility','off');
ax = nexttile(tl); hold(ax,'on'); yyaxis(ax,'left');
bandfill(ax,a,ojl,ojh,col_o);
plot(ax,a,so,'-o','Color',col_o,'MarkerFaceColor',col_o,'MarkerSize',11,'LineWidth',2.6);
ylabel(ax,'\Omega_J','FontSize',FS_LAB); set(ax,'YColor',col_o,'YLim',[0.7 1.02]);
yyaxis(ax,'right');
bandfill(ax,a,del,deh,col_d);
plot(ax,a,sd,'-s','Color',col_d,'MarkerFaceColor',col_d,'MarkerSize',11,'LineWidth',2.6);
ylabel(ax,'D_{edge}','FontSize',FS_LAB); set(ax,'YColor',col_d);
xline(ax,1,'--','Color',[0.35 0.35 0.35],'LineWidth',2,'Label','applied bias', ...
      'LabelVerticalAlignment','bottom','FontSize',15);
xlabel(ax,'bias scaling \alpha','FontSize',FS_LAB);
title(ax,'(a) mechanism vs bias amplitude','FontSize',FS_TITLE,'FontWeight','normal'); clean(ax,FS_AX);
ax = nexttile(tl); hold(ax,'on');
% shade the MFPT-speedup band along the (Omega_J, S_MFPT) trajectory
fill(ax,[so;flipud(so)],[sml;flipud(smh)],[0.30 0.30 0.30], ...
     'FaceAlpha',0.15,'EdgeColor','none','HandleVisibility','off');
plot(ax,so,sm,'-o','Color',[0.30 0.30 0.30],'MarkerFaceColor',[0.30 0.30 0.30],'MarkerSize',11,'LineWidth',2.6);
k = find(abs(a-1)<1e-9);
errorbar(ax,so(k),sm(k),sm(k)-sml(k),smh(k)-sm(k),'k','LineStyle','none', ...
         'LineWidth',2.2,'CapSize',14,'HandleVisibility','off');
plot(ax,so(k),sm(k),'p','MarkerSize',30,'MarkerFaceColor',[0.95 0.75 0.10],'MarkerEdgeColor','k','LineWidth',1.5);
text(ax,so(k)-0.006,sm(k)*0.45,'applied bias','FontSize',16,'FontName','Arial', ...
     'HorizontalAlignment','left');
yline(ax,1,':','Color',[0.5 0.5 0.5],'LineWidth',1.8);
set(ax,'YScale','log','XDir','reverse');
xlabel(ax,'\Omega_J  (mechanism conservation \rightarrow less)','FontSize',FS_LAB);
ylabel(ax,'model MFPT speedup','FontSize',FS_LAB);
title(ax,'(b) acceleration-conservation tradeoff','FontSize',FS_TITLE,'FontWeight','normal'); clean(ax,FS_AX);
% (suptitle removed: it was redundant with the LaTeX caption and collided with
%  the (b) panel title.  Both facts -- b->alpha*b scaling and "model quantity of
%  the reconstructed MSM" -- are stated in the figure caption.)
sv(f4,'fig_ras_pareto_demo');

fprintf('Saved 4 Ras figures (.png/.pdf/.fig) to %s\n', FIGURES);

function clean(ax,fs)
    set(ax,'TickDir','out','Box','off','LineWidth',1.8,'FontSize',fs);
    grid(ax,'off');
end
function saveall(f,base)
    exportgraphics(f,[base '.png'],'Resolution',600,'BackgroundColor','white');
    try, exportgraphics(f,[base '.pdf'],'ContentType','vector','BackgroundColor','white');
    catch err, warning('pdf skipped: %s',err.message); end
    savefig(f,[base '.fig']);
end
