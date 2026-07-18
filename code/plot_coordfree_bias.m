% Coordinate-free biasing comparison WITH error bars (JCTC paper).
% Means +/- std over 12 independent multi-start optimisations on the 2D-grid
% network at U_max = 3 kBT (from eigenvector_bias_err.py).
% Natural pairings: v2 for the spectral gap, committor for the MFPT.

clear; close all;
PATH = getenv('MSM_ROOT');
if isempty(PATH), PATH = fileparts(fileparts(mfilename('fullpath'))); end
% --- gamma objective ---
lab_g  = {'xy-poly','v_2-poly','per-state'};
mean_g = [50.3 48.8 91.30];   std_g = [6.8 4.4 0];
% --- MFPT objective ---
lab_m  = {'xy-poly','committor','per-state'};
mean_m = [56.4 71.44 82.68];  std_m = [5.9 0 0];

% colours: gray=needs x,y ; blue/teal=K-only intrinsic ; orange=ceiling
col_g = [0.55 0.55 0.55; 0.10 0.45 0.75; 0.85 0.40 0.10];
col_m = [0.55 0.55 0.55; 0.10 0.62 0.55; 0.85 0.40 0.10];

set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultAxesFontSize',17);
set(groot,'defaultTextFontName','Arial');
set(groot,'defaultFigureColor','w');

fig = figure('Position',[100 100 1150 480]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

panels = {mean_g,std_g,col_g,lab_g,'(a) spectral-gap speedup  \gamma/\gamma_0'; ...
          mean_m,std_m,col_m,lab_m,'(b) MFPT speedup  \tau_0/\tau'};
for p = 1:2
    m=panels{p,1}; s=panels{p,2}; c=panels{p,3}; lab=panels{p,4};
    ax=nexttile; hold(ax,'on');
    for i=1:3
        bar(ax,i,m(i),0.6,'FaceColor',c(i,:),'EdgeColor','none');
    end
    errorbar(ax,1:3,m,s,'k','LineStyle','none','LineWidth',1.4,'CapSize',12);
    for i=1:3
        txt = (s(i)>0.05) * 1;  % flag
        if s(i)>0.05
            text(ax,i,m(i)+s(i)+3,sprintf('%.0f\\pm%.0f',m(i),s(i)), ...
                 'HorizontalAlignment','center','FontSize',14);
        else
            text(ax,i,m(i)+3,sprintf('%.0f',m(i)), ...
                 'HorizontalAlignment','center','FontSize',14);
        end
    end
    set(ax,'XTick',1:3,'XTickLabel',lab,'TickDir','out','Box','off','LineWidth',1.6,'FontSize',16);
    ax.XAxis.FontSize=15;
    ylim(ax,[0 max([mean_g+std_g mean_m+std_m])*1.12]);
    ylabel(ax,'speedup','FontSize',17);
    title(ax,panels{p,5},'FontWeight','normal','FontSize',17);
end

exportgraphics(gcf,fullfile(PATH,'fig_coordfree_bias.png'),'Resolution',600,'BackgroundColor','white');
try, exportgraphics(gcf,fullfile(PATH,'fig_coordfree_bias.pdf'),'ContentType','vector','BackgroundColor','white'); catch, end
fprintf('saved fig_coordfree_bias with error bars\n');
