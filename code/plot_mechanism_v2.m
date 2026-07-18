% Alternative "cleanest final" mechanism figure (JCTC main text).
% Keeps the originals; this is fig_mechanism_main_v2.
% Layout (2x3):
%  (a) MFPT speedup vs Omega_J, with alpha labels/arrows
%  (b) Omega_J & Omega_rho vs alpha
%  (c) D_edge (current-weighted edge distortion) vs alpha
%  (d,e,f) reactive-current maps: unbiased / MFPT-opt / gamma-opt
%  (the former (d) summary table is now a LaTeX table in the manuscript)
clear; close all;
PATH = getenv('MSM_ROOT');
if isempty(PATH), PATH = fileparts(fileparts(mfilename('fullpath'))); end
load(fullfile(PATH,'mechanism_data.mat'));

set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultAxesFontSize',15);
set(groot,'defaultTextFontName','Arial');
set(groot,'defaultLineLineWidth',2.2);
set(groot,'defaultFigureColor','w');
cmf=[0.85 0.30 0.10]; cga=[0.10 0.45 0.75];
clean=@(ax) set(ax,'TickDir','out','Box','off','LineWidth',1.5,'FontSize',15);

fig=figure('Position',[40 40 1450 940]);
tl=tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

% ---------- (a) Pareto with alpha labels ----------
ax=nexttile(1); hold(ax,'on');
plot(ax,m_OmegaJ,m_mfpt_su,'-o','Color',cmf,'MarkerFaceColor',cmf,'MarkerSize',5);
plot(ax,g_OmegaJ,g_mfpt_su,'-s','Color',cga,'MarkerFaceColor',cga,'MarkerSize',5);
set(ax,'YScale','log');
% alpha anchor labels
text(ax,1.0,1.0,'  \alpha=0','FontSize',12,'Color',[0.3 0.3 0.3],'VerticalAlignment','top');
text(ax,m_OmegaJ(end),m_mfpt_su(end),'\alpha=1 ','Color',cmf,'FontSize',12,'HorizontalAlignment','right','FontWeight','bold');
[~,ipk]=max(g_mfpt_su);
text(ax,g_OmegaJ(ipk),g_mfpt_su(ipk),'  \alpha\approx0.4','Color',cga,'FontSize',12,'VerticalAlignment','bottom');
text(ax,g_OmegaJ(end),g_mfpt_su(end),'\alpha=1 ','Color',cga,'FontSize',12,'HorizontalAlignment','right','VerticalAlignment','top','FontWeight','bold');
text(ax,0.80,3e4,'increasing \alpha \rightarrow','FontSize',11,'Color',[0.3 0.3 0.3],'Rotation',-32);
xlim(ax,[0.62 1.03]); ylim(ax,[0.9 2e5]);
clean(ax);
xlabel(ax,'reactive-current overlap  \Omega_J');
ylabel(ax,'MFPT speedup  \tau_0/\tau');
legend(ax,{'MFPT-opt bias','\gamma-opt bias'},'Location','southwest','Box','off','FontSize',12);
title(ax,'(a) acceleration vs mechanism overlap','FontWeight','normal','FontSize',15);

% ---------- (b) overlaps vs alpha ----------
ax=nexttile(2); hold(ax,'on');
plot(ax,alphas,m_OmegaJ,'-o','Color',cmf,'MarkerFaceColor',cmf,'MarkerSize',5);
plot(ax,alphas,m_Omegarho,'--o','Color',cmf,'MarkerSize',5);
plot(ax,alphas,g_OmegaJ,'-s','Color',cga,'MarkerFaceColor',cga,'MarkerSize',5);
plot(ax,alphas,g_Omegarho,'--s','Color',cga,'MarkerSize',5);
ylim(ax,[0 1.05]); clean(ax);
xlabel(ax,'bias scaling  \alpha'); ylabel(ax,'overlap');
legend(ax,{'MFPT \Omega_J','MFPT \Omega_\rho','\gamma \Omega_J','\gamma \Omega_\rho'},'Location','southwest','Box','off','FontSize',11);
title(ax,'(b) current & tube overlap','FontWeight','normal','FontSize',15);

% ---------- (c) D_edge vs alpha ----------
ax=nexttile(3); hold(ax,'on');
plot(ax,alphas,m_Dedge,'-o','Color',cmf,'MarkerFaceColor',cmf,'MarkerSize',5);
plot(ax,alphas,g_Dedge,'-s','Color',cga,'MarkerFaceColor',cga,'MarkerSize',5);
plot(ax,alphas,m_Dedge95,'--o','Color',cmf,'MarkerSize',4);
plot(ax,alphas,g_Dedge95,'--s','Color',cga,'MarkerSize',4);
clean(ax);
xlabel(ax,'bias scaling  \alpha');
ylabel(ax,'edge distortion  D_{edge}');
legend(ax,{'MFPT mean','\gamma mean','MFPT 95%','\gamma 95%'},'Location','northwest','Box','off','FontSize',11);
title(ax,'(c) edge-rate distortion','FontWeight','normal','FontSize',15);

% ---------- (e,f,g) current maps ----------
flds={fx0,fy0,'(d) unbiased  K^{(0)}',4; fxm,fym,'(e) MFPT-opt  K^{(b_{MFPT})}',5; fxg,fyg,'(f) \gamma-opt  K^{(b_\gamma)}',6};
allmag=sqrt([fx0;fxm;fxg].^2+[fy0;fym;fyg].^2); amax=quantile(allmag(allmag>0),0.98);
for p=1:3
    ax=nexttile(flds{p,4}); hold(ax,'on');
    contourf(ax,xs,ys,F',18,'LineColor','none'); colormap(ax,flipud(gray(64)));
    fx=flds{p,1}; fy=flds{p,2}; mag=sqrt(fx.^2+fy.^2); sel=mag>0.012*amax;
    quiver(ax,cx(sel),cy(sel),fx(sel),fy(sel),1.7,'Color',[0.85 0.10 0.10],'LineWidth',1.1,'MaxHeadSize',0.55);
    plot(ax,start_xy(1),start_xy(2),'o','MarkerSize',12,'MarkerFaceColor',[0.10 0.45 0.85],'MarkerEdgeColor','w','LineWidth',1.5);
    plot(ax,end_xy(1),end_xy(2),'s','MarkerSize',13,'MarkerFaceColor',[0.10 0.70 0.30],'MarkerEdgeColor','w','LineWidth',1.5);
    xlim(ax,[0 2*pi]); ylim(ax,[0 2*pi]); axis(ax,'square'); clean(ax);
    xlabel(ax,'x'); if p==1, ylabel(ax,'y'); end
    title(ax,flds{p,3},'FontWeight','normal','FontSize',15);
end

exportgraphics(gcf,fullfile(PATH,'fig_mechanism_main_v2.png'),'Resolution',500,'BackgroundColor','white');
try, exportgraphics(gcf,fullfile(PATH,'fig_mechanism_main_v2.pdf'),'ContentType','vector','BackgroundColor','white'); catch, end
fprintf('saved fig_mechanism_main_v2\n');
