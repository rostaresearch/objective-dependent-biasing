function generate_all_figs()
% Regenerate every MATLAB figure and save an editable .fig alongside each, for
% manual font/layout tweaks. Best-effort: a script that needs absent data is
% caught and reported rather than aborting the run.
%
% Each script is executed inside run_one() so that a `clear` inside the script
% (several call `clear; close all`) is scoped to that helper and cannot wipe the
% driver's own counters.
%
% Output: rosta_jctc_v5/figures_editable/*.fig  +  MANIFEST.txt
% Run:    setenv('MSM_ROOT', 'C:\...\rosta_jctc_v5'); generate_all_figs

root = getenv('MSM_ROOT');
if isempty(root); root = fileparts(fileparts(mfilename('fullpath'))); end
codedir = fileparts(mfilename('fullpath'));   % where the plot scripts live
outdir  = fullfile(root, 'figures_editable');
if ~exist(outdir, 'dir'); mkdir(outdir); end

scripts = { ...
    'plot_grid_2d.m', 'plot_asymmetric_1d.m', 'plot_rkhs_2d.m', ...
    'plot_rkhs_5panel.m', 'plot_coordfree_bias.m', 'plot_mechanism.m', ...
    'plot_mechanism_v2.m', 'plot_fig2_us.m', 'plot_ras_pi_figs.m', ...
    'fig_alpha_sweep_matlab.m', 'fig_grid_feature_sweep_matlab.m', ...
    'fig_mechanism_ras_proper_matlab.m', 'fig_mechanism_symbols_inset.m'};

mfid = fopen(fullfile(outdir, 'MANIFEST.txt'), 'w');
fprintf(mfid, 'Editable .fig files (MATLAB) -- generated %s\n', datestr(now));
fprintf(mfid, 'Open in MATLAB with openfig(''name.fig'') to edit fonts/layout.\n\n');
nok = 0; nfail = 0; nfig = 0;

for s = 1:numel(scripts)
    script = scripts{s};
    if ~exist(fullfile(codedir, script), 'file')
        fprintf(mfid, '[skip] %s (not present)\n', script); continue;
    end
    base = erase(script, '.m');
    try
        names = run_one(codedir, root, script, outdir);
        if isempty(names)
            fprintf(mfid, '[warn] %s ran but left no open figure\n', script);
        else
            for k = 1:numel(names)
                fprintf(mfid, '[ ok ] %-42s <- %s\n', [names{k} '.fig'], script);
            end
            nfig = nfig + numel(names);
            nok = nok + 1;
        end
    catch ME
        nfail = nfail + 1;
        fprintf(mfid, '[FAIL] %-32s %s\n', script, ME.message);
        fprintf(2, 'FAIL %s: %s\n', script, ME.message);
    end
end

fprintf(mfid, '\nscripts ok=%d fail=%d ; .fig files written=%d\n', nok, nfail, nfig);
fprintf(mfid, '\nNote: matplotlib-generated figures (Fig 1 fig_landscape, Fig 3\n');
fprintf(mfid, 'fig_spectral_bias, fig2_notitle) have no .fig; their editable\n');
fprintf(mfid, 'vector source is the .pdf in figures/ (fonts editable in\n');
fprintf(mfid, 'Illustrator/Inkscape) plus the .py that builds them.\n');
fclose(mfid);
close all force;
fprintf('DONE: %d .fig written (%d scripts ok, %d failed). See %s\n', ...
        nfig, nok, nfail, fullfile(outdir, 'MANIFEST.txt'));
end


function names = run_one(codedir, root, script, outdir)
% Runs one plot script in this scoped workspace, then saves every figure it
% left open as a .fig. Several scripts call a bare `clear`, which wipes THIS
% function's workspace too -- so we stash what we need after the run in root
% appdata, which `clear` cannot touch, and restore it.
close all force;
setenv('MSM_ROOT', root);
base = erase(script, '.m');
setappdata(0, 'ga_outdir', outdir);
setappdata(0, 'ga_base', base);
run(fullfile(codedir, script));
outdir = getappdata(0, 'ga_outdir');    % restore in case the script cleared us
base   = getappdata(0, 'ga_base');
figs = flipud(findall(groot, 'Type', 'figure'));   % creation order
names = {};
for k = 1:numel(figs)
    if numel(figs) == 1
        nm = base;
    else
        nm = sprintf('%s_%d', base, k);
    end
    savefig(figs(k), fullfile(outdir, [nm '.fig']));
    names{end+1} = nm; %#ok<AGROW>
end
close all force;
end
