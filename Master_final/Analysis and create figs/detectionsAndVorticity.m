clear;clc;
%% Importing the time series data for vorticity and horizontal divergence
load("avgVorticity_full_run2.mat");
load("horDivergence_full_run2.mat");
%load("avgW_velocity_full.mat");
%avgW_velocity = test.avgVorticity;

avgVorticity = avgVorticity / max(avgVorticity);
%beta_t = beta_t / max(beta_t);
%avgW_velocity = avgW_velocity / max(avgW_velocity);

%% Load and find ceiling detections as a function of time
ds = load('..\createFigures\data\ceilingTrackingResults_38-2346_Wthr50_s8_Run2.mat');

%%

lifetimeThreshold = 3.33;     % keep tracks ≥2 frames long
startIdx = 38;
endIdx   = 2346;
offsetStamp = 0;

% colors will come from the default colormap
tracks = ds.tracks;
times  = ds.times;
stamps = ds.stamps;
stamps = stamps - stamps(37) - 1/30;

% compute “good” tracks
allLifetimes = cellfun(@numel, {tracks.frames});
validTracks  = find(allLifetimes >= lifetimeThreshold);

% build per-frame count
numFrames     = numel(stamps);
filteredCount = zeros(1, numFrames);
%%
% for t = startIdx:endIdx
%     frameID = times(t);
%     disp(t)
%     filteredCount(t) = sum( arrayfun(@(k) any(tracks(k).frames==frameID), validTracks) );
% end

%%

% --- define spatial window ---
xmin = 160;   % lower‐x bound
xmax = 720;   % upper‐x bound
ymin = 0;   % lower‐y bound
ymax = 700;   % upper‐y bound

% pre‐allocate
filteredCount = zeros(1, numFrames);

for t = startIdx:endIdx
    frameID = times(t);
    cnt = 0;
    for k = validTracks
        % find if track k appears in this frame
        idxInTrack = find(tracks(k).frames == frameID, 1);
        if isempty(idxInTrack)
            continue
        end
        
        % get its centroid [x,y]
        xy = tracks(k).centroids(idxInTrack, :);
        xk = xy(1);
        yk = xy(2);
        
        % only count it if it falls in box
        if xk >= xmin && xk <= xmax && yk >= ymin && yk <= ymax
            cnt = cnt + 1;
        end
    end
    filteredCount(t) = cnt;
end

%% time axis
t_plot = stamps(startIdx:endIdx) - offsetStamp;

% Choose a window length (in frames) for the running average
windowSize = 10;   % e.g. 11‐frame (~instantaneous ±5 frames) smoothing

% Compute the moving average of filtered count
runAvg_ceil = movmean(filteredCount(startIdx:endIdx), windowSize, 'omitnan', 'Endpoints', 'shrink');
runAvg_ceil = runAvg_ceil';

% now plot as before
figure;
plot(t_plot, runAvg_ceil, '-', 'LineWidth',1.5);
xlabel('Time (s)');
ylabel('Detections in region');
title(sprintf('Hits in [x∈[%g,%g], y∈[%g,%g]]', xmin,xmax,ymin,ymax));

%% load and find profilometry detections as a function of time
prof = load('..\createFigures\data\profilometryTrackingResults_1-10800_Wthr-0.160_s8_Run2.mat');

%%

lifetimeThreshold_prof = 0;     % keep tracks ≥2 frames long
startIdx_prof = 1;
endIdx_prof   = 10766;
offsetStamp = 0;

% colors will come from the default colormap
tracks_prof = prof.tracks;
stamps_prof = prof.stamps;
stamps_prof = stamps_prof';

% compute “good” tracks
allLifetimes_prof = cellfun(@numel, {tracks_prof.frames});
validTracks_prof  = find(allLifetimes_prof >= lifetimeThreshold);

% %% build per-frame count
% numFrames_prof     = numel(stamps_prof);
% filteredCount_prof = zeros(1, numFrames_prof);
% for t = startIdx_prof:endIdx_prof
%     frameID = t;
%     if mod(t,100)==0, fprintf('Working on step %d / %d \n', t, numFrames_prof), end
%     filteredCount_prof(t) = sum( arrayfun(@(k) any(tracks_prof(k).frames==frameID), validTracks_prof) );
% end
% 
% % time axis
% t_plot_prof = stamps_prof(startIdx_prof:endIdx_prof) - offsetStamp;
% filteredCount_prof = filteredCount_prof';
% %
% save('profilometryCount_filtered.mat', 'filteredCount_prof', 't_plot_prof');
% disp('SAVED!')
%%
load("profilometryCount_filtered.mat");
%% Choose a window length (in frames) for the running average
windowSize = 15;   % e.g. 11‐frame (~instantaneous ±5 frames) smoothing

% Compute the moving average of filtered count
runAvg_prof = movmean(filteredCount_prof(startIdx_prof:endIdx_prof), windowSize, 'omitnan', 'Endpoints', 'shrink');

%
figure;
plot(t_plot_prof, runAvg_prof(startIdx_prof:endIdx_prof));

%% PLOTTING THE TIME SERIES OF ALL THREE DATASETS IN SAME PLOT

hfig = figure;
% left‐hand axis
yyaxis right
p1= plot(t_plot(1:299), runAvg_ceil(1:299), '-', 'LineWidth', 1.5, DisplayName='Reflections');
p1.Color = [0, 0.4470, 0.7410];
hold on;
p2 = plot(t_plot_prof(startIdx_prof:860), runAvg_prof(startIdx_prof:860), ':', ...
     'LineWidth', 1.5, DisplayName='Profilometry');
p2.Color = [0, 0.4470, 0.7410];
ylabel('Avg vorticity', Color='k');
set(gca,'YColor',[0, 0.4470, 0.7410]);
ylabel('Detections');

% right‐hand axis (just “activate” it and copy the limits/ticks)
yyaxis left
p3=plot(timeVec(14:299), beta_t(14:299), 'LineWidth', 1.5, DisplayName='Horizontal divergence');
hold on;
%plot(timeVec(1:299), avgVorticity(1:299));
p3.Color = 'k';%[0.900, 0.3250, 0.0980];
%yticks( yyaxis('left') ); % copy the y‐ticks
ylabel('$D_h(t)^2$');
set(gca,'YColor', 'k');%[0.900, 0.3250, 0.0980]);
xlabel('Time (s)');

% Set additional plotting properties
fname = 'output/detectionsAndDivergence';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.37; % Height-width ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
legend('Location','northwest', 'FontSize', 18);
set(findall(hfig,'-property','Box'),'Box','off'); % Optional box
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);
% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
box on;
% Export the figure
%print(hfig, fname, '-dpdf', '-vector');%, '-fillpage');
%print(hfig, fname, '-dpng', '-vector');

%%

% hfig = figure;
% % left‐hand axis
% yyaxis left
% plot(t_plot_prof(startIdx_prof:endIdx_prof), runAvg_prof(startIdx_prof:endIdx_prof), '-', ...
%      'LineWidth',1.5,'MarkerSize',6);
% ylabel('Detections');
% 
% % right‐hand axis (just “activate” it and copy the limits/ticks)
% yyaxis right
% plot(timeVec(14:346), avgVorticity(14:346), 'LineWidth', 1.5);
% %yticks( yyaxis('left') ); % copy the y‐ticks
% ylabel('Avg vorticity');
% title('Profilometry');
% xlabel('Time (s)');
% 
% % Set additional plotting properties
% fname = 'output/waveletThresholds_scale10_Wthr68';
% picturewidth = 20; % Set figure width in centimeters
% hw_ratio = 0.55; % Height-width ratio
% set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
% %legend('Location','southeast', 'FontSize', 18);
% set(findall(hfig,'-property','Box'),'Box','off'); % Optional box
% set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
% set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
% set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);
% % Configure printing options
% pos = get(hfig,'Position');
% set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
% box on;
% % Export the figure
% %print(hfig, fname, '-dpdf', '-vector');%, '-fillpage');
% %print(hfig, fname, '-dpng', '-vector');


%% Correlation analysis

% % reshape the data into a column
% d_good = d_good(:);  % now 1000×1
% 
% % sanity check
% assert( isequal(size(t_good), size(d_good)), ...
%         't and d must have identical dimensions' );
% 
% % build the timetable
% TT_ceil = timetable(t_good, d_good, 'VariableNames', {'ceil'});

%% Calculate the running average of the vorticity as well
windowsize = 5;
runAvg_div = movmean(beta_t, windowSize, 'omitnan', 'Endpoints', 'shrink');


%% 1) Build timetables & synchronize
% assume t_plot and timeVec are both in seconds
TT_ceil = timetable(seconds(t_plot), runAvg_ceil, 'VariableNames', {'ceil'});
TT_prof = timetable(seconds(t_plot_prof), runAvg_prof, 'VariableNames', {'prof'});
TT_vort = timetable(seconds(timeVec(14:end)), avgVorticity(14:end), 'VariableNames', {'vort'});
TT_vort2 = timetable(seconds(timeVec(14:end)), avgVorticity(14:end), 'VariableNames', {'vort'});
TT_div = timetable(seconds(timeVec(14:end)), runAvg_div(14:end), 'VariableNames', {'div'});

%% picking a regular grid
t0_ceil = max([TT_ceil.Time(1), TT_vort.Time(1)]);
t1_ceil = min([TT_ceil.Time(end), TT_vort.Time(end)]);
dt_ceil = seconds(1/15);                % choose 1 s, or .5 s, etc.
newTime_ceil = (t0_ceil:dt_ceil:t1_ceil)';

t0_prof = max([TT_prof.Time(1), TT_vort.Time(1)]);
t1_prof = min([TT_prof.Time(end), TT_vort2.Time(end)]);
dt_prof = seconds(1/15);                % choose 1 s, or .5 s, etc.
newTime_prof = (t0_prof:dt_prof:t1_prof)';

t0_ceil_div = max([TT_ceil.Time(1), TT_div.Time(1)]);
t1_ceil_div = min([TT_ceil.Time(end), TT_div.Time(end)]);
dt_ceil_div = seconds(1/15);                % choose 1 s, or .5 s, etc.
newTime_ceil_div = (t0_ceil_div:dt_ceil_div:t1_ceil_div)';

t0_prof_div = max([TT_prof.Time(1), TT_div.Time(1)]);
t1_prof_div = min([TT_prof.Time(end), TT_div.Time(end)]);
dt_prof_div = seconds(1/15);                % choose 1 s, or .5 s, etc.
newTime_prof_div = (t0_prof_div:dt_prof_div:t1_prof_div)';

%% synchronize + linear‐interpolate missing
TT_ceil = synchronize(TT_ceil, TT_vort, newTime_ceil, 'linear');
TT_ceil_div = synchronize(TT_ceil, TT_div, newTime_ceil_div, 'linear');
TT_prof = synchronize(TT_prof, TT_vort2, newTime_prof, 'linear');
TT_prof_div = synchronize(TT_prof, TT_div, newTime_prof_div, 'linear');

% TT_all = synchronize(...
%     TT_ceil, TT_prof, TT_vort, newTime, ...    % newTime from code above
%     'linear');                       % linear interp onto the same grid
% % drop any rows where *any* of the three is NaN
% TT_all = rmmissing(TT_all);

%% now TT.ceil and TT.vort are the two aligned series
% any remaining NaNs?  e.g. at the edges—just trim them:
nanIdx = isnan(TT_ceil.ceil) | isnan(TT_ceil.vort);
TT_ceil(nanIdx,:) = [];
TT_ceil_div(nanIdx,:) = [];

%% 2) Detrend (or remove a running mean)
% % Option A: simple linear detrend
% y1 = detrend(TT.prof);
% y2 = detrend(TT.vort);
% 
% %% 3) Stationarity test
% % requires Statistics Toolbox
% [h1,p1] = adftest(y1);
% [h2,p2] = adftest(y2);
% fprintf('ADF ceil: h=%d, p=%.3f\nADF vort: h=%d, p=%.3f\n', h1,p1, h2,p2);
% % if h==0 (fail to reject), the series is non‐stationary → consider differencing
% 
% %% 4) First‐difference to enforce stationarity (if needed)
% dy1 = diff(y1);
% dy2 = diff(y2);
% N   = length(dy1);

disp(TT_ceil.vort(1:10));

%% 2) Detrend each column
y_ceil = detrend(TT_ceil.ceil);
y_prof = detrend(TT_prof.prof);
y_vort_ceil = detrend(TT_ceil.vort);
y_vort_prof = detrend(TT_prof.vort);
y_div_ceil = detrend(TT_ceil_div.div);
y_div_prof = detrend(TT_prof_div.div);

% 3) Difference to enforce stationarity
dy_ceil = diff(y_ceil);
dy_prof = diff(y_prof);
dy_vort_ceil = diff(y_vort_ceil);
dy_vort_prof = diff(y_vort_prof);
dy_div_ceil = diff(y_div_ceil);
dy_div_prof = diff(y_div_prof);

% 4) Compute normalized cross‐correlations
maxLag = 100;  % choose ±lag window
[xc_c, lags_c] = xcorr(dy_ceil, dy_vort_ceil, maxLag, 'coeff');
[xc_p, lags_p] = xcorr( dy_prof,dy_vort_prof, maxLag, 'coeff');
[xc_c_div, lags_c_div] = xcorr( dy_ceil,dy_div_ceil, maxLag, 'coeff');
[xc_p_div, lags_p_div] = xcorr(dy_prof, dy_div_prof, maxLag, 'coeff');

% 5) Plot them together with significance bounds
figure;
plot(lags_c*seconds(dt_ceil), xc_c, '-','LineWidth',1.5,'DisplayName','ceil & vort'); 
hold on;
plot(lags_c_div*seconds(dt_ceil), xc_c_div, '-','LineWidth',1.5,'DisplayName','ceil & div'); 
plot(lags_p*seconds(dt_prof), xc_p, '-','LineWidth',1.5,'DisplayName','prof & vort');
plot(lags_p_div*seconds(dt_prof), xc_p_div, '-','LineWidth',1.5,'DisplayName','prof & div');
legend;

%% 5) Cross‐correlation
maxLag = 100;  % e.g. ±100 samples
[xc, lags] = xcorr(dy2, dy1, maxLag, 'coeff');  
% here: positive lag means vort(t) leads ceil(t+lag)

%% 6) Plot with significance bounds
figure; 
stem(lags*seconds(dt), xc, '-');
xlabel('Lag (s)  [vorticity → detections]');
ylabel('Cross‐corr coeff');
title('Cross‐correlation of Δvorticity vs. Δceil');

% % 95% confidence bounds for white noise ≈ ±1.96/√N
% ci = 1.96/sqrt(N);
% hold on;
% plot(lags*seconds(dt),  ci*ones(size(lags)), '--k');
% plot(lags*seconds(dt), -ci*ones(size(lags)), '--k');
% hold off;

%%
figure;
plot(lags*seconds(dt), xc, '-');
xlabel('Lag (s)  [vorticity → detections]');
ylabel('Cross‐corr coeff');
title('Cross‐correlation of Δvorticity vs. Δceil');

%% Plot the cross-correlations together

hfig = figure; 
plot(lags_c_div*seconds(dt_ceil)/1.05, xc_c_div, 'k-','LineWidth',1.5,'DisplayName','Reflections'); 
hold on;
plot(lags_p_div*seconds(dt_prof)/1.05, xc_p_div, '-','LineWidth',1.5,'DisplayName','Profilometry', 'Color', [0, 0.4470, 0.8410]);
xlabel('Time lag / $T_\infty$');
ylabel('Normalised cross-correlation');
%title('Cross‐correlation of Δvorticity vs. Δceil');
hold on;

% 1) find the peak correlation for ceiling
[peakVal_c, idxPeak_c] = max(xc_c_div);  
lagPeak_c = lags_c(idxPeak_c) * seconds(dt_ceil);

%find the peak correlation for profilometry
[peakVal_p, idxPeak_p] = max(xc_p_div);  
lagPeak_p = lags_p(idxPeak_p) * seconds(dt_prof);

% 2) draw a dashed vertical line at that lag
yl = ylim;                           % current y‐limits of the axes
xline(lagPeak_c, '--k', ... 
     'LineWidth',1.5, 'HandleVisibility','off');
xline(lagPeak_p, '--', ... 
     'LineWidth',1.5, 'HandleVisibility','off', 'Color', [0, 0.4470, 0.8410]);

% % 3) optionally annotate the numeric value
% text(lagPeak_c, peakVal_c, ...
%      sprintf('  Peak @ %.2fs\n  r=%.2f', lagPeak_c, peakVal_c), ...
%      'VerticalAlignment','bottom', ...
%      'HorizontalAlignment','left', ...
%      'BackgroundColor','w', ...
%      'EdgeColor','k');
% % 3) optionally annotate the numeric value
% text(lagPeak_p, peakVal_p, ...
%      sprintf('  Peak @ %.2fs\n  r=%.2f', lagPeak_p, peakVal_p), ...
%      'VerticalAlignment','bottom', ...
%      'HorizontalAlignment','left', ...
%      'BackgroundColor','w', ...
%      'EdgeColor','k');



hold off;

xlim([-6.5, 6.5]);
ylim([-0.25, 0.6]);

% Set additional plotting properties
fname = 'output/correlationBothDatasets_shortWindow_DIVERGENCE_runAvg';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
legend('Location','northeast', 'FontSize', 18);
set(findall(hfig,'-property','Box'),'Box','off'); % Optional box
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);
% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
box on;
% Export the figure
%print(hfig, fname, '-dpdf', '-vector');%, '-fillpage');
%print(hfig, fname, '-dpng', '-vector');


%%

N = length(dy1);              % number of paired samples
r = peakVal;
tstat = r*sqrt((N-2)/(1-r^2));
pval  = 2*(1 - tcdf(abs(tstat), N-2));
fprintf('r=%.2f, t(%d)=%.2f, p=%.3f\n', r, N-2, tstat, pval);

%%

% Assuming already run synchronization & detrending script, so:
%   TT_ceil, TT_ceil_div, TT_prof, TT_prof_div
%   xc_c_div, lags_c_div, xc_p_div, lags_p_div
%   dt_ceil, dt_prof
% And T_inf = 1.05 (to normalize time axes)
T_inf = 1.05;

% 1) Find the “best lag” (in seconds) for each dataset
[~, idxPeak_c_div] = max(xc_c_div);
bestLag_c = lags_c_div(idxPeak_c_div) * seconds(dt_ceil) / T_inf;
disp(bestLag_c)
[~, idxPeak_p_div] = max(xc_p_div);
bestLag_p = lags_p_div(idxPeak_p_div) * seconds(dt_prof)  / T_inf;
disp(bestLag_p)
% 2) Build time vectors (normalized by T_inf)
t_ceil = seconds(TT_ceil.Time) / T_inf;
t_div_ceil = seconds(TT_ceil_div.Time) / T_inf;
t_prof = seconds(TT_prof.Time) / T_inf;
t_div_prof = seconds(TT_prof_div.Time) / T_inf;

% 3) Plot
figure;

% --- Ceiling panel ---
subplot(2,1,1);
plot(t_ceil,           TT_ceil.ceil / max(TT_ceil.ceil), 'k-',  'LineWidth',1.5); hold on;
plot(t_div_ceil,       TT_ceil_div.div / max(TT_ceil_div.div), 'r-',  'LineWidth',1.5);
plot(t_div_ceil + bestLag_c, TT_ceil_div.div / max(TT_ceil_div.div), 'k--', 'LineWidth',1.5);
xlabel('Time / $T_\infty$','Interpreter','latex');
ylabel('Ceil & Divergence','Interpreter','latex');
title('Ceiling vs. Divergence (with Shift)','Interpreter','latex');
legend({'Ceiling','Divergence','Ceiling shifted'}, ...
       'Location','best','Interpreter','latex');
grid on;

% --- Profilometry panel ---
subplot(2,1,2);
plot(t_prof,          TT_prof.prof, 'b-',  'LineWidth',1.5); hold on;
plot(t_div_prof,      TT_prof_div.div, 'm-',  'LineWidth',1.5);
plot(t_prof + bestLag_p, TT_prof.prof, 'b--', 'LineWidth',1.5);
xlabel('Time / $T_\infty$','Interpreter','latex');
ylabel('Prof & Divergence','Interpreter','latex');
title('Profilometry vs. Divergence (with Shift)','Interpreter','latex');
legend({'Profilometry','Divergence','Profilometry shifted'}, ...
       'Location','best','Interpreter','latex');
grid on;

% 4) Tidy up overall figure size, fonts, etc.
set(gcf,'Units','centimeters','Position',[2 2 20 12]);
set(findall(gcf,'-property','FontSize'),'FontSize',16);
