clear;clc;

%% load results
% load each file into a separate struct
data30 = load('data\ceilingTrackingResults_38-2346_Wthr26_s5_Run2.mat');
data45 = load('data\ceilingTrackingResults_38-2346_Wthr50_s8_Run2.mat');
data70 = load('data\ceilingTrackingResults_38-2346_Wthr61_s10_Run2.mat');

%% — Plot structures as a function of time — 

% % 1) lifetime threshold (in # of frames)
% lifetimeThreshold = 2;    
% 
% % 2) compute each track’s lifetime
% allLifetimes = cellfun(@numel, {tracks.frames});
% 
% % 3) find the “good” tracks
% validTracks = find(allLifetimes >= lifetimeThreshold);
% 
% % 4) preallocate an array as long as video
% numFrames     = numel(stamps);     % or: numFrames = size(video,3);
% filteredCount = zeros(1, numFrames);
% 
% % define frame‐range of interest
% startIdx = 38;
% endIdx   = 338;
% 
% % compute the same offset used in the loop
% offsetStamp = 0;%stamps(startIdx-1) + 1/30;
% 
% % 5) fill in per-frame counts
% for t_index = startIdx:endIdx
%     % this is the “frame number” stored in tracks.frames
%     frameID = times(t_index);
% 
%     % count how many valid tracks include this frameID
%     filteredCount(t_index) = sum( ...
%         arrayfun(@(k) any(tracks(k).frames == frameID), validTracks) );
% end
% 
% % build the time axis exactly as inside the loop
% t_plot = stamps(startIdx:endIdx) - offsetStamp;
% 
% 
% % instantaneous (per-frame) count
% figure;
% plot( t_plot, filteredCount(startIdx:endIdx), 'o-','LineWidth',1.5,'MarkerSize',6 );
% xlabel('Time (s)');
% ylabel(['# Tracks (lifetime \ge ' num2str(lifetimeThreshold) ' frames)']);
% title('Filtered Tracks vs. Time');
% grid on;
% 
% % cumulative count
% cumFilt = cumsum( filteredCount(startIdx:endIdx) );
% figure;
% plot( t_plot, cumFilt, 's-','LineWidth',1.5,'MarkerSize',6 );
% xlabel('Time (s)');
% ylabel('Cumulative # Tracks');
% title('Cumulative Filtered Tracks');
% grid on;

%%


datasets = { data30, data45, data70 };
labels   = {'$W_{\rm thr} = 26$','$W_{\rm thr} = 50$','$W_{\rm thr} = 61$'};
lifetimeThreshold = 5;     % keep tracks ≥2 frames long
startIdx = 38;
endIdx   = 338;
offsetStamp = 0; 

% colors will come from the default colormap
hfig = figure; hold on;
for i = 1:numel(datasets)
    ds = datasets{i};
    tracks = ds.tracks;
    times  = ds.times;
    stamps = ds.stamps;

    % compute “good” tracks
    allLifetimes = cellfun(@numel, {tracks.frames});
    validTracks  = find(allLifetimes >= lifetimeThreshold);

    % build per-frame count
    numFrames     = numel(stamps);
    filteredCount = zeros(1, numFrames);
    for t = startIdx:endIdx
        frameID = times(t);
        filteredCount(t) = sum( arrayfun(@(k) any(tracks(k).frames==frameID), validTracks) );
    end

    % time axis
    t_plot = stamps(startIdx:endIdx) - offsetStamp;

    % plot instantaneous
    plot( t_plot, filteredCount(startIdx:endIdx), '-', ...
          'LineWidth',1.5,'MarkerSize',6);
end
xlabel('Time (s)');
ylabel(sprintf('# Tracks (lifetime \\ge %d frames)', lifetimeThreshold));
title('Filtered Tracks per Frame Comparison');
legend(labels,'Location','best');
grid on;
hold off;

% Set additional plotting properties
fname = 'output/waveletThresholds_scale10_Wthr68';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
%legend('Location','southeast', 'FontSize', 18);
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


%% — now cumulative on one figure —
hfig = figure; hold on;
for i = 1:numel(datasets)
    ds = datasets{i};
    % recompute filteredCount as above (or cache it)
    tracks = ds.tracks;
    times  = ds.times;
    stamps = ds.detectionTime;
    allLifetimes = cellfun(@numel, {tracks.frames});
    validTracks  = find(allLifetimes >= lifetimeThreshold);
    filteredCount = zeros(1, numel(stamps));
    for t = startIdx:endIdx
        frameID = times(t);
        filteredCount(t) = sum( arrayfun(@(k) any(tracks(k).frames==frameID), validTracks) );
    end
    t_plot = stamps(startIdx:endIdx) - offsetStamp;
    
    % plot cumulative
    cumFilt = cumsum( filteredCount(startIdx:endIdx) );
    plot( t_plot, cumFilt, 'LineWidth',1.5,'MarkerSize',6 );
end
xlabel('Time (s)');
ylabel('Cumulative # Tracks');
title('Cumulative Filtered Tracks Comparison');
legend(labels,'Location','best');
grid on;
hold off;

% Set additional plotting properties
fname = 'output/waveletThresholds_scale10_Wthr68';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
%legend('Location','southeast', 'FontSize', 18);
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

%% Subplots

% — assume data30, data45, data70 and labels already in workspace —

datasets = { data30, data45, data70 };
labels   = {'$W_{\rm thr} = 26$, $s=5$','$W_{\rm thr} = 50$, $s=8$','$W_{\rm thr} = 61$, $s=10$'};
linestyles = {'--', '-', ':'};   % solid, dashed, dotted
colors          = {'#0072BD',    '#0072BD',    '#0072BD'};   % hex colors
lifetimeThreshold = 5;     
startIdx = 38;
endIdx   = 338;
offsetStamp = 0;           

% Create figure
hfig = figure; 
tiledlayout(2,1, "TileSpacing","tight", "Padding","tight")
% — First subplot: instantaneous counts —
ax1=nexttile; hold on;
for i = 1:numel(datasets)
    ds = datasets{i};
    tracks = ds.tracks;
    times  = ds.times;
    stamps = ds.detectionTime;

    % find valid tracks
    allLifetimes = cellfun(@numel, {tracks.frames});
    validTracks  = find(allLifetimes >= lifetimeThreshold);

    % per-frame count
    filteredCount = zeros(1, numel(stamps));
    for t = startIdx:endIdx
        frameID = times(t);
        filteredCount(t) = sum( arrayfun(@(k) any(tracks(k).frames==frameID), validTracks) );
    end

    t_plot = stamps(startIdx:endIdx) - offsetStamp;
    plot(t_plot, filteredCount(startIdx:endIdx), linestyles{i}, ...
         'LineWidth',1.5,'MarkerSize',6, 'Color',colors{i});
end
%xlabel('Time (s)');
ylabel('Detections');
%title('Instantaneous Filtered Tracks');
legend(labels,'Location','northwest','Interpreter','latex');
grid on;
box(ax1,'on');
hold off;
%set(gca, 'XTick', []);
xlim([0.9,20])
ylim([0,25])

% — Second subplot: cumulative counts —
ax2=nexttile; hold on;
for i = 1:numel(datasets)
    ds = datasets{i};
    tracks = ds.tracks;
    times  = ds.times;
    stamps = ds.detectionTime;

    allLifetimes = cellfun(@numel, {tracks.frames});
    validTracks  = find(allLifetimes >= lifetimeThreshold);

    filteredCount = zeros(1, numel(stamps));
    for t = startIdx:endIdx
        frameID = times(t);
        filteredCount(t) = sum( arrayfun(@(k) any(tracks(k).frames==frameID), validTracks) );
    end

    t_plot = stamps(startIdx:endIdx) - offsetStamp;
    cumFilt = cumsum(filteredCount(startIdx:endIdx));
    plot(t_plot, cumFilt, linestyles{i}, 'LineWidth',1.5,'MarkerSize',6, 'Color',colors{i});
end

xlabel('Time (s)');
ylabel('Cumulative detections');
%title('Cumulative Filtered Tracks');
%legend(labels,'Location','best','Interpreter','latex');
grid on;
box(ax2,'on');
hold off;
xlim([0.9,20]);

% Set additional plotting properties
fname = 'output/compareScales_NEW';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.75; % Height-width ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
%legend('Location','southeast', 'FontSize', 18);
set(findall(hfig,'-property','Box'),'Box','on'); % Optional box
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);
% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
%box on;
% Export the figure
%print(hfig, fname, '-dpdf', '-vector');%, '-fillpage');
%print(hfig, fname, '-dpng', '-vector');

%% plot number of detection as a function of lifetimeThreshold
% define the range of thresholds to test
thresholds       = 1:15;                
nT               = numel(thresholds);
totalDetections  = zeros(1,nT);         % <-- 1×nT, not nT×nT
startIdx         = 1;
endIdx           = 7349;

% load dataset
%data    = load('data\ceilingTrackingResults_38-2346_Wthr40_s8_Run2.mat');
data = load('..\createFigures\data\profilometryTrackingResults_1-10800_Wthr-0.200_s8_Run1.mat');
tracks  = data.tracks;
%times   = data.times;
stamps  = data.stamps;
nFrames = numel(stamps);

%% precompute each track’s lifetime only once
allLifetimes = cellfun(@numel, {tracks.frames});

for k = 1:nT
    thr = thresholds(k);
    disp(thr)
    % find valid tracks
    validTracks = find(allLifetimes >= thr);
    
    % per-frame count
    filteredCount = zeros(1,nFrames);
    for t = startIdx:endIdx
        if mod(t,100)==0, fprintf('Working on step %d / %d. Lifetime threshold: %d \n', t, nFrames, k), end
        fID = (t);
        filteredCount(t) = sum( arrayfun(@(kk) any(tracks(kk).frames==fID), validTracks) );
    end
    
    % total detections in window
    totalDetections(k) = sum( filteredCount(startIdx:endIdx) );
end

%% plot
hfig = figure;
plot(thresholds/(29.97*1.05), totalDetections, 'ko-','LineWidth',1.5,'MarkerSize',6)
xlabel('$t_{min} / T_\infty$')
ylabel('Total detections')
%title('Total # Detections vs. Lifetime Threshold','Interpreter','latex')
grid on
%xlim([0,15])

% Set additional plotting properties
fname = 'output/detectionsAsFuncOfLifetimeThreshold';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
%legend('Location','southeast', 'FontSize', 18);
set(findall(hfig,'-property','Box'),'Box','on'); % Optional box
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);
% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
%box on;
% Export the figure
%print(hfig, fname, '-dpdf', '-vector');%, '-fillpage');
%print(hfig, fname, '-dpng', '-vector');

%%

%--- precompute per‐track stats
allLifetimes     = cellfun(@numel, {tracks.frames});
trackWindowCount = cellfun( ...
    @(f) sum(f>= startIdx & f<= endIdx), ...
    {tracks.frames} );

%--- vectorized threshold sweep
totalDetections = arrayfun(@(thr) ...
    sum( trackWindowCount(allLifetimes >= thr) ), ...
    thresholds);

%--- (optional) plot
plot(thresholds, totalDetections, 'o-')
xlabel('Lifetime threshold')
ylabel('Total detections')

%%
%--- compute once
sumAll = sum(trackWindowCount);    % total detections if thr=1

%--- thresholds
thresholds      = 1:15;
totalDetections = zeros(size(thresholds));

% assign the trivial case by hand
totalDetections(1) = sumAll;

% then only loop (or arrayfun) for thr = 2:15
for k = 2:numel(thresholds)
    thr = thresholds(k);
    totalDetections(k) = sum( trackWindowCount(allLifetimes >= thr) );
end

%--- (optional) plot
plot(thresholds, totalDetections, 'o-')
xlabel('Lifetime threshold')
ylabel('Total detections')

%% --- load data and precompute lifetimes

data1         = load('data\ceilingTrackingResults_38-2346_Wthr50_s8_Run2.mat');
data2 = load('..\createFigures\data\profilometryTrackingResults_1-10800_Wthr-0.200_s8_Run1.mat');

tracks1       = data1.tracks;
allLifetimes = cellfun(@numel, {tracks1.frames});

%--- define thresholds and vectorize
thresholds_ceiling   = 1:30 * (29.97*1.05);                                    % minimum lifetime in frames
%
nTracksByThr1 = arrayfun(@(thr) ...
                     sum(allLifetimes >= thr), ...
                     thresholds);

tracks2       = data2.tracks;
allLifetimes = cellfun(@numel, {tracks2.frames});

                                    % minimum lifetime in frames
nTracksByThr2 = arrayfun(@(thr) ...
                     sum(allLifetimes >= thr), ...
                     thresholds);


%--- plot
hfig = figure;
% Store the colors by plotting and getting the handles
h1 = plot(thresholds/(45*1.05), nTracksByThr2, 'o-', 'LineWidth', 1.5, 'DisplayName', 'Profilometry');
hold on
h2 = plot(thresholds(1:12)/(29.97*1.05), nTracksByThr1(1:12), 'ko-', 'LineWidth', 1.5, 'DisplayName', 'Ceiling');

% Save the colors to reorder
color1 = h1.Color;
color2 = h2.Color;

% Clear and replot in new order with preserved colors
clf
% First plot: Ceiling (originally second)
plot(thresholds(1:12)/(29.97*1.05), nTracksByThr1(1:12), 'o-', ...
    'LineWidth', 1.5, 'DisplayName', 'Ceiling', 'Color', color2);
hold on
xline(0.104, '-.', 'HandleVisibility','off');
% Second plot: Profilometry (originally first)
plot(thresholds/(45*1.05), nTracksByThr2, 'o-', ...
   'LineWidth', 1.5, 'DisplayName', 'Profilometry', 'Color', color1);

xlabel('$t_{\rm min} / T_\infty$')
ylabel('Total tracks')
grid on
xlim([0,0.4])
ylim([-500, 1E4])

% Set additional plotting properties
fname = 'output/tracksAsFuncOfLifetimeThreshold_ceiling';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
set(findall(hfig,'-property','Box'),'Box','on'); % Optional box
legend('Location','northeast', 'FontSize', 18, Box='off');
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);
% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);
%box on;
% Export the figure
%print(hfig, fname, '-dpdf', '-vector');%, '-fillpage');
%print(hfig, fname, '-dpng', '-vector');

%%

% --- load and extract lifetimes as before
data1 = load('data\ceilingTrackingResults_38-2346_Wthr50_s8_Run2.mat');
tracks1 = data1.tracks;
lifetimes1 = cellfun(@numel, {tracks1.frames});

data2 = load('..\createFigures\data\profilometryTrackingResults_1-10800_Wthr-0.200_s8_Run1.mat');
tracks2 = data2.tracks;
lifetimes2 = cellfun(@numel, {tracks2.frames});

% --- define raw‐frame thresholds (as before)
%    here we take 1:30×(frameRate×1.05)
thrFrames1 = 1 : floor(30*(29.97*1.05));
thrFrames2 = 1 : floor(30*(45   *1.05));

% --- now compute TOTAL DETECTIONS (i.e. sum of lifetimes) but only
%    for those tracks whose lifetime ≥ thr
nDetsByThr1 = arrayfun(@(t) sum(lifetimes1(lifetimes1>=t)), thrFrames1);
nDetsByThr2 = arrayfun(@(t) sum(lifetimes2(lifetimes2>=t)), thrFrames2);

% --- plot, normalizing the x‐axis
h = figure; hold on;
h1 = plot(thrFrames2/(45*1.05), nDetsByThr2, 'o-', 'LineWidth',1.5, 'DisplayName','Profilometry');
h2 = plot(thrFrames1/(29.97*1.05), nDetsByThr1, 's-', 'LineWidth',1.5, 'DisplayName','Ceiling');
xlabel('$t_{\rm min}/T_\infty$','Interpreter','latex')
ylabel('Total detections','Interpreter','latex')
xlim([0 .4]); ylim([0 max([nDetsByThr1(:);nDetsByThr2(:)])*1.1])
grid on
legend('Location','northeast','Interpreter','latex')
set(findall(h,'-property','FontSize'),'FontSize',18)


%% Plot results
hfig = figure;

plot((thresholds/45)/1.05, accuracy_prof, '-o','LineWidth',1.5, 'DisplayName','Profilometry')
hold on;
plot((thresholds(1:12)/29.97)/1.05, accuracy_ceil(1:12), 'k-o','LineWidth',1.5, 'DisplayName','Ceiling');
xlabel('$t_{\rm min} / T_\infty$');
ylabel('Matching Accuracy');
%yline(0.15373, 'k--', LineWidth=1, HandleVisibility='off');
xline(0.104, 'k-.', LineWidth=1, HandleVisibility='off');
grid on;
xlim([0, 0.4]);
ylim([0, 1]);

% Set additional plotting properties
fname = 'output/MA_lifetimeThresh_r10_Wthr50_s8_lambda_6_minarea1_Run2_newNames_0to1Limit';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
legend('Location', 'best', 'FontSize', 18);
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
%save('accuracyAsFuncOfLifetimeThreshold_Wthr50_lambda_6_minArea1_r10', 'thresholds', 'accuracy_by_thresh');
%%
import = load("accuracyAsFuncOfLifetimeThreshold_Profilometry_Wthr0p2_lambda_6_minArea1_r10.mat");
accuracy_prof = import.accuracy_by_thresh;
import2 = load("accuracyAsFuncOfLifetimeThreshold_Wthr50_lambda_6_minArea1_r10_CORRECTED.mat");
accuracy_ceil = import2.accuracy_by_thresh;
thresholds = import.thresholds;

%% Try to plot trackliftetime and lifetime accuracy in same figure

%--- 2 rows, 1 column, tight spacing
hfig = figure;
t = tiledlayout(2,1, ...
    'TileSpacing','compact', ...
    'Padding','tight');

%=== Top tile: Total tracks ===
ax1 = nexttile;
plot(thresholds/(45*1.05), nTracksByThr2, 'o-', 'LineWidth', 1.5, 'DisplayName', 'Profilometry','Color',color1);
hold on
plot(thresholds(1:12)/(29.97*1.05), nTracksByThr1(1:12), 'o-', ...
     'LineWidth',1.5, 'DisplayName','Reflection','Color',color2);
hold on
xline(0.104,'-.','HandleVisibility','off');
%xlabel('$t_{\rm min}/T_\infty$','Interpreter','latex')
ylabel('Total tracks','Interpreter','latex')
% 1) Enable minor ticks (and minor grid):
legend('Location', 'best', 'FontSize', 20);
%xlim([0 0.4]);
ylim([-500, 2.5e4])
grid on;

%=== Bottom tile: Matching Accuracy ===
ax2 = nexttile;
plot((thresholds/45)/1.05, accuracy_prof, '-o','LineWidth',1.5, 'HandleVisibility','off')
hold on;
plot((thresholds(1:12)/29.97)/1.05, accuracy_ceil(1:12), 'k-o', ...
     'LineWidth',1.5, 'HandleVisibility','off');
hold on
xline(0.104,'-.','HandleVisibility','off');
xlabel('$t_{\rm min}/T_\infty$','Interpreter','latex')
ylabel('$\mathcal{M}$','Interpreter','latex')
xlim([0 0.4]); ylim([0 1])
grid on


% Set additional plotting properties
fname = 'output/trackLifetimeAndAccuracyLifetimeSubplot_bothDatasets_corrected';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.75; % Height-width ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
set(findall(hfig,'-property','Box'),'Box','on'); % Optional box
legend('Location', 'best', 'FontSize', 20, Box='off');
set(findall(hfig,'-property','Interpreter'),'Interpreter','latex');
set(findall(hfig,'-property','TickLabelInterpreter'),'TickLabelInterpreter','latex');
set(hfig,'Units','centimeters','Position',[3 3 picturewidth hw_ratio*picturewidth]);
% Configure printing options
pos = get(hfig,'Position');
set(hfig,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[pos(3), pos(4)]);

lg1 = legend(ax1);        % get the legend object on the top tile
lg1.Box = 'off';          % remove its border

% Export the figure
print(hfig, fname, '-dpdf', '-vector');%, '-fillpage');
%print(hfig, fname, '-dpng', '-vector');
