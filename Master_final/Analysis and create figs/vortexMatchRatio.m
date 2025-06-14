clear;clc;

%% load computer vision detections
load('..\createFigures\data\ceilingTrackingResults_38-2346_Wthr50_s8_Run2.mat');   % brings in tracks, trackInfo, stamps, times, tform, …
%load('..\createFigures\data\profilometryTrackingResults_1-10800_Wthr-0.16_s8_Run1.mat');
%% Importing transform file from manual calibration
transformFile = load('..\data\transform_apr11_test1_firstchunk.mat');
%transformFile = load('..\data\transform_may10_test1_profilometry_firstchunk.mat');
tform = transformFile.ell;
disp(tform.T);

%% Params
externalLower = 14;  % lower bound for externalCoords time index
externalUpper = 3600;  % upper bound for externalCoords time index
trackLower = 0.9;    % lower bound for tracks timestamps (in seconds)
trackUpper = 239.933;    % upper bound for tracks timestamps (in seconds)

% Spatial tolerance (error radius) for matching, in physical units.
error_radius = 10;  % Adjust as needed
lifetimeThreshold = 3.33;

t_min_vals    = 1:20;          % t_min values
nT            = numel(t_min_vals);

% Define the set of frames to process based on externalCoords.
frameRange = externalLower:externalUpper;  
nFrames = length(frameRange);

% Map the external frame range linearly to the track time range.
% (Assumes that externalLower corresponds to trackLower and externalUpper to trackUpper.)
track_time_vector = linspace(trackLower, trackUpper, nFrames);

% Preallocate
mean_sensitivity = nan(1, nT);
total_vortex_count = nan(1, nT);
%% Prepare to loop over frames

for iT = 1:nT
    t_min = t_min_vals(iT);

    % Load the external coords file for this t_min
    % (Adjust the file name pattern as needed)
    fn = sprintf('lambda2_data/filteredCentroids_14-3600_thresh6_area1_NormCoords_lifetimeThr_%02d.mat', t_min);
    data = load(fn, 'filteredCentroids');
    externalCoords = data.filteredCentroids;

    % Preallocate storage for unique matches per frame.
    allFrameMatches = cell(nFrames, 1);
    allFrameMatchCounts = zeros(nFrames, 1);
    accuracy = zeros(nFrames, 1);
    sensitivity = zeros(nFrames, 1);

    % Open a new figure for multi-frame overlays.
    %figure('Name','Unique Matching Per Frame','NumberTitle','off');
    
    for idx = 1:nFrames
        % For the current frame:
        currFrame = frameRange(idx);
        
        % --- External Detections ---
        % Filter externalCoords for detections at the current frame.
        % Here, externalCoords(:,3) is assumed to be the frame number.
        ext_idx = (externalCoords(:,3) == currFrame);
        ext_current = externalCoords(ext_idx, :);
        % Reorder to [x, y] (since externalCoords is [y, x, time]).
        ext_coords = [ext_current(:,2), ext_current(:,1)];
        % New addition: Filter external coordinates to include only those with y > 50.
        ext_coords = ext_coords(ext_coords(:,2) > 45, :);
        %ext_coords = ext_coords(ext_coords(:,2) > -110, :);
        %ext_coords = ext_coords(ext_coords(:,2) < 150, :);
        
        % --- Track (Dimple) Detections ---
        % Define the target time for this frame from the mapping.
        time_target = track_time_vector(idx);
        % Define a small tolerance for track time matching (in seconds).
        tolerance_time = 0.035;
        
        % Initialize an array to hold transformed track detections for this frame.
        current_track_coords = [];
        for k = 1:length(tracks)
            if trackInfo(k).lifetime >= lifetimeThreshold
                % Get the timestamps and centroids for the current track.
                track_times = tracks(k).timestamp;      % e.g., [1.5005 1.5672 ...]
                centroids   = tracks(k).centroids;        % [N x 2] (pixel coordinates)
                % Logical filter: select detections close to time_target.
                % CHOOSING THE CLOSEST TIME IDX MATCH
                selected_idx = abs(track_times - time_target) < tolerance_time;
                if any(selected_idx)
                    % Get the indices of detections meeting the condition.
                    candidateIdx = find(selected_idx);
                    % Compute the time difference for those candidates.
                    candidateDiffs = abs(track_times(candidateIdx) - time_target);
                    % Choose the candidate with the minimum difference.
                    [~, bestIdxRel] = min(candidateDiffs);
                    bestIdx = candidateIdx(bestIdxRel);
                    
                    % Now select only that one detection.
                    best_centroid = centroids(bestIdx, :);
                    
                    % Transform the pixel coordinate to physical coordinate.
                    phys_coords = transformPointsForward(tform, best_centroid);
                    
                    % ─── ADVECT IN X BY U_mean * dt ───
                    % Compute signed dt (track occurred after ext? dt>0 shifts forward)
                    dt              = track_times(bestIdx) - time_target;
                    U_mean          = 250;    % mean x‐speed [units/sec]
                    phys_coords(1)  = phys_coords(1) + U_mean * dt;
                    % ──────────────────────────────────
    
                    % Apply the x-coordinate restriction.
                    if phys_coords(1) >= 0 && phys_coords(1) <= 265 %% when using normalized coords this should be 0-265, else -88,133
                        % Append this detection.
                        current_track_coords = [current_track_coords; phys_coords]; %#ok<AGROW>
                    end
                end
            end
        end
        
        % --- Matching & Metrics per Frame ---
        if ~isempty(ext_coords) && ~isempty(current_track_coords)
            % 1) Compute distances and assignments
            D          = pdist2(ext_coords, current_track_coords);
            assignment = greedyMatchPairs(D, error_radius);
            matchCount = size(assignment,1);
        
            % 2) Store the raw matches
            allFrameMatches{idx}     = assignment;
            allFrameMatchCounts(idx) = matchCount;
        
            % 3) Accuracy (precision / correspondence fraction)
            total_dimples    = size(current_track_coords,1);
            accuracy(idx)    = matchCount / total_dimples;
        
            % 4) Sensitivity (recall)
            vortex_count_total = size(ext_coords,1);
            if vortex_count_total > 0
                vortex_match      = any(D <= error_radius, 2);
                sensitivity(idx)  = allFrameMatchCounts(idx) / vortex_count_total;
            else
                sensitivity(idx)  = NaN;
            end
        else
            % No data → zero matches, NaN stats
            allFrameMatches{idx}     = [];
            allFrameMatchCounts(idx) = 0;
            accuracy(idx)            = NaN;
            sensitivity(idx)         = NaN;
        end       
    end

    % store mean sensitivity (omit frames where ext_pts was empty)
    mean_sensitivity(iT) = mean(sensitivity, 'omitnan');
    total_vortex_count(iT) = sum(allFrameMatchCounts);
    fprintf('t_min=%02d → mean sensitivity = %.3f\n', t_min, mean_sensitivity(iT));
    fprintf('t_min=%02d → total matches = %.3f\n', t_min, total_vortex_count(iT));
end

fprintf('Average accuracy: ');
disp(mean(accuracy, 'omitnan'));

fprintf('Average Sensitivity: %.2f\n', mean(sensitivity, 'omitnan'));

fprintf('Total matches: ');
disp(sum(allFrameMatchCounts));

%%
save('vortexMatchRatio_1-20.mat', 't_min_vals', 'mean_sensitivity', 'total_vortex_count');
disp('SAVED')

%load('vortexMatchRatio_1-20.mat');

%% Plot the mean sensitivity as a function of subsurface lifetime threshold

hfig = figure;
plot(t_min_vals/(15*1.05), mean_sensitivity, 'k-o', 'LineWidth',1.5, 'HandleVisibility','off');
hold on;
yline(0.0213, '--', 'LineWidth',1.5, 'DisplayName', '$P[R_D]$');

xlim([0, 1.3]);

xlabel('$t_{\rm min, subsurface} / T_\infty$');
ylabel('$\mathcal{V}$');
grid on;
% Set additional plotting properties
fname = 'output/VORTEXMATCHRATIO_lifetimeThresh_r10_Wthr50_s8_lambda_6_minarea1';
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


%% HELPER

function assignment = greedyMatchPairs(D, threshold)
% greedyMatchPairs performs a greedy one-to-one assignment based on the cost
% matrix D. It returns assignments [i, j] such that D(i,j) is the pair's cost.
% Only pairs with cost <= threshold are accepted.
%
% D: cost matrix (rows: external detections, columns: track detections)
% threshold: maximum allowed distance for an accepted match

    assignment = [];
    % Copy D so we can modify it without altering the original.
    D_local = D;
    
    % Continue until no valid pairs remain.
    while true
        [minVal, linearIdx] = min(D_local(:));
        if isempty(minVal) || minVal > threshold
            break;
        end
        [row, col] = ind2sub(size(D_local), linearIdx);
        % Save this match.
        assignment(end+1,:) = [row, col];  %#ok<AGROW>
        % Remove the matched row and column by setting them to Inf.
        D_local(row, :) = Inf;
        D_local(:, col) = Inf;
    end
end

%% 1) Extract lifetimes and detection-counts per track
nTracks = numel(trackInfo);
lifetimes      = [trackInfo.lifetime];                % lifetime (in frames or seconds)
detCountPerTrk = arrayfun(@(t) size(t.coordinates,1), trackInfo);

%% 2) Define thresholds to test
thrVec = 0:1:20;   % e.g. from 0 up to 20 seconds (or frames, depending lifetime unit)
nT     = numel(thrVec);

%% 3) For each threshold, compute:
%   a) # of tracks with lifetime ≥ threshold
%   b) total detections from those tracks
trackCounts     = nan(1,nT);
detectionCounts = nan(1,nT);

for i = 1:nT
    thr = thrVec(i);
    idxAbove      = lifetimes >= thr;
    trackCounts(i)     = sum(idxAbove);
    detectionCounts(i) = sum(detCountPerTrk(idxAbove));
end

%% 4) Display as a table
T = table(thrVec.', trackCounts.', detectionCounts.', ...
          'VariableNames', {'LifetimeThreshold','NumTracks','NumDetections'});
disp(T);

%% 5) (Optional) Plot
figure;
yyaxis left
plot(thrVec, trackCounts, 'o-','LineWidth',1.5);
ylabel('Number of Tracks');

yyaxis right
plot(thrVec, detectionCounts, 's--','LineWidth',1.5);
ylabel('Total Detections');

xlabel('Lifetime Threshold');
title('Tracks and Detections Above Lifetime Threshold');
legend('Tracks','Detections','Location','best');
grid on;