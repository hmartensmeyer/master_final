clear;clc;

%% load computer vision detections
%load('..\createFigures\data\ceilingTrackingResults_38-2346_Wthr50_s8_Run2.mat');   % brings in tracks, trackInfo, stamps, times, tform, …
load('..\createFigures\data\profilometryTrackingResults_1-10800_Wthr-0.200_s8_Run1.mat');
%% Importing transform file from manual calibration
% transformFile = load('..\data\transform_apr11_test1_firstchunk.mat');
% tform = transformFile.ell;

%% Importing the coordinates from the lambda2 PIV data

test = load("lambda2_data\allCentroids_14-3600_thresh6_area1_OrigCoords_blur_3x3.mat", "allCentroids"); %importing the lambda 2 coordinates
externalCoords = test.allCentroids;

% test = load('lambda2_data\filteredCentroids_14-3600_thresh6_area1_OrigCoords_lifetimeThr_2.mat');
% externalCoords = test.filteredCentroids;

% plot(externalCoords(:,1), externalCoords(:,2), 'rx', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'PIV detections');
% hold on;
% plot(nan, nan, 'ko', 'LineWidth', 2, 'MarkerSize', 15, 'DisplayName', 'Ceiling detections');

lifetimeThreshold = 0;
%% Loop through each track and convert the pixel coordinates to physical coordinates.
% for k = 1:length(trackInfo)
%     if trackInfo(k).lifetime > lifetimeThreshold
%         % Get the pixel coordinates from track
%         centroids_pixel = trackInfo(k).coordinates;
% 
%         % Transform to physical coordinates using calibration transform.
%         centroids_physical = transformPointsForward(tform, centroids_pixel);
% 
%         % Plot the track; swap coordinates if needed.
%         if size(centroids_physical,1) > 1
%             plot(centroids_physical(:,2), centroids_physical(:,1), 'o-', 'LineWidth', 2, 'MarkerSize', 12, 'Color', 'k', 'HandleVisibility','off');
%         else
%             plot(centroids_physical(2), centroids_physical(1), 'o', 'LineWidth', 2, 'MarkerSize', 12, 'Color', 'k', 'HandleVisibility','off');
%         end
%     end
% end

%% Define Time Ranges for Each Field

% For externalCoords: using the third column as time index.
externalLower = 14;  % lower bound for externalCoords time index
externalUpper = 3600;  % upper bound for externalCoords time index

% For tracks: using the timestamp field in seconds.
trackLower = 0.778;    % lower bound for tracks timestamps (in seconds)
trackUpper = 240;    % upper bound for tracks timestamps (in seconds)

% Parameters
error_radius    = 10;          % Spatial tolerance for matching
lifetimeThreshold = 5;         % Keep tracks ≥ 5 frames long

% Define the wavelet thresholds
W_thr_list = -0.20;
nW = numel(W_thr_list);

% Preallocate
accuracy_by_Wthr    = nan(1, nW);
sensitivity_by_Wthr= nan(1, nW);
totalMatches_by_Wthr= zeros(1, nW);

frameRange        = externalLower:externalUpper;  
nFrames           = numel(frameRange);
track_time_vector = linspace(trackLower, trackUpper, nFrames);
tolerance_time    = 0.035;      % time tolerance

%% Loop over each W_thr
for wi = 1:nW
    W_thr = W_thr_list(wi);

    % Load the corresponding tracking results
    % Build the exact filename
    matname = sprintf('profilometryTrackingResults_1-10800_Wthr%.3f_s8_Run1.mat', W_thr);
    fname   = fullfile('data', matname);
    fprintf('Loading file: "%s"\n', fname);

    S = load(fname, 'tracks','trackInfo','stamps');
    tracks    = S.tracks;
    trackInfo = S.trackInfo;
    stamps    = S.stamps;

    allFrameMatchCounts = zeros(nFrames,1);
    accuracy            = nan(nFrames,1);
    sensitivity         = nan(nFrames, 1);

    for idx = 1:nFrames
        currFrame = frameRange(idx);
        %disp(idx)
        if mod(idx,100)==0, fprintf('Working on step %d / %d. Wavelet threshold: %d \n', idx, nFrames, W_thr), end
        % --- External detections ---
        ext_idx     = (externalCoords(:,3) == currFrame);
        ext_current = externalCoords(ext_idx, :);
        ext_coords  = [ext_current(:,2), ext_current(:,1)];
        %ext_coords  = ext_coords(ext_coords(:,2) > 45, :);

        % --- Track detections ---
        time_target          = track_time_vector(idx);
        current_track_coords = [];
        for k = 1:numel(tracks)
            if trackInfo(k).lifetime >= lifetimeThreshold
                track_times = tracks(k).timestamp;
                centroids   = tracks(k).centroids;
                sel = abs(track_times - time_target) < tolerance_time;
                if any(sel)
                    candidateIdx = find(sel);
                    diffs        = abs(track_times(candidateIdx) - time_target);
                    [~, bestRel] = min(diffs);
                    bestIdx      = candidateIdx(bestRel);
                    best_centroid= centroids(bestIdx,:);
                    phys_coords  = best_centroid;
                    if phys_coords(1) >= -88 && phys_coords(1) <= 133
                        current_track_coords(end+1,:) = phys_coords; %#ok<AGROW>
                    end
                end
            end
        end

        % % --- Matching ---
        % if ~isempty(ext_coords) && ~isempty(current_track_coords)
        %     D          = pdist2(ext_coords, current_track_coords);
        %     assignment = greedyMatchPairs(D, error_radius);
        %     allFrameMatchCounts(idx) = size(assignment,1);
        % else
        %     allFrameMatchCounts(idx) = 0;
        % end
        % 
        % % --- Accuracy per frame ---
        % total_dimples = size(current_track_coords,1);
        % % if total_dimples > 0
        % %     accuracy(idx) = allFrameMatchCounts(idx)/ total_dimples;
        % % end
        % if total_dimples>0
        %   accuracy(idx)= allFrameMatchCounts(idx)/total_dimples;
        % else
        %   accuracy(idx)= NaN;
        % end
        % 
        % vortex_match = any(D <= error_radius, 2);
        % vortex_count_total = size(ext_coords, 1);
        % total_vortices_matched = sum(vortex_match);
        % 
        % if vortex_count_total > 0
        %     sensitivity(idx) = allFrameMatchCounts(idx) / vortex_count_total;
        % else
        %     sensitivity(idx) = NaN;  % Skip frame in average if no vortices are present.
        % end
        % --- Matching & Metrics ---
        if ~isempty(ext_coords) && ~isempty(current_track_coords)
            % Compute pairwise distances & matches
            D          = pdist2(ext_coords, current_track_coords);
            assignment = greedyMatchPairs(D, error_radius);
            matchCount = size(assignment,1);
            
            % Accuracy (precision / correspondence fraction)
            total_dimples = size(current_track_coords,1);
            accuracy(idx) = matchCount / total_dimples;
        
            % Sensitivity (recall / hit rate)
            vortex_count_total = size(ext_coords,1);
            if vortex_count_total > 0
                vortex_match = any(D <= error_radius, 2);
                sensitivity(idx) = sum(vortex_match) / vortex_count_total;
            else
                sensitivity(idx) = NaN;
            end
        else
            % No matches at all
            matchCount             = 0;
            allFrameMatchCounts(idx) = matchCount;
            accuracy(idx)          = NaN;
            sensitivity(idx)       = NaN;
        end
        
        % Store the match count (do this outside if/else so it’s always set)
        allFrameMatchCounts(idx) = matchCount;

    end

    % Store results for this W_thr
    accuracy_by_Wthr(wi)     = mean(accuracy, 'omitnan');
    sensitivity_by_Wthr(wi)  = mean(sensitivity, 'omitnan');
    totalMatches_by_Wthr(wi) = sum(allFrameMatchCounts);
end
%%
fprintf('Accuracy: %.5f \n', accuracy_by_Wthr);
fprintf('Sensitivity: %.5f \n', sensitivity_by_Wthr);

disp(totalMatches_by_Wthr)

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

%% Plot the Matching Accuracy as a function of lifetimeThreshold

% Parameters
error_radius = 10;               % Spatial tolerance for matching
thresholds = 1:1:18;               % Lifetime thresholds to test
nThresh    = numel(thresholds);

% Preallocate
accuracy_by_thresh = nan(1, nThresh);

% Frame mapping (unchanged)
frameRange        = externalLower:externalUpper;  
nFrames           = numel(frameRange);
track_time_vector = linspace(trackLower, trackUpper, nFrames);
tolerance_time    = 0.035;  % time tolerance

for t = 1:nThresh
    lifetimeThreshold = thresholds(t);
    disp(t);
    % Storage per-frame
    allFrameMatchCounts = zeros(nFrames,1);
    accuracy            = nan(nFrames,1);

    for idx = 1:nFrames
        currFrame = frameRange(idx);
        if mod(idx,100)==0, fprintf('Working on step %d / %d. Lifetime threshold: %d \n', idx, nFrames, t), end
        % --- External detections ---
        ext_idx     = (externalCoords(:,3) == currFrame);
        ext_current = externalCoords(ext_idx, :);
        ext_coords  = [ext_current(:,2), ext_current(:,1)];
        %ext_coords  = ext_coords(ext_coords(:,2) > 45, :);

        % --- Track detections ---
        time_target         = track_time_vector(idx);
        current_track_coords = [];
        for k = 1:numel(tracks)
            if trackInfo(k).lifetime >= lifetimeThreshold
                track_times = tracks(k).timestamp;
                centroids   = tracks(k).centroids;

                sel = abs(track_times - time_target) < tolerance_time;
                if any(sel)
                    candidateIdx   = find(sel);
                    diffs          = abs(track_times(candidateIdx) - time_target);
                    [~, bestRel]   = min(diffs);
                    bestIdx        = candidateIdx(bestRel);
                    best_centroid  = centroids(bestIdx,:);
                    phys_coords    = best_centroid;
                    if phys_coords(1)>=-88 && phys_coords(1)<=133
                        current_track_coords(end+1,:) = phys_coords;  %#ok<AGROW>
                    end
                end
            end
        end

        % --- Matching ---
        if ~isempty(ext_coords) && ~isempty(current_track_coords)
            D          = pdist2(ext_coords, current_track_coords);
            assignment = greedyMatchPairs(D, error_radius);
            allFrameMatchCounts(idx) = size(assignment,1);
        else
            allFrameMatchCounts(idx) = 0;
        end

        % --- Accuracy per frame ---
        total_dimples = size(current_track_coords,1);
        % if total_dimples>0
        %     accuracy(idx) = allFrameMatchCounts(idx)/ total_dimples;
        % end
        if total_dimples>0
          accuracy(idx)= allFrameMatchCounts(idx)/total_dimples;
        else
          accuracy(idx)= NaN;
        end
    end

    % Mean accuracy for this threshold
    accuracy_by_thresh(t) = mean(accuracy, 'omitnan');
end

%% Plot results
hfig = figure;
plot(thresholds/(45*1.05), accuracy_by_thresh, 'k-o','LineWidth',1.5);
xlabel('$t_{min} / T_\infty$');
ylabel('Matching Accuracy');
hold on;
%yline(0.14688, 'k--', LineWidth=1.5);
grid on;
%xlim([0.05,1.15]);

% Set additional plotting properties
fname = 'output/MA_lifetimeThresh_profilometry_r10_Wthr0p2_s8_lambda_6_minarea1_Run2';
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

%%
%save('accuracyAsFuncOfLifetimeThreshold_Profilometry_Wthr0p2_lambda_6_minArea1_r10', 'thresholds', 'accuracy_by_thresh');
%%



%% Plot the Matching Accuracy as a function of error radii
% Parameters
lifetimeThreshold = 5;         % fixed for this sweep
errorRadii = 1:1:15;          % spatial tolerances to test (in physical units)
nR        = numel(errorRadii);

% Frame mapping (unchanged)
frameRange        = externalLower:externalUpper;
nFrames           = numel(frameRange);
track_time_vector = linspace(trackLower, trackUpper, nFrames);
tolerance_time    = 0.035;      % time tolerance

% Preallocate
accuracy_by_radius = nan(1, nR);

for rIdx = 1:nR
    error_radius = errorRadii(rIdx);
    disp(rIdx)

    % Storage per-frame
    allFrameMatchCounts = zeros(nFrames,1);
    accuracy            = nan(nFrames,1);

    for idx = 1:nFrames
        currFrame = frameRange(idx);
        if mod(idx,100)==0, fprintf('Working on step %d / %d. Search radius: %d \n', idx, nFrames, error_radius), end
        % --- External detections ---
        ext_idx     = (externalCoords(:,3) == currFrame);
        ext_current = externalCoords(ext_idx, :);
        ext_coords  = [ext_current(:,2), ext_current(:,1)];
        %ext_coords  = ext_coords(ext_coords(:,2) > 45, :);

        % --- Track detections ---
        time_target          = track_time_vector(idx);
        current_track_coords = [];
        for k = 1:numel(tracks)
            if trackInfo(k).lifetime >= lifetimeThreshold
                track_times = tracks(k).timestamp;
                centroids   = tracks(k).centroids;

                sel = abs(track_times - time_target) < tolerance_time;
                if any(sel)
                    candidateIdx   = find(sel);
                    diffs          = abs(track_times(candidateIdx) - time_target);
                    [~, bestRel]   = min(diffs);
                    bestIdx        = candidateIdx(bestRel);
                    best_centroid  = centroids(bestIdx,:);
                    phys_coords    = best_centroid;
                    if phys_coords(1)>=-88 && phys_coords(1)<=133
                        current_track_coords(end+1,:) = phys_coords;  %#ok<AGROW>
                    end
                end
            end
        end

        % --- Matching ---
        if ~isempty(ext_coords) && ~isempty(current_track_coords)
            D          = pdist2(ext_coords, current_track_coords);
            assignment = greedyMatchPairs(D, error_radius);
            allFrameMatchCounts(idx) = size(assignment,1);
        else
            allFrameMatchCounts(idx) = 0;
        end

        % --- Accuracy per frame ---
        total_dimples = size(current_track_coords,1);
        if total_dimples>0
            accuracy(idx) = allFrameMatchCounts(idx)/ total_dimples;
        else
          accuracy(idx)= NaN;
        end
    end

    % Mean accuracy for this error radius
    accuracy_by_radius(rIdx) = mean(accuracy, 'omitnan');
end

%% Plot results
hfig = figure;
plot(errorRadii/20, accuracy_by_radius, '-o', 'LineWidth',1.5);
xlabel('Error Radius');
ylabel('Mean Accuracy');
title('Matching Accuracy vs. Spatial Tolerance');
grid on;

% Set additional plotting properties
fname = 'output/MA_radiusThreshold_profilometry_t5_Wthr0p2_s8_lambda_6_minarea1_Run2';
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

%% Plot with probability as well

% --- Compute P(random) for each error_radius (with r^2) ---
% P_random = ((55652/3546) * r^2 * pi) / 755252
%P_random = ((109952/3546) .* (errorRadii.^2) * pi) ./ 75252;
P_random = ((132534/3546) .* (errorRadii.^2) * pi) ./ 75525;

% --- Plot both curves on the same y-axis ---
hfig = figure;

plot(errorRadii(2:end) / 16.7, accuracy_by_radius(2:end), 'k-o', 'LineWidth',1.5);
hold on;
plot(errorRadii(2:end) / 16.7, P_random(2:end), 'k--', 'LineWidth',1);
hold off;

xlabel('$r_s / \lambda_T$');
ylabel('Matching accuracy');
legend('Matching accuracy','$P[V_{\rm sub}(r_s)]$','Location','northwest');
xlim([0.1, 0.92])

grid on;

% Set additional plotting properties
fname = 'output/MA_radiusThreshold_profilometry_t5_Wthr0p2_s8_lambda_6_minarea1_Run2';
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
print(hfig, fname, '-dpdf', '-vector');%, '-fillpage');
%print(hfig, fname, '-dpng', '-vector');

%% Save results to enable simul plotting

save('accuracyAsFuncOfRadius_Profilometry_Wthr0p2_lambda_6_minArea1_t5', 'errorRadii', 'accuracy_by_radius');