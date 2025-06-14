clear;clc;

%% load computer vision detections
load('..\createFigures\data\ceilingTrackingResults_38-2346_Wthr50_s8_Run2.mat');   % brings in tracks, trackInfo, stamps, times, tform, …
%load('..\createFigures\data\profilometryTrackingResults_1-10800_Wthr-0.16_s8_Run1.mat');
%% Importing transform file from manual calibration
transformFile = load('..\data\transform_apr11_test1_firstchunk.mat');
%transformFile = load('..\data\transform_may10_test1_profilometry_firstchunk.mat');
tform = transformFile.ell;
disp(tform.T);

%% Importing the coordinates from the lambda2 PIV data

test = load("lambda2_data\allCentroids_14-3600_thresh6_area1_NormCoords_blur_3x3.mat", "allCentroids"); %importing the lambda 2 coordinates
externalCoords = test.allCentroids;

% test = load('lambda2_data\filteredCentroids_14-3600_thresh6_area1_NormCoords_lifetimeThr_2.mat');
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
trackLower = 0.9;    % lower bound for tracks timestamps (in seconds)
trackUpper = 239.933;    % upper bound for tracks timestamps (in seconds)


%% Create a logical filter for externalCoords based on the time index.
% timeFilter_ext = (externalCoords(:,3) >= externalLower) & (externalCoords(:,3) <= externalUpper);
% filtered_external = externalCoords(timeFilter_ext, :);
% 
% % Initialize an empty matrix to collect transformed coordinates.
% all_physical_coords = [];
% lifetimeThreshold = 0;
% % Loop through every track in the structure array.
% for k = 1:length(tracks)
%     if trackInfo(k).lifetime >= lifetimeThreshold
%         % Obtain the timestamp vector and centroids for the current track.
%         track_times = tracks(k).timestamp;      % e.g., [1.5005 1.5672 ... 2.2341]
%         centroids   = tracks(k).centroids;        % e.g., [N x 2] double array
% 
%         % Create a logical filter for timestamps that fall within the desired window.
%         selected_idx = (track_times >= trackLower) & (track_times <= trackUpper);
% 
%         % If there are any detections in this time window...
%         if any(selected_idx)
%             % Extract the centroids for the matching indices.
%             selected_centroids = centroids(selected_idx, :);
% 
%             % Transform the pixel coordinates to physical coordinates using tform.
%             phys_coords = transformPointsForward(tform, selected_centroids);
% 
%             % Aggregate these transformed coordinates.
%             all_physical_coords = [all_physical_coords; phys_coords]; %#ok<AGROW>
%         end
%     end
% end

%% TESTING TO LOOP IN TIME AND SEE IF WE GET ANY MATCHES
% Define Time Ranges for Each Field
% External detections: using column 3 as time index (frame numbers).
% externalLower = 14;  % lower bound for externalCoords time index
% externalUpper = 24;  % upper bound for externalCoords time index
% 
% % Track detections: using the timestamp field in seconds.
% trackLower = 0.9;    % lower bound for tracks timestamps (in seconds)
% trackUpper = 1.64;    % upper bound for tracks timestamps (in seconds)

% Spatial tolerance (error radius) for matching, in physical units.
error_radius = 10;  % Adjust as needed

% For processing tracks, already use a lifetime threshold.
lifetimeThreshold = 3.33;

% Prepare to loop over frames
% Define the set of frames to process based on externalCoords.
frameRange = externalLower:externalUpper;  
nFrames = length(frameRange);

% Map the external frame range linearly to the track time range.
% (Assumes that externalLower corresponds to trackLower and externalUpper to trackUpper.)
track_time_vector = linspace(trackLower, trackUpper, nFrames);

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
            % Logical filter: select detections close to time_target
            %(COULD CHOOSE MULTIPLE IDXS)
            % selected_idx = abs(track_times - time_target) < tolerance_time;
            % if any(selected_idx)
            %     % Extract the matching centroids.
            %     selected_centroids = centroids(selected_idx, :);
            %     % Transform pixel coordinates to physical coordinates using tform.
            %     phys_coords = transformPointsForward(tform, selected_centroids);
            %     % New restriction: keep only dimples with x-coordinate between 0 and 265.
            %     phys_coords = phys_coords(phys_coords(:,1) >= 0 & phys_coords(:,1) <= 265, :);
            % 
            %     % Append these detections.
            %     current_track_coords = [current_track_coords; phys_coords]; %#ok<AGROW>
            %end
        end
    end
    
    % % We'll match external detections (vortices) with track detections (dimples)
    % % for the current frame on a one-to-one basis.
    % if ~isempty(ext_coords) && ~isempty(current_track_coords)
    %     % Compute pairwise Euclidean distance matrix.
    %     D = pdist2(ext_coords, current_track_coords);
    % 
    %     % Use matchpairs to obtain unique assignments (only assignments with cost <= error_radius).
    %     % This function returns an array 'assignment' where each row is [vortex_index, dimple_index].
    %     % (Ensure have the Statistics and Machine Learning Toolbox.)
    %     %[assignment, cost] = matchpairs(D, error_radius);
    %     assignment = greedyMatchPairs(D, error_radius);
    % 
    %     allFrameMatches{idx} = assignment;
    %     allFrameMatchCounts(idx) = size(assignment, 1);
    % else
    %     allFrameMatches{idx} = [];
    %     allFrameMatchCounts(idx) = 0;
    % end
    % 
    % % % Visualization for the current frame
    % % % We plot two subplots per frame in an overall figure:
    % % % - Upper row: an overlay with both sets and dashed lines connecting matched pairs.
    % % % - Lower row: a simple plot showing the external detections and track detections.
    % % 
    % % % Determine subplot positions.
    % % % (For simplicity, we arrange as two rows and nFrames columns.)
    % % subplot(1, nFrames, idx);  % Upper row for overlay
    % % 
    % % % Plot external detections (vortices) in red.
    % % if ~isempty(ext_coords)
    % %     plot(ext_coords(:,1), ext_coords(:,2), 'ro', 'MarkerSize',8, 'LineWidth',1.5);
    % % end
    % % hold on;
    % % % Plot track detections (dimples) in blue.
    % % if ~isempty(current_track_coords)
    % %     plot(current_track_coords(:,1), current_track_coords(:,2), 'bo', 'MarkerSize',8, 'LineWidth',1.5);
    % % end
    % % % If there are matches, draw dashed lines connecting matched pairs.
    % % if ~isempty(allFrameMatches{idx})
    % %     for m = 1:size(allFrameMatches{idx},1)
    % %         vortexIdx = allFrameMatches{idx}(m,1);
    % %         dimpleIdx = allFrameMatches{idx}(m,2);
    % %         line([ext_coords(vortexIdx,1) current_track_coords(dimpleIdx,1)], ...
    % %              [ext_coords(vortexIdx,2) current_track_coords(dimpleIdx,2)], ...
    % %              'Color','k','LineStyle','-','LineWidth',2);
    % %     end
    % % end
    % 
    % % --- New Code: Compute the match ratio for the current frame
    % % Total number of dimple detections in current frame:
    % total_dimples = size(current_track_coords,1);
    % disp(total_dimples)
    % if total_dimples > 0
    %     accuracy(idx) = allFrameMatchCounts(idx) / total_dimples;
    % else
    %     accuracy(idx) = NaN;
    % end
    % 
    % % NEW CODE: Calculate sensitivity for the current frame.
    % % For each external detection (vortex), mark it as matched if any dimple is within error_radius.
    % vortex_match = any(D <= error_radius, 2);
    % vortex_count_total = size(ext_coords, 1);
    % total_vortices_matched = sum(vortex_match);
    % 
    % if vortex_count_total > 0
    %     sensitivity(idx) = allFrameMatchCounts(idx) / vortex_count_total;
    % else
    %     sensitivity(idx) = NaN;  % Skip frame in average if no vortices are present.
    % end
    
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
        disp(total_dimples)
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

    % % --- New Code: Annotate the overlay subplot with the match ratio
    % text(10, 330, sprintf('Accuracy: %.2f', accuracy), 'Color', 'k', 'FontSize', 10, 'FontWeight', 'bold');
    % 
    % title(sprintf('Frame %d, Matches: %d', currFrame, allFrameMatchCounts(idx)));
    % set(gca, 'YDir','reverse');
    % xlim([0,260]); ylim([0,350]);
    % grid on;
    
end

% Final Aggregated Matching Results
%fprintf('Unique matches per frame:\n');
%disp(allFrameMatchCounts);

% fprintf('Sensitivity per frame:');
% disp(sensitivity);

fprintf('Average accuracy: ');
disp(mean(accuracy, 'omitnan'));

fprintf('Average Sensitivity: %.2f\n', mean(sensitivity, 'omitnan'));

fprintf('Total matches: ');
disp(sum(allFrameMatchCounts));

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
thresholds = 1:18;               % Lifetime thresholds to test
nThresh    = numel(thresholds);

% Preallocate
accuracy_by_thresh = nan(1, nThresh);

% Frame mapping (unchanged)
frameRange        = externalLower:externalUpper;  
nFrames           = numel(frameRange);
track_time_vector = linspace(trackLower, trackUpper, nFrames);
tolerance_time    = 0.04;  % time tolerance

for t = 1:nThresh
    lifetimeThreshold = thresholds(t);
    disp(t);
    % Storage per-frame
    allFrameMatchCounts = zeros(nFrames,1);
    accuracy            = nan(nFrames,1);

    for idx = 1:nFrames
        currFrame = frameRange(idx);

        % --- External detections ---
        ext_idx     = (externalCoords(:,3) == currFrame);
        ext_current = externalCoords(ext_idx, :);
        ext_coords  = [ext_current(:,2), ext_current(:,1)];
        ext_coords  = ext_coords(ext_coords(:,2) > 45, :);

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
                    phys_coords    = transformPointsForward(tform, best_centroid);

                     % ─── ADVECT IN X BY U_mean * dt ───
                    % Compute signed dt (track occurred after ext? dt>0 shifts forward)
                    dt              = track_times(bestIdx) - time_target;
                    U_mean          = 250;    % mean x‐speed [units/sec]
                    phys_coords(1)  = phys_coords(1) + U_mean * dt;
                    % ──────────────────────────────────

                    if phys_coords(1)>=0 && phys_coords(1)<=265
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
        end
    end

    % Mean accuracy for this threshold
    accuracy_by_thresh(t) = mean(accuracy, 'omitnan');
end

%% Plot results
hfig = figure;

plot((thresholds/45)/1.05, accuracy_prof, '-o','LineWidth',1.5, 'DisplayName','Profilometry')
hold on;
plot((thresholds(1:12)/29.97)/1.05, accuracy_by_thresh(1:12), 'k-o','LineWidth',1.5, 'DisplayName','Ceiling');
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
save('accuracyAsFuncOfLifetimeThreshold_Wthr50_lambda_6_minArea1_r10_CORRECTED', 'thresholds', 'accuracy_by_thresh');
%%
import = load("accuracyAsFuncOfLifetimeThreshold_Profilometry_Wthr0p2_lambda_6_minArea1_r10.mat");
accuracy_prof = import.accuracy_by_thresh;
import2 = load("accuracyAsFuncOfLifetimeThreshold_Wthr50_lambda_6_minArea1_r10_CORRECTED.mat");
accuracy_ceil = import2.accuracy_by_thresh;
thresholds = import.thresholds;

%% Plot the Matching Accuracy as a function of error radii
% Parameters
lifetimeThreshold = 3.33333;         % fixed for this sweep
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
        ext_coords  = ext_coords(ext_coords(:,2) > 45, :);

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
                    phys_coords    = transformPointsForward(tform, best_centroid);

                    % ─── ADVECT IN X BY U_mean * dt ───
                    % Compute signed dt (track occurred after ext? dt>0 shifts forward)
                    dt              = track_times(bestIdx) - time_target;
                    U_mean          = 250;    % mean x‐speed [units/sec]
                    phys_coords(1)  = phys_coords(1) + U_mean * dt;
                    % ──────────────────────────────────

                    if phys_coords(1)>=0 && phys_coords(1)<=265
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
        end
    end

    % Mean accuracy for this error radius
    accuracy_by_radius(rIdx) = mean(accuracy, 'omitnan');
end

%% Plot results
hfig = figure;
plot(errorRadii(2:end)/16.7, accuracy_by_radius(2:end), '-o', 'LineWidth',1.5);
xlabel('Error Radius');
ylabel('Mean Accuracy');
title('Matching Accuracy vs. Spatial Tolerance');
grid on;

% Set additional plotting properties
fname = 'output/waveletThresholds_scale10_Wthr40';
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
P_random = ((132534/3546) .* (errorRadii.^2) * pi) ./ 75252;
P_random_filt = ((82173/3546) .* (errorRadii.^2) * pi) ./ 75252;

% --- Plot both curves on the same y-axis ---
hfig = figure;

%plot(errorRadii(3:end) / 16.7, acc_prof(3:end), 'k-o', 'LineWidth',1.5);
plot(errorRadii(3:end) / 16.7, acc_prof(3:end), '-o', 'LineWidth',1.5, 'DisplayName','Profilometry');
hold on;
plot(errorRadii(3:end) / 16.7, accuracy_by_radius_unfilt(3:end), 'k-o', 'LineWidth',1.5, 'DisplayName','Reflection');
hold on;
%plot(errorRadii(3:end) / 16.7, accuracy_by_radius_filt(3:end), '-o', 'LineWidth',1.5, 'DisplayName', 'Ceiling $(t_{\rm min, subsurface} = 0.105T_\infty)$', 'Color', [0.86 0.4 0.2]);
%plot(errorRadii(3:end) / 16.7, P_random(3:end), 'k--', 'LineWidth',1, 'DisplayName','$P[R_v(r_s)]$');
%plot(errorRadii(3:end) / 16.7, P_random_filt(3:end), '-.', 'LineWidth',1, 'DisplayName','$P[R_v(r_s)] (t_{\rm min, subsurface} = 0.105T_\infty)$', 'Color', [0.86 0.4 0.2]);
hold off;

xlabel('$r_s / \lambda_T$');
ylabel('$\mathcal{M}$');
legend('Location','northwest');
xlim([0.15, 0.92])
ylim([0, 1]);

grid on;

% Set additional plotting properties
fname = 'output/MA_time3p33_radius_Wthr50ceil_Wthr0p2prof_s8_lambda_6_minarea1_Run2_corrected_noRandom';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio
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
print(hfig, fname, '-dpdf', '-vector');%, '-fillpage');
%print(hfig, fname, '-dpng', '-vector');

%% Saving and loading
% 
%save('accuracyAsFuncOfRadius_Ceiling_Wthr50_lambda_6_minArea1_t3p33_advected_tmin_subsurf_5', 'errorRadii', 'accuracy_by_radius');

%%
import = load("accuracyAsFuncOfRadius_Profilometry_Wthr0p2_lambda_6_minArea1_t5.mat");
acc_prof = import.accuracy_by_radius;
import2 = load('accuracyAsFuncOfRadius_Ceiling_Wthr50_lambda_6_minArea1_t3p33_advected');
accuracy_by_radius_unfilt = import2.accuracy_by_radius;
errorRadii = import2.errorRadii;
import3 = load('accuracyAsFuncOfRadius_Ceiling_Wthr50_lambda_6_minArea1_t3p33_advected_tmin_subsurf_2');
accuracy_by_radius_filt = import3.accuracy_by_radius;