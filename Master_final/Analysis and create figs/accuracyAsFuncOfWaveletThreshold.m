clear;clc;
%% Script reading all files with different wavelets thresholds and plots the 
% matching accuracy and match count as a function of wavelet threshold
% Important that all the files have the same scale


% Parameters
error_radius    = 10;          % Spatial tolerance for matching
lifetimeThreshold = 3.333;         % Keep tracks ≥ 5 frames long

externalLower   = 14;          
externalUpper   = 3600;        
trackLower      = 0.9;         
trackUpper      = 239.933;     

frameRange        = externalLower:externalUpper;  
nFrames           = numel(frameRange);
track_time_vector = linspace(trackLower, trackUpper, nFrames);
tolerance_time    = 0.04;      % time tolerance

% load once
E = load("lambda2_data\allCentroids_14-3600_thresh6_area1_NormCoords_blur_3x3.mat", "allCentroids");
externalCoords = E.allCentroids;
% calibration
TF   = load('..\data\transform_apr11_test1_firstchunk.mat');
tform= TF.ell;

%% Define wavelet thresholds
W_thr_list = 20:2:60;
nW = numel(W_thr_list);

% Preallocate
accuracy_by_Wthr    = nan(1, nW);
totalMatches_by_Wthr= zeros(1, nW);

%% Loop over each W_thr
for wi = 1:nW
    W_thr = W_thr_list(wi);
    disp(wi);

    % Load the corresponding tracking results
    % Build the exact filename
    matname = sprintf('ceilingTrackingResults_38-2346_Wthr%d_s8_Run2.mat', W_thr);
    fname   = fullfile('data', matname);
    fprintf('Loading file: "%s"\n', fname);

    S = load(fname, 'tracks','trackInfo','stamps','times');
    tracks    = S.tracks;
    trackInfo = S.trackInfo;
    stamps    = S.stamps;
    times     = S.times;

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
                    phys_coords  = transformPointsForward(tform, best_centroid);
                    if phys_coords(1)>=0 && phys_coords(1)<=265
                        current_track_coords(end+1,:) = phys_coords; %#ok<AGROW>
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

    % Store results for this W_thr
    accuracy_by_Wthr(wi)     = mean(accuracy, 'omitnan');
    totalMatches_by_Wthr(wi) = sum(allFrameMatchCounts);
end

%% Plot results
% hfig = figure;
% 
% % 1) Mean matching accuracy vs W_thr
% subplot(2,1,1);
% plot(W_thr_list, accuracy_by_Wthr, 'k-o','LineWidth',1.5,'MarkerSize',6);
% xlabel('Wavelet threshold $W_{\rm thr}$','Interpreter','latex');
% ylabel('Mean matching accuracy','Interpreter','latex');
% title(sprintf('Lifetime $\\ge %d$ frames', lifetimeThreshold),'Interpreter','latex');
% grid on;
% 
% % 2) Total # matches vs W_thr
% subplot(2,1,2);
% plot(W_thr_list, totalMatches_by_Wthr, 'b-s','LineWidth',1.5,'MarkerSize',6);
% xlabel('Wavelet threshold $W_{\rm thr}$','Interpreter','latex');
% ylabel('Total # matches','Interpreter','latex');
% grid on;
% 
% % Optional formatting
% set(findall(gcf,'-property','FontSize'),'FontSize',14);
% set(findall(gcf,'-property','Interpreter'),'Interpreter','latex');
% set(gcf,'Units','centimeters','Position',[3 3 20 16]);
% box on;

hfig = figure; 
set(hfig, 'Units','centimeters','Position',[3 3 20 12]);

yyaxis right
plot(W_thr_list, accuracy_by_Wthr, 'k-o', ...
     'LineWidth',1.5,'MarkerSize',6, 'Color','k');
ylabel('$\mathcal{M}$', 'Color','k','Interpreter','latex');
set(gca, 'YColor', 'k')
%ylim([0.1,0.5]);        % since accuracy is a fraction

yyaxis left
plot(W_thr_list, totalMatches_by_Wthr, '-s', ...
     'LineWidth',1.5,'MarkerSize',6);
ylabel('Total matches','Interpreter','latex');

xlabel('$W_{\rm thr}$','Interpreter','latex');

grid on;
box on;

% Legend
legend({ 'Total matches', 'Match ratio' }, ...
       'Location','best','Interpreter','latex');

% Formatting
set(findall(gcf,'-property','FontSize'),'FontSize',14);
set(findall(gcf,'-property','Interpreter'),'Interpreter','latex');


% Set additional plotting properties
fname = 'output/accuracyAndMatchesAsFuncOfWaveletThreshold_s8_Run2_lambda2_6_minArea1_blur_rad10_tmin3p33';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
legend('Location','south', 'FontSize', 18);
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

%% Helper: greedyMatchPairs

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