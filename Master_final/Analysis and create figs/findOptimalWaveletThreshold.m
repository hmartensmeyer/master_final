% %% I. Read Video and Prepare Timestamps
% data = load ('..\data\SZ_VFD10p5Hz_TimeResolved_Run1_30fps.mat');
% video = data.calibratedVideo;
% times = data.timeIndeces;
% stamps = data.timeStamps;
% 
% [height, width, numFrames] = size(video);
% eta = video;  % Preallocate the 3D frame matrix
% disp('Data read and converted to correct form.');
% 
% %% MEAN SUBTRACTION TO REMOVE THE BLACK CEILING PANELS
% mean_frame = mean(eta, 3);      % Compute mean frame
% eta_meansub = double(eta) - mean_frame;  % Mean-subtracted data
% 
% %% II. Wavelet Analysis Parameters
% scales = 1:5;
% selected_scale = 5;  % Use the desired scale (as before)
% % (W_thr is not a constant now since we will vary it)
% eccentricity_threshold = 0.9;
% solidity_threshold = 0.6;
% 
% %% III. Define Wavelet Threshold Optimization Parameters
% % Define a range of wavelet thresholds to test. Adjust the range as needed.
% waveletThresholds = linspace(-25, 50, 75);  % For example, 20 threshold values from 5 to 50
% numThresh = length(waveletThresholds);
% 
% % Choose the frame indices to analyze
% frameIndices = 38:numFrames;
% numFramesForAnalysis = length(frameIndices);
% 
% % Preallocate matrices to store both raw and filtered coverage rates.
% % Raw: Fraction of pixels in the wavelet coefficient map exceeding the threshold.
% % Filtered: Fraction of pixels remaining after filtering for eccentricity and solidity.
% coverageRates_raw = zeros(numFramesForAnalysis, numThresh);
% %coverageRates_filtered = zeros(numFramesForAnalysis, numThresh);
% 
% %% IV. Loop Over Selected Frames and Wavelet Thresholds
% for idx = 1:numFramesForAnalysis
%     t_index = frameIndices(idx);
%     disp(t_index);
% 
%     % Get the current frame (mean-subtracted)
%     snapshot = eta_meansub(:, :, t_index);
% 
%     % Perform the continuous wavelet transform
%     cwt_result = cwtft2(snapshot, 'Wavelet', 'mexh', 'Scales', scales);
%     wavelet_coefficients = cwt_result.cfs(:, :, selected_scale);
% 
%     for th = 1:numThresh
%         current_thr = waveletThresholds(th);
% 
%         %----- RAW COVERAGE RATE COMPUTATION -----
%         % Create a binary mask where the wavelet coefficient values exceed the threshold.
%         mask_raw = wavelet_coefficients > current_thr;
%         coverageRates_raw(idx, th) = sum(mask_raw(:)) / (numel(mask_raw)); %Subtracting the distortion volume
% 
%         %----- FILTERING FOR ECCENTRICITY AND SOLIDITY -----
%         % Identify connected components in the raw mask.
%         % connected_components = bwconncomp(mask_raw);
%         % if connected_components.NumObjects > 0
%         %     % Compute region properties (including eccentricity and solidity).
%         %     region_props = regionprops(connected_components, 'Area', 'Eccentricity', 'Solidity');
%         %     % Identify regions that meet the filtering criteria.
%         %     validIdx = find([region_props.Eccentricity] < eccentricity_threshold & ...
%         %                       [region_props.Solidity] > solidity_threshold);
%         %     % Create a binary mask of valid regions.
%         %     % labelmatrix converts connected components to a label image.
%         %     eccentric_regions = ismember(labelmatrix(connected_components), validIdx);
%         % else
%         %     % If no regions are found, the filtered mask is all false.
%         %     eccentric_regions = false(size(mask_raw));
%         % end
%         % 
%         % % Compute the filtered coverage rate (fraction of pixels in valid regions).
%         % coverageRates_filtered(idx, th) = sum(eccentric_regions(:)) / numel(eccentric_regions);
%     end
% end
% 
% %% V. Compute the Overall Coverage Rates vs. Threshold
% % Average (or RMS) the coverage rate across the selected frames for each threshold value.
% meanCoverageRate_raw = mean(coverageRates_raw, 1);
% %meanCoverageRate_filtered = mean(coverageRates_filtered, 1);
% 
% %% VI. Plot the Results
% % Plot Mean Raw Coverage Rate versus Wavelet Threshold
% hfig = figure;
% plot(waveletThresholds, meanCoverageRate_raw, 'k', 'LineWidth', 1);
% hold on;
% plot(waveletThresholds, meanCoverageRate_raw, 'k+', 'LineWidth', 2);
% xlabel('$\mathcal{W}_{thr}$', Interpreter='latex');
% ylabel('Coverage fraction');
% grid on;
% xlim([-25, 50]);
% ylim([-0.01,1.03])
% 
% % Set additional plotting properties
% fname = 'output/test';
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
% 
% % % Plot Mean Filtered Coverage Rate (after eccentricity & solidity) versus Wavelet Threshold
% % figure;
% % plot(waveletThresholds, meanCoverageRate_filtered, 'LineWidth', 2);
% % xlabel('Wavelet Threshold');
% % ylabel('Mean Filtered Coverage Rate');
% % title('Wavelet Threshold vs. Mean Filtered Coverage Rate');
% % grid on;
% % 
% % % Optionally, plot both curves on the same figure for comparison.
% % figure;
% % plot(waveletThresholds, meanCoverageRate_raw, 'b-', 'LineWidth', 2); hold on;
% % plot(waveletThresholds, meanCoverageRate_filtered, 'r-', 'LineWidth', 2);
% % xlabel('Wavelet Threshold');
% % ylabel('Mean Coverage Rate');
% % title('Raw vs. Filtered Coverage Rate');
% % legend('Raw', 'Filtered (Eccentricity & Solidity)');
% % grid on;


%% ALTERNATIVE CONSTRUCTION USING ONLY AREA OF INTERST
% Full pipeline: load video, mean-subtract, warp frames, wavelet analysis,
% threshold looping, ROI-restricted coverage rates, and plotting.

%% 0. Load Video Data and Spatial Transform
% Load video and saved spatial transform (replace path as needed)
data = load('..\data\SZ_VFD10p5Hz_TimeResolved_Run1_30fps.mat');
video = data.calibratedVideo;
times = data.timeIndeces;
stamps = data.timeStamps;
[height, width, numFrames] = size(video);

% Convert video to double for processing
etav = double(video);

% Load spatial transform object (from imregtform, projective2d, etc.)
S = load('..\data\transform_apr11_test1_firstchunk.mat');
tform = S.ell;

%% 1. Define Output View for Warping
% Compute world limits by transforming image corners
gcorn = {[1,1]; [width,1]; [width,height]; [1,height]};
[xx, yy] = deal(zeros(1,4));
for k = 1:4
    [xx(k), yy(k)] = transformPointsForward(tform, gcorn{k}(1), gcorn{k}(2));
end
xOut = [min(xx), max(xx)];
yOut = [min(yy), max(yy)];
outputView = imref2d([height, width], xOut, yOut);

% Build world-grid of the warped pixel centers
xWorld = linspace(outputView.XWorldLimits(1), outputView.XWorldLimits(2), outputView.ImageSize(2));
yWorld = linspace(outputView.YWorldLimits(1), outputView.YWorldLimits(2), outputView.ImageSize(1));
[Xworld, Yworld] = meshgrid(xWorld, yWorld);

%% 2. Define ROI Mask:  x > -130, x < 129, y > -110, y < 150
ROI = (Xworld > 0) & (Xworld < 265) & (Yworld > 50) & (Yworld < 330);
numROI = nnz(ROI);

%% 3. Mean-Subtract to Remove Static Background
mean_frame   = mean(etav, 3);
etameansub   = etav - mean_frame;

disp('Data loaded and mean-frame computed');

%% 4. Wavelet + Threshold + ROI Parameters
scales                 = 1:8;
selected_scale         = 8;
waveletThresholds      = linspace(-15, 70, 76);
numThresh              = numel(waveletThresholds);
frameIndices           = 38:338;
numFramesForAnalysis   = numel(frameIndices);
% Geometric filtering thresholds
eccentricity_threshold = 0.85;
solidity_threshold     = 0.6;

% Preallocate coverage-rate matrices
coverageRates_raw      = zeros(numFramesForAnalysis, numThresh);
coverageRates_filtered = zeros(numFramesForAnalysis, numThresh);

%% 5. Main Loop: Warp → CWT → Threshold → ROI→ Filter
for idx = 1:numFramesForAnalysis
    t_index = frameIndices(idx);
    disp(t_index);
    
    % Get the current frame (mean-subtracted)
    snapshot = etameansub(:, :, t_index);
    
    % Perform the continuous wavelet transform
    cwt_result = cwtft2(snapshot, 'Wavelet', 'mexh', 'Scales', selected_scale);
    wavelet_coefficients = cwt_result.cfs;
    
    for th = 1:numThresh
        current_thr = waveletThresholds(th);
        
        %----- RAW COVERAGE RATE COMPUTATION -----
        % Create a binary mask where the wavelet coefficient values exceed the threshold.
        mask_raw = wavelet_coefficients > current_thr;
        coverageRates_raw(idx, th) = sum(mask_raw(:)) / (numel(mask_raw)); %Subtracting the distortion volume
        
        %----- FILTERING FOR ECCENTRICITY AND SOLIDITY -----
        % %Identify connected components in the raw mask.
        % connected_components = bwconncomp(mask_raw);
        % if connected_components.NumObjects > 0
        %     % Compute region properties (including eccentricity and solidity).
        %     region_props = regionprops(connected_components, 'Area', 'Eccentricity', 'Solidity');
        %     % Identify regions that meet the filtering criteria.
        %     validIdx = find([region_props.Eccentricity] < eccentricity_threshold & ...
        %                       [region_props.Solidity] > solidity_threshold);
        %     % Create a binary mask of valid regions.
        %     % labelmatrix converts connected components to a label image.
        %     eccentric_regions = ismember(labelmatrix(connected_components), validIdx);
        % else
        %     % If no regions are found, the filtered mask is all false.
        %     eccentric_regions = false(size(mask_raw));
        % end

        % Compute the filtered coverage rate (fraction of pixels in valid regions).
        %coverageRates_filtered(idx, th) = sum(eccentric_regions(:)) / numel(eccentric_regions);
    end
end

%% 6. Plot Coverage Rates vs. Threshold

meanCoverageRate_raw = mean(coverageRates_raw, 1);
meanCoverageRate_filtered = mean(coverageRates_filtered, 1);

hfig = figure;
plot(waveletThresholds, meanCoverageRate_raw, 'LineWidth', 2, 'DisplayName', 'All structures');
hold on;
%plot(waveletThresholds, meanCoverageRate_filtered, 'k', 'LineWidth', 2, 'DisplayName', 'Filtered structures');
%plot(waveletThresholds, meanCoverageRate_raw, 'k+', 'LineWidth', 2);
xline(50, 'k--', 'LineWidth',1.5, 'HandleVisibility','off');
xlabel('${W}_{\rm thr}$', Interpreter='latex');
ylabel('$A(W > W_{\rm thr})/A_{\rm tot}$');
grid on;
xlim([-15, 70]);
ylim([-0.01,1.03]);

% Set additional plotting properties
fname = 'output/waveletThresholds_scale8_Wthr50_filtered';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.55; % Height-width ratio
set(findall(hfig,'-property','FontSize'),'FontSize',20); % Adjust font size
legend('Location','north', 'FontSize', 18);
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


%% 3. Visualize One Warped Frame with ROI Overlay
% Pick a representative frame index, e.g. frame 38
t_vis = 38;
snap_vis = etav(:,:,t_vis) - mean(etav,3);
warped_vis = imwarp(snap_vis, tform, 'OutputView', outputView);

figure;
imshow(warped_vis, [], 'XData', outputView.XWorldLimits, 'YData', outputView.YWorldLimits);
axis on; xlabel('X (world units)'); ylabel('Y (world units)');
title(sprintf('Warped Frame %d with ROI', t_vis));
hold on;
rectPos = [0, 50, 265-0, 330-50];  % [x_min, y_min, width, height]
rectangle('Position', rectPos, 'EdgeColor', 'g', 'LineWidth', 2);
