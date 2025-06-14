clearvars;
clc;
%%
addpath('..\utils\readimx-v2.1.9-win64\readimx-v2.1.9-win64\');
%% Parameters
h = 1.8305;
%kernel = [1, 0, -1] / (2);
kernel = [-1, 8, 0, -8, 1] / (12); % 5-point
%kernel = [1, -9, 45, 0, -45, 9, -1] / (60); %7-point
%kernel = [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]; %9-point
dataDir = fullfile('..\data\PIV_10mm_3600\StereoPIV_MPd(2x64x64_50%ov)\');
vc7files = dir(fullfile(dataDir, '*.vc7'));
allCentroids = []; % To accumulate vortex core positions
tracks = {};
% Gaussian blur parameters
sigma_blur = 1;    % standard deviation in grid points;

%% Calculate the global mean flow
% nFiles = numel(vc7files);
% mean_U_arr = zeros(nFiles,1); % Preallocate a column vector
% mean_V_arr = zeros(nFiles,1);
% for f = 1:300
%     disp(f);
%     filePath = fullfile(vc7files(f).folder, vc7files(f).name);
%     data = readimx(filePath);
%     frameData = data.Frames{1};
%     % Extract velocity data
%     vec_data = create2DVec(frameData);
%     % Transposing to match physical coordinate grids
%     U0 = vec_data.U';
%     V0 = vec_data.V';
%     % Calculate the overall mean of U0 (all pixels)
%     mean_U_arr(f) = mean(U0(:));
%     mean_V_arr(f) = mean(V0(:));
% end
% % Calculate the overall mean U, V to subtract from all frames
% mean_U = mean(mean_U_arr);
% mean_V = mean(mean_V_arr);

%% Preallocate cell array for storing lambda2_approx for each frame
% Adjust the indexing if needed; here, we assume frames from 14 to 299.
% numFrames = 299 - 14 + 1;
% lambda2_series = cell(1, numFrames);
frameStart = 14;
frameEnd = 3600;
nFrames = frameEnd - frameStart;
avgVorticity = nan(nFrames,1);       % pre‐allocate
timeVec      = nan(nFrames,1);       % pre‐allocate
beta_t      = nan(nFrames,1);       % pre‐allocate

threshold = -6;

%%
disp(47/15-1/15)

%% Process each file
% outputVideo=VideoWriter('lambda2-field3.mp4', 'MPEG-4');
% outputVideo.FrameRate=5;
% outputVideo.Quality = 100;
% open(outputVideo);

for f = 14:1000
    fprintf('Processing frame: %d \n', f);
    filePath = fullfile(vc7files(f).folder, vc7files(f).name);
    data = readimx(filePath);
    frameData = data.Frames{1};
    % %Extract velocity data
    vec_data = create3DVec(frameData);
    % Transposing
    % U0 = vec_data.U .* double((frameData.Components{6}.Planes{1}));
    % V0 = vec_data.V .* double((frameData.Components{6}.Planes{1}));
    % W0 = vec_data.W;
    mask = frameData.Components{16}.Planes{1} == 0; % Identify masked areas
    U0 = vec_data.U .* double((frameData.Components{16}.Planes{1}));
    V0 = vec_data.V .* double((frameData.Components{16}.Planes{1}));
    W0 = vec_data.W;
    % Set masked regions to NaN
    U0(mask) = NaN;
    V0(mask) = NaN;
    W0(mask) = NaN;
    X = vec_data.X;
    Y = vec_data.Y;
    U0 = U0';
    V0 = V0';
    W0 = W0';
    X = X';
    Y = Y';
    X = X;% - X(1,6); %normalizing coordinates
    Y = Y;% - Y(146);
    mask_vel = U0 >= 0.05;
    U0(mask_vel) = U0(mask_vel) - 0.24739;
    V0(mask_vel) = V0(mask_vel) - 7.3139E-3;

    % --- NEW: Gaussian blur each component ---
    U0 = imgaussfilt(U0, sigma_blur, 'Padding', 'replicate', 'FilterSize',3);
    V0 = imgaussfilt(V0, sigma_blur, 'Padding', 'replicate', 'FilterSize',3);

    % Compute spatial derivatives using 5-point central differences
    dx = (X(1,2) - X(1,1))*1e-3; % spacing in the x-direction
    dy = (Y(2,1) - Y(1,1))*1e-3; % spacing in the y-direction
    dU_dx = imfilter(U0, kernel, 'replicate', 'conv') / dx;
    dU_dy = imfilter(U0, kernel', 'replicate', 'conv') / dy;
    dV_dx = imfilter(V0, kernel, 'replicate', 'conv') / dx;
    dV_dy = imfilter(V0, kernel', 'replicate', 'conv') / dy;
    dW_dx = imfilter(W0, kernel, 'replicate', 'conv') / dx;
    dW_dy = imfilter(W0, kernel', 'replicate', 'conv') / dy;
    % Compute approximate lambda2 field in 2D
    lambda2_approx = dU_dx.^2 + dV_dx .* dU_dy;
    %lambda2_approx = dV_dy.^2 + dU_dy .* dV_dx; %the other eignenvalue for 2D flow
    % Create binary mask for vortex candidates (threshold adjustable)
    negativeRegions = lambda2_approx < threshold;
    % Identify connected components and filter small regions (area > n pixels)
    cc = bwconncomp(negativeRegions, 8);
    stats = regionprops(cc, 'Area', 'Centroid');
    validStats = stats([stats.Area] >= 1);

    % Store the computed lambda2_approx.
    %lambda2_series{f - 13} = lambda2_approx;  % Adjust index: frame 14 maps to index 1
    
    if ~isempty(validStats)
        % Get centroids in pixel coordinates [x, y]
        centroids = cat(1, validStats.Centroid);
        [nrows, ncols] = size(U0);
        % Convert to physical coordinates
        physX = interp2(1:ncols, 1:nrows, X, centroids(:,1), centroids(:,2));
        physY = interp2(1:ncols, 1:nrows, Y, centroids(:,1), centroids(:,2));
        % Store the physical centroids with their respective timestamp (frame number)
        timestamps = f * ones(size(physX)); % Store frame number
        physicalCentroids = [physX, physY, timestamps];
        currentCentroids = [physX, physY];
        % Append to the accumulated vortex core positions.
        allCentroids = [allCentroids; physicalCentroids];
    else
        currentCentroids = [];
    end

    % Compute vorticity_z
    vorticity_z = dV_dx - dU_dy;
    % Compute enstrophy (square of vorticity_z)
    enstrophy = vorticity_z.^2;

    % calculate horizontal divergence
    div_h = dU_dx + dV_dy;
    beta2_field = div_h.^2;

    % Calculate trace and determinant of the 2x2 velocity gradient tensor
    traceJ = dU_dx + dV_dy;
    detJ = dU_dx .* dV_dy - dU_dy .* dV_dx;
    
    % Compute the discriminant
    D = traceJ.^2 - 4 .* detJ;
    
    % Initialize swirling strength field (lambda_ci)
    swirling_strength = zeros(size(D));

    % Only compute swirling strength where the discriminant is negative (indicative of swirling motion)
    idx = D < 0;
    swirling_strength(idx) = 0.5 * sqrt(-D(idx));

    S_sq = dU_dx.^2 + dV_dy.^2 + 2*(0.5*(dU_dy + dV_dx)).^2;
    Omega_sq = 2*(0.5*(dU_dy - dV_dx)).^2;
    Q = 0.5 * (Omega_sq - S_sq);

    % Store the spatial mean of the absolute vorticity
    avgVorticity(f) = mean(abs(vorticity_z(:)), 'omitnan');
    
    %store the rms divergence field
    beta_t(f) = mean(beta2_field(:), 'omitnan');

    % Compute time in seconds (for frame f at 15 Hz)
    timeVec(f) = (f - 1)/15;       % since t = (frame-1)/fps

    % Snippet for visualising field and saving to file
    % hfig=figure;
    % colormap gray;
    % contourf(X, Y, lambda2_approx, 50 ,'LineStyle','none');
    % hold on;
    % if ~isempty(currentCentroids)
    %     plot(currentCentroids(:,1), currentCentroids(:,2), 'ko', ...
    %          'MarkerSize', 30, 'LineWidth', 2);
    % end
    % clim([min(lambda2_approx(:)), 0]);
    % xlabel('$x_1$ [mm]');
    % ylabel('$x_2$ [mm]');
    % 
    % % Set additional plotting properties
    % fname = 'output/lambda2Example';
    % picturewidth = 20; % Set figure width in centimeters
    % hw_ratio = 0.75; % Height-width ratio
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
    % % Capture and write only this frame’s plot
    % % Turn off all axes decorations
    % set(gca, 'Visible', 'off');
    % 
    % % Stretch the axes to fill the figure
    % %set(gca, 'Position', [0 0 1 1]);
    % 
    % drawnow;  % make sure it’s rendered
    % 
    % % Grab only the axes content (no borders, titles, ticks, etc.)
    % frame = getframe(gca);
    % % Rotate it 90° CCW and crop back to the original size
    % rotFrame = imrotate(frame.cdata, -90, 'crop');
    % writeVideo(outputVideo, rotFrame);
    % close(hfig);

    % % Normalize for visualization (optional but helps for contrast)
    % lambda2_vis = lambda2_approx;
    % lambda2_vis(isnan(lambda2_vis)) = 0;  % Replace NaNs to visualize
    % 
    % % Create figure (without displaying)
    % fig = figure('Visible', 'off');
    % %imagesc(lambda2_vis);
    % imagesc(X(1, :), Y(:, 1), lambda2_vis);
    % set(gca, 'YDir', 'normal'); % Flip Y-axis to match physical layout
    % axis equal tight;
    % %colormap('gray');
    % colorbar;
    % title(['\lambda_2 Approx - Frame ', num2str(f), ' timestamp: ', num2str(f/15-1/15)]);
    % %clim([min(lambda2_approx(:)), 0]);
    % clim([0, 200]);
    % 
    % % Capture the frame
    % frame = getframe(fig);
    % writeVideo(v, frame);
    % close(fig);  % Close the figure to avoid memory issues
end

%close(outputVideo)

%% Save the time series to a .mat file
% save('lambda2_series.mat', 'lambda2_series', 'X', 'Y');
% disp('Saved!')

% save('horDivergence_full_run2.mat', 'beta_t', 'timeVec');
% save('avgVorticity_full_run2.mat', 'avgVorticity', 'timeVec');
% disp('Saved successfully');

%%
figure;
plot(timeVec, avgVorticity/max(avgVorticity), 'LineWidth', 1.5, 'Displayname', 'VORT');
hold on;
plot(timeVec, beta_t/max(beta_t), DisplayName='DIV');
legend;
xlabel('Time \it{t} (s)', 'Interpreter','latex');
ylabel('Mean |$\omega_z$| (s$^{-1}$)', 'Interpreter','latex');
title('Average Spanwise Vorticity vs. Time', 'Interpreter','latex');
grid on;

%% Plot centroid positions on image U0
if ~isempty(allCentroids)
    % Display the background image U0
    figure;
    %imagesc(U0_raw);
    quiver(X, Y, U0, V0);
    colormap(gray); % Adjust colormap as needed
    %colorbar;
    hold on;
    % Overlay the centroid positions as red markers.
    % Assuming allCentroids is an N-by-2 array with [x, y] positions.
    plot(allCentroids(:,1), allCentroids(:,2), 'ko', 'MarkerSize', 20, 'LineWidth', 2);
    %scatter(allCentroids(:,1), allCentroids(:,2), 50, label = allCentroids(:,3));
    title('Vortex core detections');
    xlabel('X [mm]');
    ylabel('Y [mm]');
    axis image; % Maintain aspect ratio according to image dimensions
    hold off;
    else
    disp('No vortex cores detected in the provided files.');
end
%% Plot centroid positions on image U0 with lambda_2 as well
if ~isempty(allCentroids)
    % Display the background image U0
    hfig = figure;
    contourf(X, Y, lambda2_approx, 50 ,'LineStyle','none');
    clim([min(lambda2_approx(:)), 0]);
    colormap('gray');
    hold on;
    quiver(X, Y, U0, V0, 1, 'k', 'LineWidth',1);
    % Overlay the centroid positions as red markers.
    % Assuming allCentroids is an N-by-2 array with [x, y] positions.
    plot(allCentroids(:,1), allCentroids(:,2), 'ko', 'MarkerSize', 30, 'LineWidth', 2);
    %scatter(allCentroids(:,1), allCentroids(:,2), 50, label = allCentroids(:,3));
    %title('Vortex core detections');
    xlabel('$x_1$ [mm]');
    ylabel('$x_2$ [mm]');
    %set(gca, 'XTick', [], 'YTick', [])
    axis image; % Maintain aspect ratio according to image dimensions
    hold off;
    else
    disp('No vortex cores detected in the provided files.');
end
% Set additional plotting properties
fname = 'output/lambda2Example';
picturewidth = 20; % Set figure width in centimeters
hw_ratio = 0.75; % Height-width ratio
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

%% Plot the smaller regions
hfig= figure;
contourf(X, Y, lambda2_approx, 50, 'LineStyle','none');
clim([min(lambda2_approx(:)), 0]);
colormap('gray');
hold on;
% Plot the velocity field using quiver (with scaling factor if needed)
quiver(X, Y, U0, V0, 2, 'k');
set(gca, 'XTick', [], 'YTick', []);
% Define the region of interest
x_min = 88; x_max = 112;
y_min = 31; y_max = 48;
% Set the axes limits to zoom in
xlim([x_min, x_max]);
ylim([y_min, y_max]);
hold off;
% Set additional plotting properties
fname = 'output/reg4';
picturewidth = 10; % Set figure width in centimeters
hw_ratio = 0.75; % Height-width ratio
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
% figure;
% imagesc([min(X(:)) max(X(:))], [max(Y(:)) min(Y(:))], lambda2_approx);
% set(gca, 'YDir', 'normal'); % Ensure the y-axis is not flipped
% colorbar;
% xlabel('X [mm]');
% ylabel('Y [mm]');
% title('Lambda2 Field with Physical Coordinates');

% % Heatmap of detected vortices
% if ~isempty(allCentroids)
% % Define number of bins for X and Y (adjust as needed)
% numBinsX = 100;
% numBinsY = 100;
% % Use the physical coordinate ranges from allCentroids
% xEdges = linspace(min(allCentroids(:,1)), max(allCentroids(:,1)), numBinsX+1);
% yEdges = linspace(min(allCentroids(:,2)), max(allCentroids(:,2)), numBinsY+1);
% % Compute 2D histogram counts
% % Note: histcounts2 uses the first input as Y and second as X when plotting.
% heatmapCounts = histcounts2(allCentroids(:,2), allCentroids(:,1), yEdges, xEdges);
% % Create a disk kernel with a 5-pixel radius
% diskKernel = fspecial('disk', 4);
% % Convolve the histogram with the disk kernel to smooth it
% smoothedHeatmap = conv2(heatmapCounts, diskKernel, 'same');
% % Plot the smoothed heatmap
% figure;
% imagesc(xEdges, yEdges, smoothedHeatmap);
% set(gca, 'YDir', 'normal'); % Ensure that the y-axis increases upward
% colormap('hot');
% colorbar;
% title('Smoothed Vortex Core Positions Heatmap');
% xlabel('X [mm]');
% ylabel('Y [mm]');
% else
% disp('No vortex cores detected.');
% end

%% Lambda2 Field Contour Plot
% figure;
% contourf(X, Y, lambda2_approx, 50 ,'LineStyle','none');
% colorbar;
% colormap('gray'); % or preferred colormap
% clim([min(lambda2_approx(:)), 0]);
% xlabel('X [mm]');
% ylabel('Y [mm]');
% title('Lambda_2 Field (Contour Plot)');
% axis equal tight; % Maintains aspect ratio and tight bounds

%% Lambda_2 plotting for figures
% % Now plot using the new coordinates:
% hfig=figure;
% contourf(X, Y, lambda2_approx, 50, 'LineStyle', 'none');
% colorbar;
% colormap('gray'); % or preferred colormap
% clim([min(lambda2_approx(:)), 0]);
% hold on;
% % Overlay the centroid positions as markers.
% plot(allCentroids(:,1), allCentroids(:,2), 'ko', 'MarkerSize', 20, 'LineWidth', 1.5);
% xlabel('$x_1$ [mm]');
% ylabel('$x_2$ [mm]');
% %title('Lambda_2 Field (Contour Plot) with New Coordinate System');
% axis equal tight;
% % Set additional plotting properties
% fname = 'output/lambda2_thr8_t156';
% picturewidth = 18; % Set figure width in centimeters
% hw_ratio = 0.75; % Height-width ratio
% set(findall(hfig,'-property','FontSize'),'FontSize',22); % Adjust font size
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

%% Lambda2 Field 3D Surface Plot
figure;
surf(X, Y, lambda2_approx, 'EdgeColor', 'none'); % 'EdgeColor', 'none' removes grid lines
colormap('jet'); % Choose appropriate colormap
colorbar;
xlabel('X [mm]');
ylabel('Y [mm]');
zlabel('\lambda_2 magnitude');
title('3D Surface Plot of \lambda_2 Field');
view(45, 35); % Adjust viewing angle as needed
axis tight;

%% Compute and Plot Enstrophy (vorticity_z^2)
% Compute vorticity_z
vorticity_z = dV_dx - dU_dy;
% Compute enstrophy (square of vorticity_z)
enstrophy = vorticity_z.^2;
% 3D surface plot of Enstrophy
figure;
contourf(X, Y, enstrophy, 50, 'EdgeColor', 'none');
colormap('jet');
colorbar;
xlabel('X [mm]');
ylabel('Y [mm]');
%zlabel('Enstrophy (\omega_z^2)');
title('3D Surface Plot of Enstrophy');
%view(45, 35); % Adjust view angle as needed
axis tight;


%%

% Display the background image U0
hfig = figure;
contourf(X, Y, swirling_strength, 'LineStyle','none');
%clim([min(lambda2_approx(:)), 0]);
%colormap('gray');
hold on;
quiver(X, Y, U0, V0, 1, 'k', 'LineWidth',1);
% Overlay the centroid positions as red markers.
% Assuming allCentroids is an N-by-2 array with [x, y] positions.
plot(allCentroids(:,1), allCentroids(:,2), 'ko', 'MarkerSize', 20, 'LineWidth', 2);
%scatter(allCentroids(:,1), allCentroids(:,2), 50, label = allCentroids(:,3));
%title('Vortex core detections');
xlabel('$x_1$ [mm]');
ylabel('$x_2$ [mm]');
%set(gca, 'XTick', [], 'YTick', [])
axis image; % Maintain aspect ratio according to image dimensions
hold off;


%% Save allCentriods to .mat file

%save('allCentroids_14-3600_thresh6_area1_OrigCoords_blur_3x3.mat', 'allCentroids');
%disp('saved')