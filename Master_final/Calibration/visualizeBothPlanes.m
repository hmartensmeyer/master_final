clear; clc;
% THIS MESSY SCRIPT HAS BEEN USED FOR PLANE CALIBRATION
% LOAD THE DATA AND RUN ONE OF THE TWO FUNCTIONS DEFINED IN THE BOTTOM OF
% THIS SCRIPT

% INPUT 1 SHOULD BE THE 'GROUND TRUTH' (I USED THE WAVELET COEFFS FROM THE CEILING), WHILE THE SECOND WAS PIV

%% -----------------------------------------------------------------------
data3 = load('lambda2_series_1.mat');        % data1.lambda2_series is a cell array.
data1 = load('wavelet_coeff_all.mat'); % wavelet of profilometry data
data2 = load('waveletCoefficients.mat');     % data2.wavelet_coeff_all is a 3D array.
%%
prof = load("profilometry_first860Values.mat");
%%
stamps = data1.stamps; %imprt stamps from filter_detect_track
X = prof.yGrid; % NOTE: HERE THE COORDINATES ARE INTERCHANGED TO MATCH PIV FIELD
Y = prof.xGrid;

%%
[pivPoints, ceilingPoints] = interactive_calibration_from_mat_prof(data1, data2, stamps, prof);

%%
X_phys = prof.xGrid(1, :);  % x coordinates from the first row
Y_phys = prof.yGrid(:, 1);  % y coordinates from the first column


%% Ensure enough matching points:
if size(pivPoints,1) >= 3 && size(ceilingPoints,1) >= 3
    % Computes a transformation from ceilingPoints to pivPoints.
    tform = fitgeotrans(ceilingPoints, pivPoints, 'affine');
    disp('Affine transformation computed:');
    disp(tform);
else
    error('Not enough matching points to compute an affine transformation.');
end

%%
% suppose ceilingPoints is N×2 = [row, col]
ceilingXY = [ ceilingPoints(:,1), ceilingPoints(:,2) ];  
pivXY     = [ pivPoints(:,1),     pivPoints(:,2) ];  

tform = fitgeotrans( ceilingXY, pivXY, 'affine' );

%%
disp(stamps(70)-stamps(37)-1/30)

%%
% Choose the frame indices to use for calibration
idxPIV = 101;      % For PIV data (lambda2_series)
idxCeiling = 70;  % For ceiling data (wavelet_coeff_all)

% Extract the frames:
% For the PIV data (lambda2_series is a cell array, so use curly braces)
%framePIV = data1.lambda2_series{idxPIV};
framePIV = data1.wavelet_coeff_all(:,:,idxPIV);

% For the ceiling data (assumed to be a 3D array)
frameCeiling = data2.wavelet_coeff_all(:, :, idxCeiling);

% Create an imref2d object based on the size of the rotated PIV frame.
outputRef = imref2d(size(framePIV));

% Set the world limits to reflect the physical coordinates.
% Since we rotated by -90°, the new horizontal axis (XWorldLimits) should correspond to the original Y range,
% and the new vertical axis (YWorldLimits) should correspond to the original X range.
% (Assuming data1.Y and data1.X are 146x194 arrays.)
%outputRef.XWorldLimits = [min(data1.Y(:)), max(data1.Y(:))];
%outputRef.YWorldLimits = [min(data1.X(:)), max(data1.X(:))];
outputRef.XWorldLimits = [min(prof.xGrid(:)), max(prof.xGrid(:))];
outputRef.YWorldLimits = [min(prof.yGrid(:)), max(prof.yGrid(:))];

% Apply the transformation to the ceiling frame.
% 'tform' must have been computed earlier
transformedFrame = imwarp(frameCeiling, tform, 'OutputView', outputRef);

% Display the transformed ceiling frame alongside the rotated PIV frame for comparison.
figure;

% Display the PIV frame (rotated) using the physical coordinates.
subplot(1,2,1);
% Use imagesc with XData and YData taken from the outputRef's world limits.
imagesc(Y_phys, X_phys, framePIV);
axis image;
title('PIV (Lambda2) Frame (Rotated)');
colormap gray;
colorbar;

% Display the transformed ceiling frame using the same physical coordinate mapping.
subplot(1,2,2);
imagesc(outputRef.YWorldLimits, outputRef.XWorldLimits, transformedFrame);
axis image;
title('Ceiling Frame Transformed to PIV Coordinates');
colormap gray;
colorbar;

%% THIS IS THE IMPORTANT PART HERE
disp(tform);
ell = tform;
save("transform_may10_test1_profilometry_firstchunk.mat", "ell");
disp('success');


% %% FROM HERE IT IS JUST TESTING

% 
% % Assume data2.wavelet_coeff_all is a 3D array: [height x width x numFrames]
% [numRows, numCols, numFrames] = size(data2.wavelet_coeff_all);
% 
% % Preallocate the output array (might want to convert to double if needed)
% transformedCeiling = zeros(outputRef.ImageSize(1), outputRef.ImageSize(2), numFrames);
% 
% for k = 1:numFrames
%     currentFrame = data2.wavelet_coeff_all(:, :, k);
%     transformedFrame = imwarp(currentFrame, tform, 'OutputView', outputRef);
%     transformedCeiling(:, :, k) = transformedFrame;
% end
% 
% disp('All ceiling frames have been transformed to match PIV coordinates.');
% 
% %%
% 
% % Check first 3 transformed frames against PIV
% for k = 38
%     figure;
%     subplot(1, 2, 1);
%     imagesc(Y_phys, X_phys, data1.lambda2_series{k}');
%     axis image;
%     title(['PIV Frame (Rotated)', num2str(round(double((k+13)/15-1/15),4))]);
%     colorbar;
%     colormap('gray');
% 
%     subplot(1, 2, 2);
%     imagesc(outputRef.XWorldLimits, outputRef.YWorldLimits, transformedCeiling(:,:,k+37));
%     axis image;
%     title(['Transformed Ceiling Frame ' num2str(stamps(k+37)-stamps(37)-1/30)]);
%     colorbar;
%     pause(0.5);
%     colormap('gray');
% end
% 
% 
% %%
% % 
% % save('calibrated_ceiling.mat', "transformedCeiling");
% % disp('successfully saved');
% 
% %%
% 
% xWidth_mm = max(data1.Y(:)) - min(data1.Y(:));
% yHeight_mm = max(data1.X(:)) - min(data1.X(:));
% 
% [heightCeil, widthCeil, ~] = size(data2.wavelet_coeff_all);
% pixelsPerMM_X = widthCeil / xWidth_mm;
% pixelsPerMM_Y = heightCeil / yHeight_mm;
% 
% outputRef = imref2d([round(pixelsPerMM_Y * yHeight_mm), round(pixelsPerMM_X * xWidth_mm)]);
% outputRef.XWorldLimits = [min(data1.Y(:)), max(data1.Y(:))];
% outputRef.YWorldLimits = [min(data1.X(:)), max(data1.X(:))];
% 
% numFrames = size(data2.wavelet_coeff_all, 3);
% transformedCeiling = zeros(outputRef.ImageSize(1), outputRef.ImageSize(2), numFrames);
% 
% for k = 1:numFrames
%     currentFrame = data2.wavelet_coeff_all(:, :, k);
%     transformedFrame = imwarp(currentFrame, tform, 'OutputView', outputRef);
%     transformedCeiling(:, :, k) = transformedFrame;
% end
% disp('done');
% 
% %%
% 
% % Choose a transformed ceiling frame to show
% frameToShow = 38;
% 
% figure;
% imagesc(outputRef.XWorldLimits, outputRef.YWorldLimits, transformedCeiling(:,:,frameToShow));
% axis image;
% set(gca, 'YDir', 'reverse');  % Ensure origin is in lower-left if needed
% xlabel('Y (mm)');
% ylabel('X (mm)');
% title(['Transformed Ceiling Frame #' num2str(frameToShow) ' (in PIV Coordinates)']);
% colorbar;
% 
% %%
% save("calibrated_ceiling_highres.mat", "transformedCeiling", '-v7.3');
% disp('done');
% 
% %% THIS IS THE IMPORTANT PART HERE
% disp(tform);
% ell = tform;
% save("transform_apr11_test1_firstchunk.mat", "ell");
% disp('success');
% 
% %% 1. Load video data (example):
% data = load('..\data\SZ_VFD10p5Hz_TimeResolved_Run1_30fps.mat');
% video = data.calibratedVideo;
% times = data.timeIndeces;
% stamps = data.timeStamps;
% 
% [height, width, numFrames] = size(video);
% disp('Video data loaded.');
% 
% %% 2. Load saved transformation
% load('transform_apr11_test1_firstchunk.mat', 'ell');  % ell is the transformation object
% 
% % Compute physical coordinate limits from data1.
% physicalXLimits = [min(data1.Y(:)), max(data1.Y(:))];  % Horizontal limits in mm.
% physicalYLimits = [min(data1.X(:)), max(data1.X(:))];  % Vertical limits in mm.
% 
% % 3. Define New Resolution in Physical Units
% newWidth = round(physicalXLimits(2) - physicalXLimits(1) + 1);
% newHeight = round(physicalYLimits(2) - physicalYLimits(1) + 1);
% 
% % Create a new spatial reference object with physical resolution.
% outputRef_physical = imref2d([newHeight, newWidth], physicalXLimits, physicalYLimits);
% 
% % 4. Preallocate a Cell Array for Transformed Frames at New Resolution
% transformedVideoCell_physical = cell(numFrames, 1);
% 
% % 5. Loop Over Each Frame and Apply the Transformation
% for t = 1:400
%     currentFrame = video(:, :, t);
%     % Apply the transformation with the new output reference (resampled to physical units)
%     transformedFrame = imwarp(currentFrame, ell, 'OutputView', outputRef_physical);
%     transformedVideoCell_physical{t} = transformedFrame;
% 
%     % Optionally display progress
%     if mod(t, 50) == 0
%         fprintf('Transformed frame %d of %d\n', t, 400);
%     end
% end
% 
% disp('All video frames have been transformed to physical resolution.');
% 
% % 6. Save the Transformed Video Series
% save('SZ_VFD10p5Hz_TimeResolved_Run1_30fps_calibrated_physical.mat', ...
%      'transformedVideoCell_physical', 'stamps', 'times', '-v7.3');
% 
% %%
% 
% test = load('SZ_VFD10p5Hz_TimeResolved_Run1_30fps_calibrated.mat');
% 
% %%
% 
% if isfield(test, 'outputRef')
%     outputRef = test.outputRef;
% else
%     % If outputRef was not saved, recreate it given calibration data. For example:
%     % (Replace these with known physical limits.)
%     physicalXLimits = [0, 330];  % for example, in millimeters
%     physicalYLimits = [0, 265];  % for example, in millimeters
%     [rows, cols] = size(test.transformedVideoCell{38});
%     outputRef = imref2d([rows, cols], physicalXLimits, physicalYLimits);
% end
% 
% % Now display the first frame using the physical coordinate limits
% figure;
% imagesc(outputRef.XWorldLimits, outputRef.YWorldLimits, test.transformedVideoCell{69});
% axis image;
% set(gca, 'YDir', 'reverse'); % Set YDir to normal if the origin at the lower-left
% xlabel('X [mm]');
% ylabel('Y [mm]');
% title('Transformed Frame with Real Coordinates');
% colorbar;


%%

function [pivPoints, ceilingPoints] = interactive_calibration_from_mat_prof(data1, data2, stamps, prof)
    % This function assumes:
    %   - data1.lambda2_series is a cell array of images (each an array)
    %   - data2.wavelet_coeff_all is a 3D array (rows x cols x frames)
    
    % Initialize the frame indices for both datasets.
    idx1 = 1;  % lambda2_series current frame index
    idx2 = 38;  % wavelet_coeff_all current frame index

    % Extract initial frames.
    %frame1 = data1.lambda2_series{idx1};
    %frame1 = data1.surfValues860(:, :, idx1);
    frame1 = flipud(fliplr( data1.wavelet_coeff_all(:,:,idx1)));
    frame1_rotated = imrotate(frame1, -90);
    frame2 = data2.wavelet_coeff_all(:, :, idx2);

    % Extract the physical coordinate vectors.
    % Assuming data1.X and data1.Y are 146x194, we extract vectors:
    %X_phys = data1.X(1, :);  % x coordinates from the first row
    %Y_phys = data1.Y(:, 1);  % y coordinates from the first column

    X_phys = prof.xGrid(1, :);  % x coordinates from the first row
    Y_phys = prof.yGrid(:, 1);  % y coordinates from the first column

    X_phys = X_phys';
    Y_phys = Y_phys';
    % X_phys = X_phys - X_phys(1,6);
    % Y_phys = Y_phys - Y_phys(146);

    % Initialize correspondence arrays.
    pivPoints = [];
    ceilingPoints = [];
    
    % Create the calibration figure.
    hFig = figure('Name', 'Interactive Calibration', 'NumberTitle', 'off', ...
                  'Position', [100, 100, 1400, 700]);
    
    % Create axes for each dataset.
    hAx1 = axes('Parent', hFig, 'Position', [0.05, 0.2, 0.35, 0.7]); % Left axis: lambda2_series
    hAx2 = axes('Parent', hFig, 'Position', [0.45, 0.2, 0.50, 0.7]); % Right axis: wavelet_coeff_all

    % Display the images using imagesc (native pixel coordinates are used automatically).
    hImg1 = imagesc(Y_phys, X_phys, frame1, 'Parent', hAx1);
    title(hAx1, sprintf('Profilometry (Timestamp %d)', ((idx1+35)/45-1/45)));
    axis(hAx1, 'image'); 
    %set(hAx1, 'YDir', 'reverse');
    %set(hAx1, 'XDir', 'reverse');
    minVal = min(frame1(:));  % or set a specific value
    %clim(hAx1, [minVal, 0]);   % All values > 0 will show as 0.
    colormap('gray');

    hImg2 = imagesc(frame2, 'Parent', hAx2);
    title(hAx2, sprintf('Wavelet Coefficients (Timestamp %d)', stamps(idx2) - stamps(37) - 1/30));
    axis(hAx2, 'image');
    set(hAx2, 'YDir', 'reverse');
    
    % Set up click callbacks to collect calibration points.
    set(hImg1, 'ButtonDownFcn', @clickPIV);
    set(hImg2, 'ButtonDownFcn', @clickCeiling);
    
    % Add navigation buttons for data1 (lambda2_series).
    uicontrol('Style', 'pushbutton', 'String', 'Prev Frame Lambda2',...
              'Position', [50, 20, 120, 30], 'Callback', @prevFrameLambda2);
    uicontrol('Style', 'pushbutton', 'String', 'Next Frame Lambda2',...
              'Position', [180, 20, 120, 30], 'Callback', @nextFrameLambda2);
    
    % Add navigation buttons for data2 (wavelet_coeff_all).
    uicontrol('Style', 'pushbutton', 'String', 'Prev Frame Wavelet',...
              'Position', [600, 20, 140, 30], 'Callback', @prevFrameWavelet);
    uicontrol('Style', 'pushbutton', 'String', 'Next Frame Wavelet',...
              'Position', [750, 20, 140, 30], 'Callback', @nextFrameWavelet);
    
    % Add a "Done" button to finish the calibration session.
    uicontrol('Style', 'pushbutton', 'String', 'Done', ...
              'Position', [650, 70, 100, 40], 'Callback', @doneCallback);
    
    % Block execution until the session is finished.
    uiwait(hFig);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Callback Functions (nested to share workspace)

    function clickPIV(~, ~)
        cp = get(hAx1, 'CurrentPoint');
        x = cp(1,1);
        y = cp(1,2);
        pivPoints = [pivPoints; x, y];
        hold(hAx1, 'on');
        plot(hAx1, x, y, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
        hold(hAx1, 'off');
        disp(['Lambda2 Series: Click at (', num2str(x), ', ', num2str(y), ')']);
    end

    function clickCeiling(~, ~)
        cp = get(hAx2, 'CurrentPoint');
        x = cp(1,1);
        y = cp(1,2);
        ceilingPoints = [ceilingPoints; x, y];
        hold(hAx2, 'on');
        plot(hAx2, x, y, 'go', 'MarkerSize', 10, 'LineWidth', 2);
        hold(hAx2, 'off');
        disp(['Wavelet Coefficients: Click at (', num2str(x), ', ', num2str(y), ')']);
    end

    function prevFrameLambda2(~, ~)
        if idx1 > 1
            idx1 = idx1 - 1;
            %frame1 = data1.surfValues860(:,:,idx1);
            frame1 = flipud(fliplr(data1.wavelet_coeff_all(:,:,idx1)));
            %frame1_rotated = imrotate(frame1, -90);
            set(hImg1, 'CData', frame1, 'XData', Y_phys, 'YData', X_phys);
            minVal = min(frame1(:));
            maxVal = max(frame1(:));
            %clim(hAx1, [minVal, 0]);
            colormap('gray');
            title(hAx1, sprintf('Profilometry (Timestamp %d)', ((idx1+35)/45-1/45)));
        end
    end
    
    function nextFrameLambda2(~, ~)
        if idx1 < numel(data1.wavelet_coeff_all)
            idx1 = idx1 + 1;
            %frame1 = data1.surfValues860(:,:,idx1);
            frame1 = flipud(fliplr(data1.wavelet_coeff_all(:,:,idx1)));
            %frame1_rotated = imrotate(frame1, -90);
            set(hImg1, 'CData', frame1, 'XData', Y_phys, 'YData', X_phys);
            %set(hAx1, 'YDir', 'reverse');
            minVal = min(frame1(:));
            maxVal = max(frame1(:));
            %clim(hAx1, [minVal, 0]);
            colormap('gray');
            title(hAx1, sprintf('Profilometry (Timestamp %d)', ((idx1+35)/45-1/45)));
        end
    end

    function prevFrameWavelet(~, ~)
        if idx2 > 1
            idx2 = idx2 - 1;
            frame2 = data2.wavelet_coeff_all(:, :, idx2);
            set(hImg2, 'CData', frame2);
            title(hAx2, sprintf('Wavelet Coefficients (Timestamp %.4f)', stamps(idx2) - stamps(37) - 1/30));
        end
    end

    function nextFrameWavelet(~, ~)
        if idx2 < size(data2.wavelet_coeff_all, 3)
            idx2 = idx2 + 1;
            frame2 = data2.wavelet_coeff_all(:, :, idx2);
            set(hImg2, 'CData', frame2);
            title(hAx2, sprintf('Wavelet Coefficients (Timestamp %.4f)', stamps(idx2) - stamps(37) - 1/30));
        end
    end

    function doneCallback(~, ~)
        uiresume(hFig);
        close(hFig);
    end

end

%%  

function [pivPoints, ceilingPoints] = interactive_calibration_from_mat(data1, data2, stamps)
    % This function assumes:
    %   - data1.lambda2_series is a cell array of images (each an array)
    %   - data2.wavelet_coeff_all is a 3D array (rows x cols x frames)
    
    % Initialize the frame indices for both datasets.
    idx1 = 1;  % lambda2_series current frame index
    idx2 = 38;  % wavelet_coeff_all current frame index

    % Extract initial frames.
    frame1 = data1.lambda2_series{idx1};
    frame1_rotated = imrotate(frame1, -90);
    frame2 = data2.wavelet_coeff_all(:, :, idx2);

    % Extract the physical coordinate vectors.
    % Assuming data1.X and data1.Y are 146x194, we extract vectors:
    X_phys = data1.X(1, :);  % x coordinates from the first row
    Y_phys = data1.Y(:, 1);  % y coordinates from the first column

    X_phys = X_phys';
    Y_phys = Y_phys';
    % X_phys = X_phys - X_phys(1,6);
    % Y_phys = Y_phys - Y_phys(146);

    % Initialize correspondence arrays.
    pivPoints = [];
    ceilingPoints = [];
    
    % Create the calibration figure.
    hFig = figure('Name', 'Interactive Calibration', 'NumberTitle', 'off', ...
                  'Position', [100, 100, 1400, 700]);
    
    % Create axes for each dataset.
    hAx1 = axes('Parent', hFig, 'Position', [0.05, 0.2, 0.35, 0.7]); % Left axis: lambda2_series
    hAx2 = axes('Parent', hFig, 'Position', [0.45, 0.2, 0.50, 0.7]); % Right axis: wavelet_coeff_all

    % Display the images using imagesc (native pixel coordinates are used automatically).
    hImg1 = imagesc(Y_phys, X_phys, frame1', 'Parent', hAx1);
    title(hAx1, sprintf('Lambda2 Series (Timestamp %d)', ((idx1+13)/15-1/15)));
    axis(hAx1, 'image'); 
    %set(hAx1, 'YDir', 'reverse');
    %set(hAx1, 'XDir', 'reverse');
    minVal = min(frame1(:));  % or set a specific
    clim(hAx1, [minVal, 0]);   % All values > 0 will show as 0.
    colormap('gray');

    hImg2 = imagesc(frame2, 'Parent', hAx2);
    title(hAx2, sprintf('Wavelet Coefficients (Timestamp %d)', stamps(idx2) - stamps(37) - 1/30));
    axis(hAx2, 'image');
    set(hAx2, 'YDir', 'reverse');
    
    % Set up click callbacks to collect calibration points.
    set(hImg1, 'ButtonDownFcn', @clickPIV);
    set(hImg2, 'ButtonDownFcn', @clickCeiling);
    
    % Add navigation buttons for data1 (lambda2_series).
    uicontrol('Style', 'pushbutton', 'String', 'Prev Frame Lambda2',...
              'Position', [50, 20, 120, 30], 'Callback', @prevFrameLambda2);
    uicontrol('Style', 'pushbutton', 'String', 'Next Frame Lambda2',...
              'Position', [180, 20, 120, 30], 'Callback', @nextFrameLambda2);
    
    % Add navigation buttons for data2 (wavelet_coeff_all).
    uicontrol('Style', 'pushbutton', 'String', 'Prev Frame Wavelet',...
              'Position', [600, 20, 140, 30], 'Callback', @prevFrameWavelet);
    uicontrol('Style', 'pushbutton', 'String', 'Next Frame Wavelet',...
              'Position', [750, 20, 140, 30], 'Callback', @nextFrameWavelet);
    
    % Add a "Done" button to finish the calibration session.
    uicontrol('Style', 'pushbutton', 'String', 'Done', ...
              'Position', [650, 70, 100, 40], 'Callback', @doneCallback);
    
    % Block execution until the session is finished.
    uiwait(hFig);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Callback Functions (nested to share workspace)

    function clickPIV(~, ~)
        cp = get(hAx1, 'CurrentPoint');
        x = cp(1,1);
        y = cp(1,2);
        pivPoints = [pivPoints; x, y];
        hold(hAx1, 'on');
        plot(hAx1, x, y, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
        hold(hAx1, 'off');
        disp(['Lambda2 Series: Click at (', num2str(x), ', ', num2str(y), ')']);
    end

    function clickCeiling(~, ~)
        cp = get(hAx2, 'CurrentPoint');
        x = cp(1,1);
        y = cp(1,2);
        ceilingPoints = [ceilingPoints; x, y];
        hold(hAx2, 'on');
        plot(hAx2, x, y, 'go', 'MarkerSize', 10, 'LineWidth', 2);
        hold(hAx2, 'off');
        disp(['Wavelet Coefficients: Click at (', num2str(x), ', ', num2str(y), ')']);
    end

    function prevFrameLambda2(~, ~)
        if idx1 > 1
            idx1 = idx1 - 1;
            frame1 = data1.lambda2_series{idx1};
            frame1_rotated = imrotate(frame1, -90);
            set(hImg1, 'CData', frame1', 'XData', Y_phys, 'YData', X_phys);
            minVal = min(frame1(:));
            clim(hAx1, [minVal, 0]);
            colormap('gray');
            title(hAx1, sprintf('Lambda2 Series (Timestamp %4.f)', (idx1+13)/15-1/15));
        end
    end
    
    function nextFrameLambda2(~, ~)
        if idx1 < numel(data1.lambda2_series)
            idx1 = idx1 + 1;
            frame1 = data1.lambda2_series{idx1};
            frame1_rotated = imrotate(frame1, -90);
            set(hImg1, 'CData', frame1', 'XData', Y_phys, 'YData', X_phys);
            minVal = min(frame1(:));
            clim(hAx1, [minVal, 0]);
            colormap('gray');
            title(hAx1, sprintf('Lambda2 Series (Timestamp %.4f)', (idx1+13)/15-1/15));
        end
    end

    function prevFrameWavelet(~, ~)
        if idx2 > 1
            idx2 = idx2 - 1;
            frame2 = data2.wavelet_coeff_all(:, :, idx2);
            set(hImg2, 'CData', frame2);
            title(hAx2, sprintf('Wavelet Coefficients (Timestamp %.4f)', stamps(idx2) - stamps(37) - 1/30));
        end
    end

    function nextFrameWavelet(~, ~)
        if idx2 < size(data2.wavelet_coeff_all, 3)
            idx2 = idx2 + 1;
            frame2 = data2.wavelet_coeff_all(:, :, idx2);
            set(hImg2, 'CData', frame2);
            title(hAx2, sprintf('Wavelet Coefficients (Timestamp %.4f)', stamps(idx2) - stamps(37) - 1/30));
        end
    end

    function doneCallback(~, ~)
        uiresume(hFig);
        close(hFig);
    end

end
