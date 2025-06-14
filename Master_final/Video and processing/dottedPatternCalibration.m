% Load the video
videoFile = 'calibrationData\calibrationImage_manualCropping.mp4';
vidObj = VideoReader(videoFile);

%%
calibrationImage = vidObj.readFrame;

%% Display image and manually select 4 points
imagesc(calibrationImage);
title('Click on Four Known Circles (Corners)');
[x, y] = ginput(4); % User selects four points

%% Convert to grayscale and enhance contrast
I = rgb2gray(calibrationImage);
I = imadjust(I, stretchlim(I), []);
% Define the corresponding real-world points
% These should correspond to where the four selected points "should be" after correction
% Well, this was not the case here. However, worked anyways as the
% spatial calibration was done later
realWorldPoints = [0, 0; 840, 0; 840, 600; 0, 600]; % Adjust scale if necessary
% Compute the homography transformation
tform = fitgeotrans([x, y], realWorldPoints, 'projective');
% Apply the transformation to warp the image
rectifiedImage = imwarp(I, tform);

%% Display the original and rectified images for comparison
figure;
colormap('gray');
subplot(1,2,1);
imagesc(I);
title('Original Image');
subplot(1,2,2);
imagesc(rectifiedImage);
title('Rectified Image (After Calibration)');

%% Now apply this transform to the whole time series

%data = load('..\data\SZ_VFD10p5Hz_TimeResolved_Run1_30fps.mat');
%data = load ('..\data\SZ_VFD10p5Hz_TimeResolved_Run1_30fps_newMay4th.mat');
data = load('..\data\SZ_VFD10p5Hz_TimeResolved_Run1_30fps_25limit.mat');

%%
disp(data)

%%
ceilingVideo = data.filteredFramesGray;
calibratedVideo = zeros(793, 1128, 2346);

%%
imagesc(ceilingVideo{346})

%% Loop over each frame and apply the transformation
for i = 1:2346
    disp(i);
    frame = ceilingVideo{i};
    
    % Apply the transformation with specified OutputView
    calibratedFrame = imwarp(frame, tform);
    
    % Store the calibrated frame
    calibratedVideo(:,:,i) = calibratedFrame;
end

%%
figure;
imagesc(calibratedVideo(:,:,1));
title('Undistorted frame')
colormap('gray')
figure;
imagesc(ceilingVideo{1});
title('Original frame')
colormap (gray)

%%
timeIndeces = data.filteredTimeindeces;
timeStamps = data.filteredTimestamps;

%%
save('..\data\SZ_VFD10p5Hz_TimeResolved_Run1_30fps_25limit_May4th.mat', 'calibratedVideo', 'timeIndeces', "timeStamps", '-v7.3');
disp('Filtered grayscale frames and timestamps saved to data folder.');
