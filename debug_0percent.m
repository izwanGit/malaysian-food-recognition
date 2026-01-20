% DEBUG SCRIPT FOR 0% SEGMENTATION
clc; clear; close all;

% Path to the specific problematic image
imgPath = '/Users/izwan/.gemini/antigravity/brain/d1b56b10-f125-4cd9-98fa-3a2f82f97969/uploaded_image_1768918792835.png';
fprintf('DEBUG: Loading image: %s\n', imgPath);

try
    img = imread(imgPath);
    % Resize like the app does
    img = imresize(img, [768, 1024]); 
catch ME
    fprintf('Error loading image: %s\n', ME.message);
    return;
end

% Preprocessing
img = im2double(img);
fprintf('DEBUG: Image loaded and resized. Size: %s\n', mat2str(size(img)));

% Force class to 'popiah' as detected by the user's log
foodClass = 'popiah';
fprintf('DEBUG: Forcing foodClass = %s\n', foodClass);

% --- TRACE HSV THRESHOLD ---
addpath('segmentation');
fprintf('\n--- TRACING HSV THRESHOLD ---\n');

% Copy logic from hsvThreshold.m to trace internal variables
hsvImg = rgb2hsv(img);
H = hsvImg(:,:,1);
S = hsvImg(:,:,2);
V = hsvImg(:,:,3);

% Popiah params
satMin = 0.05; satMax = 0.9;
valMin = 0.15; valMax = 1.0;

satMask = (S >= satMin) & (S <= satMax);
valMask = (V >= valMin) & (V <= valMax);
fprintf('DEBUG: satMask pixels: %d\n', sum(satMask(:)));
fprintf('DEBUG: valMask pixels: %d\n', sum(valMask(:)));
fprintf('DEBUG: Combined Basic Mask pixels: %d\n', sum(satMask(:) & valMask(:)));

% Background Killer Trace
localStd = stdfilt(V, true(15));
whiteBgMask = (S < 0.05) & (V > 0.85) & (localStd < 0.03);
bgMask = imdilate(whiteBgMask, strel('disk', 2));
fprintf('DEBUG: Background Mask pixels: %d\n', sum(bgMask(:)));

baseMask = satMask & valMask & ~bgMask;
fprintf('DEBUG: After Background subtraction: %d\n', sum(baseMask(:)));

% Fallback check
if ~any(baseMask(:)) && any(satMask(:) & valMask(:))
    fprintf('DEBUG: Fallback triggered in HSV!\n');
end

% Run actual function
mask = hsvThreshold(img, foodClass);
fprintf('DEBUG: Final hsvThreshold output pixels: %d\n', sum(mask(:)));

if sum(mask(:)) == 0
    fprintf('CRITICAL FAILURE: hsvThreshold returned empty mask.\n');
else
    % --- TRACE SEGMENTFOOD ---
    fprintf('\n--- TRACING SEGMENTFOOD ---\n');
    cleanMask = mask;
    
    % Step 2: Morphology
    cleanMask = morphologyClean(cleanMask);
    fprintf('DEBUG: After Morphology Clean: %d\n', sum(cleanMask(:)));
    
    % Step 3.5: K-Means
    fprintf('DEBUG: Running K-Means...\n');
    % ... (Kmeans logic copy check if needed, but let's assume log output covers it) ...
    
    % Run actual segmentFood
    [finalMask, ~, ~] = segmentFood(img, foodClass);
    fprintf('DEBUG: Final segmentFood output pixels: %d\n', sum(finalMask(:)));
end
