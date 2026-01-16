%% TEST SEGMENTATION - Unit Tests for Segmentation Module
% Tests the food segmentation functions
%
% Run with: run('tests/testSegmentation.m')

function testSegmentation()
    fprintf('\n=== Testing Segmentation Module ===\n\n');
    
    % Get project path
    testDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(testDir);
    addpath(genpath(projectRoot));
    
    % Create a synthetic test image with food-like colors
    testImg = zeros(512, 512, 3);
    % Add "food" region in center (brown/orange color)
    [X, Y] = meshgrid(1:512, 1:512);
    foodMask = ((X-256).^2 + (Y-256).^2) < 150^2;
    testImg(:,:,1) = 0.7 * foodMask;  % R
    testImg(:,:,2) = 0.4 * foodMask;  % G
    testImg(:,:,3) = 0.2 * foodMask;  % B
    testImg = testImg + 0.9 * ~foodMask;  % White background
    
    %% Test 1: HSV Thresholding
    fprintf('Test 1: HSV Thresholding... ');
    mask = hsvThreshold(testImg);
    assert(~isempty(mask), 'HSV threshold failed');
    assert(islogical(mask), 'Mask should be logical');
    fprintf('PASSED\n');
    
    %% Test 2: Morphology Cleaning
    fprintf('Test 2: Morphology Cleaning... ');
    % Create noisy mask
    noisyMask = mask;
    noisyMask(1:50, 1:50) = true;  % Add noise
    cleanMask = morphologyClean(noisyMask);
    assert(~isempty(cleanMask), 'Morphology clean failed');
    % Clean mask should have less noise
    fprintf('PASSED\n');
    
    %% Test 3: K-means Segmentation
    fprintf('Test 3: K-means Segmentation... ');
    testImg8 = im2uint8(testImg);
    labeledRegions = kmeansSegment(testImg8, foodMask, 3);
    assert(~isempty(labeledRegions), 'K-means segmentation failed');
    assert(max(labeledRegions(:)) <= 3, 'Too many clusters');
    fprintf('PASSED\n');
    
    %% Test 4: Full Segmentation Pipeline
    fprintf('Test 4: Full Segmentation Pipeline... ');
    [mask, labeled, segImg] = segmentFood(testImg8);
    assert(~isempty(mask), 'Full segmentation failed');
    assert(~isempty(labeled), 'Label matrix is empty');
    assert(size(segImg, 3) == 3, 'Segmented image should be RGB');
    fprintf('PASSED\n');
    
    %% Test 5: Real Image Segmentation
    fprintf('Test 5: Real Image Segmentation... ');
    datasetPath = fullfile(projectRoot, 'dataset', 'train', 'laksa');
    images = dir(fullfile(datasetPath, '*.jpg'));
    if ~isempty(images)
        realImg = imread(fullfile(datasetPath, images(1).name));
        realImg = imresize(realImg, [512, 512]);
        [realMask, ~, ~] = segmentFood(realImg);
        coverage = sum(realMask(:)) / numel(realMask);
        assert(coverage > 0.05, 'Food coverage too low');
        assert(coverage < 0.95, 'Food coverage too high');
        fprintf('PASSED (%.1f%% coverage)\n', coverage * 100);
    else
        fprintf('SKIPPED (no images)\n');
    end
    
    %% Summary
    fprintf('\n=== All Segmentation Tests PASSED ===\n');
end
