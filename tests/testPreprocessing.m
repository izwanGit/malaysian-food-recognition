%% TEST PREPROCESSING - Unit Tests for Pre-processing Module
% Tests the image pre-processing functions
%
% Run with: run('tests/testPreprocessing.m')

function testPreprocessing()
    fprintf('\n=== Testing Pre-processing Module ===\n\n');
    
    % Get project path
    testDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(testDir);
    addpath(genpath(projectRoot));
    
    % Get a sample image
    datasetPath = fullfile(projectRoot, 'dataset', 'train', 'nasi_lemak');
    images = dir(fullfile(datasetPath, '*.jpg'));
    
    if isempty(images)
        images = dir(fullfile(datasetPath, '*.jpeg'));
    end
    if isempty(images)
        images = dir(fullfile(datasetPath, '*.png'));
    end
    
    if isempty(images)
        error('No test images found in dataset/train/nasi_lemak');
    end
    
    testImagePath = fullfile(datasetPath, images(1).name);
    fprintf('Using test image: %s\n\n', images(1).name);
    
    %% Test 1: Image Loading
    fprintf('Test 1: Image Loading... ');
    img = imread(testImagePath);
    assert(~isempty(img), 'Image loading failed');
    assert(size(img, 3) == 3, 'Image should be RGB');
    fprintf('PASSED\n');
    
    %% Test 2: Histogram Stretching
    fprintf('Test 2: Histogram Stretching... ');
    imgDouble = im2double(img);
    stretched = histogramStretch(imgDouble);
    assert(~isempty(stretched), 'Histogram stretch failed');
    assert(all(stretched(:) >= 0) && all(stretched(:) <= 1), 'Output out of range');
    fprintf('PASSED\n');
    
    %% Test 3: Noise Filter (Median)
    fprintf('Test 3: Noise Filter (Median)... ');
    filtered = noiseFilter(imgDouble, 'median', 3);
    assert(~isempty(filtered), 'Median filter failed');
    assert(isequal(size(filtered), size(imgDouble)), 'Size mismatch');
    fprintf('PASSED\n');
    
    %% Test 4: Noise Filter (Gaussian)
    fprintf('Test 4: Noise Filter (Gaussian)... ');
    filtered = noiseFilter(imgDouble, 'gaussian', 5);
    assert(~isempty(filtered), 'Gaussian filter failed');
    fprintf('PASSED\n');
    
    %% Test 5: Full Pre-processing Pipeline
    fprintf('Test 5: Full Pre-processing Pipeline... ');
    [processedImg, originalSize] = preprocessImage(img);
    assert(~isempty(processedImg), 'Pre-processing failed');
    assert(size(processedImg, 1) == 512, 'Output size incorrect');
    assert(size(processedImg, 2) == 512, 'Output size incorrect');
    assert(isa(processedImg, 'uint8'), 'Output type incorrect');
    fprintf('PASSED\n');
    
    %% Test 6: Pre-processing from Path
    fprintf('Test 6: Pre-processing from Path... ');
    processedImg2 = preprocessImage(testImagePath);
    assert(~isempty(processedImg2), 'Pre-processing from path failed');
    fprintf('PASSED\n');
    
    %% Summary
    fprintf('\n=== All Pre-processing Tests PASSED ===\n');
end
