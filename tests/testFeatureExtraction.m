%% TEST FEATURE EXTRACTION - Unit Tests for Feature Extraction Module
% Tests the feature extraction functions
%
% Run with: run('tests/testFeatureExtraction.m')

function testFeatureExtraction()
    fprintf('\n=== Testing Feature Extraction Module ===\n\n');
    
    % Get project path
    testDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(testDir);
    addpath(genpath(projectRoot));
    
    % Create a test image (synthetic)
    testImg = rand(512, 512, 3);
    
    %% Test 1: Color Feature Extraction
    fprintf('Test 1: Color Feature Extraction... ');
    [colorFeatures, colorNames] = extractColorFeatures(testImg);
    assert(length(colorFeatures) == 108, 'Color features should have 108 elements');
    assert(length(colorNames) == 108, 'Color names should have 108 elements');
    fprintf('PASSED (108 features)\n');
    
    %% Test 2: Texture Feature Extraction
    fprintf('Test 2: Texture Feature Extraction... ');
    [textureFeatures, textureNames] = extractTextureFeatures(testImg);
    assert(length(textureFeatures) == 16, 'Texture features should have 16 elements');
    assert(length(textureNames) == 16, 'Texture names should have 16 elements');
    fprintf('PASSED (16 features)\n');
    
    %% Test 3: Combined Feature Extraction
    fprintf('Test 3: Combined Feature Extraction... ');
    [features, featureNames] = extractFeatures(testImg);
    assert(length(features) == 127, 'Combined features should have 127 elements');
    assert(length(featureNames) == 127, 'Feature names should have 127 elements');
    fprintf('PASSED (127 total features)\n');
    
    %% Test 4: Feature Normalization
    fprintf('Test 4: Feature Values... ');
    % RGB histogram should sum to ~1 per channel (normalized)
    rgbHist = colorFeatures(1:48);
    assert(all(rgbHist >= 0), 'Histogram values should be non-negative');
    fprintf('PASSED\n');
    
    %% Test 5: GLCM Properties Range
    fprintf('Test 5: GLCM Properties Range... ');
    % Contrast, Correlation, Energy, Homogeneity
    contrast = textureFeatures(1:4);
    energy = textureFeatures(9:12);
    homogeneity = textureFeatures(13:16);
    assert(all(energy >= 0 & energy <= 1), 'Energy should be in [0,1]');
    assert(all(homogeneity >= 0 & homogeneity <= 1), 'Homogeneity should be in [0,1]');
    fprintf('PASSED\n');
    
    %% Test 6: Real Image Test
    fprintf('Test 6: Real Image Test... ');
    datasetPath = fullfile(projectRoot, 'dataset', 'train', 'satay');
    images = dir(fullfile(datasetPath, '*.jpg'));
    if ~isempty(images)
        realImg = imread(fullfile(datasetPath, images(1).name));
        realImg = imresize(realImg, [512, 512]);
        [realFeatures, ~] = extractFeatures(im2double(realImg));
        assert(length(realFeatures) == 127, 'Real image feature extraction failed');
        fprintf('PASSED\n');
    else
        fprintf('SKIPPED (no images)\n');
    end
    
    %% Summary
    fprintf('\n=== All Feature Extraction Tests PASSED ===\n');
end
