%% EXTRACT TEXTURE FEATURES - GLCM Texture Feature Extraction
% Extracts Gray Level Co-occurrence Matrix (GLCM) texture features
%
% Syntax:
%   textureFeatures = extractTextureFeatures(img)
%   [textureFeatures, featureNames] = extractTextureFeatures(img)
%
% Inputs:
%   img - RGB image (double or uint8)
%
% Outputs:
%   textureFeatures - 1 x 19 feature vector
%   featureNames    - Cell array of feature names
%
% Texture Features (19 total):
%   GLCM (16): 4 orientations (0°, 45°, 90°, 135°) x 4 properties:
%     - Contrast: Measures local intensity variations
%     - Correlation: Measures linear dependency of gray levels
%     - Energy: Measures textural uniformity (sum of squared elements)
%     - Homogeneity: Measures closeness to diagonal elements
%   Statistical (3): Mean, Standard Deviation, Smoothness (Rubric requirement)

function [textureFeatures, featureNames] = extractTextureFeatures(img)
    %% Convert to grayscale
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    
    % Ensure uint8 for GLCM computation
    if ~isa(grayImg, 'uint8')
        grayImg = im2uint8(grayImg);
    end
    
    %% GLCM Parameters
    numGrayLevels = 32;  % Reduce to 32 levels for robustness
    offsets = [0 1; -1 1; -1 0; -1 -1];  % 0°, 45°, 90°, 135°
    
    %% Quantize gray levels
    grayImg = uint8(floor(double(grayImg) / 256 * numGrayLevels));
    
    %% Compute GLCM for each orientation
    glcms = graycomatrix(grayImg, 'Offset', offsets, ...
                         'NumLevels', numGrayLevels, ...
                         'GrayLimits', [0, numGrayLevels-1], ...
                         'Symmetric', true);
    
    %% Extract GLCM properties
    stats = graycoprops(glcms, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    
    %% Calculate Statistical Features (Rubric Requirement)
    % Mean, Standard Deviation, and Smoothness from grayscale image
    pixelValues = double(grayImg(:)) / 255;  % Normalize to [0,1]
    
    statMean = mean(pixelValues);
    statStd = std(pixelValues);
    statSmoothness = 1 - (1 / (1 + statStd^2));  % R = 1 - 1/(1+var)
    
    %% Arrange features
    % 16 GLCM features + 3 Statistical features = 19 Texture Features
    textureFeatures = [stats.Contrast, stats.Correlation, ...
                       stats.Energy, stats.Homogeneity, ...
                       statMean, statStd, statSmoothness];
    
    %% Generate feature names
    if nargout > 1
        orientations = {'0', '45', '90', '135'};
        properties = {'Contrast', 'Correlation', 'Energy', 'Homogeneity'};
        
        featureNames = cell(1, 19);
        idx = 1;
        
        % GLCM Names
        for p = 1:4
            for o = 1:4
                featureNames{idx} = sprintf('GLCM_%s_%sdeg', properties{p}, orientations{o});
                idx = idx + 1;
            end
        end
        
        % Statistical Names
        featureNames{17} = 'Intensity_Mean';
        featureNames{18} = 'Intensity_Std';
        featureNames{19} = 'Intensity_Smoothness';
    end
end
