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
%   textureFeatures - 1 x 16 feature vector
%   featureNames    - Cell array of feature names
%
% GLCM Features (16 total):
%   4 orientations (0°, 45°, 90°, 135°) x 4 properties:
%   - Contrast: Measures local intensity variations
%   - Correlation: Measures linear dependency of gray levels
%   - Energy: Measures textural uniformity (sum of squared elements)
%   - Homogeneity: Measures closeness to diagonal elements

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
    
    %% Arrange features (all 4 orientations for each property)
    textureFeatures = [stats.Contrast, stats.Correlation, ...
                       stats.Energy, stats.Homogeneity];
    
    %% Generate feature names
    if nargout > 1
        orientations = {'0', '45', '90', '135'};
        properties = {'Contrast', 'Correlation', 'Energy', 'Homogeneity'};
        
        featureNames = cell(1, 16);
        idx = 1;
        for p = 1:4
            for o = 1:4
                featureNames{idx} = sprintf('GLCM_%s_%sdeg', properties{p}, orientations{o});
                idx = idx + 1;
            end
        end
    end
end
