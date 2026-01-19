%% EXTRACT FEATURES - Combined Feature Extraction
% Extracts combined color and texture features from food images
%
% Syntax:
%   features = extractFeatures(img)
%   [features, featureNames] = extractFeatures(img)
%
% Inputs:
%   img - RGB image (preprocessed, uint8 or double)
%
% Outputs:
%   features     - 1 x N feature vector (N = 127 features)
%   featureNames - Cell array of feature names for interpretation
%
% Feature Composition:
%   - Color features: 108 features (RGB + HSV histograms + statistics)
%   - Texture features: 19 features (GLCM + Mean/Std/Smoothness)

function [features, featureNames] = extractFeatures(img)
    %% Input validation
    if isempty(img)
        error('extractFeatures:EmptyInput', 'Input image is empty');
    end
    
    % Handle file path input
    if ischar(img) || isstring(img)
        if ~exist(img, 'file')
            error('extractFeatures:FileNotFound', 'Image file not found: %s', img);
        end
        img = imread(img);
    end
    
    % Convert to double if needed
    if isa(img, 'uint8')
        img = im2double(img);
    end
    
    %% Extract color features
    [colorFeatures, colorNames] = extractColorFeatures(img);
    
    %% Extract texture features
    [textureFeatures, textureNames] = extractTextureFeatures(img);
    
    %% Extract HOG features (Shape)
    % Critical for distinguishing shapes (e.g. Satay sticks vs Popiah rolls)
    % CellSize [32 32] is optimized for speed and general shape capture
    grayImg = rgb2gray(img);
    resizedGray = imresize(grayImg, [256 256]); % Standardize for HOG
    [hogFeatures, visualization] = extractHOGFeatures(resizedGray, 'CellSize', [32 32]);
    
    % Generate HOG feature names
    hogNames = arrayfun(@(x) sprintf('HOG_%d', x), 1:length(hogFeatures), 'UniformOutput', false);

    %% Combine all features (color + texture + HOG)
    features = [colorFeatures, textureFeatures, hogFeatures];
    
    %% Combine feature names
    if nargout > 1
        featureNames = [colorNames, textureNames, hogNames];
    end
end
