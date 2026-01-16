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
%   features     - 1 x N feature vector (N = 124 features)
%   featureNames - Cell array of feature names for interpretation
%
% Feature Composition:
%   - Color features: 108 features (RGB + HSV histograms + statistics)
%   - Texture features: 16 features (GLCM at 4 orientations)

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
    
    %% Combine all features
    features = [colorFeatures, textureFeatures];
    
    %% Combine feature names
    if nargout > 1
        featureNames = [colorNames, textureNames];
    end
end
