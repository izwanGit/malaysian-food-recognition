%% EXTRACT COLOR FEATURES - Color-based Feature Extraction
% Extracts color histogram and statistical features from food images
%
% Syntax:
%   colorFeatures = extractColorFeatures(img)
%   [colorFeatures, featureNames] = extractColorFeatures(img)
%
% Inputs:
%   img - RGB image (double, range [0,1])
%
% Outputs:
%   colorFeatures - 1 x 108 feature vector
%   featureNames  - Cell array of feature names
%
% Features (108 total):
%   - RGB histogram: 16 bins x 3 channels = 48 features
%   - HSV histogram: 16 bins x 3 channels = 48 features
%   - Channel statistics: mean + std x 6 channels = 12 features

function [colorFeatures, featureNames] = extractColorFeatures(img)
    %% Parameters
    numBins = 16;  % Number of histogram bins
    
    %% Ensure image is double
    if isa(img, 'uint8')
        img = im2double(img);
    end
    
    %% RGB Histogram Features
    rgbHist = zeros(1, numBins * 3);
    for c = 1:3
        channel = img(:,:,c);
        hist = imhist(channel, numBins);
        hist = hist / sum(hist);  % Normalize
        rgbHist((c-1)*numBins + 1 : c*numBins) = hist';
    end
    
    %% HSV Histogram Features
    hsvImg = rgb2hsv(img);
    hsvHist = zeros(1, numBins * 3);
    for c = 1:3
        channel = hsvImg(:,:,c);
        hist = imhist(channel, numBins);
        hist = hist / sum(hist);  % Normalize
        hsvHist((c-1)*numBins + 1 : c*numBins) = hist';
    end
    
    %% Statistical Features
    rgbStats = zeros(1, 6);
    for c = 1:3
        channel = img(:,:,c);
        rgbStats(c) = mean(channel(:));      % Mean
        rgbStats(c + 3) = std(channel(:));   % Std
    end
    
    hsvStats = zeros(1, 6);
    for c = 1:3
        channel = hsvImg(:,:,c);
        hsvStats(c) = mean(channel(:));      % Mean
        hsvStats(c + 3) = std(channel(:));   % Std
    end
    
    %% Combine all color features
    colorFeatures = [rgbHist, hsvHist, rgbStats, hsvStats];
    
    %% Generate feature names
    if nargout > 1
        featureNames = cell(1, length(colorFeatures));
        idx = 1;
        
        % RGB histogram names
        channels = {'R', 'G', 'B'};
        for c = 1:3
            for b = 1:numBins
                featureNames{idx} = sprintf('%s_hist_%d', channels{c}, b);
                idx = idx + 1;
            end
        end
        
        % HSV histogram names
        channels = {'H', 'S', 'V'};
        for c = 1:3
            for b = 1:numBins
                featureNames{idx} = sprintf('%s_hist_%d', channels{c}, b);
                idx = idx + 1;
            end
        end
        
        % RGB statistics names
        featureNames{idx} = 'R_mean'; idx = idx + 1;
        featureNames{idx} = 'G_mean'; idx = idx + 1;
        featureNames{idx} = 'B_mean'; idx = idx + 1;
        featureNames{idx} = 'R_std'; idx = idx + 1;
        featureNames{idx} = 'G_std'; idx = idx + 1;
        featureNames{idx} = 'B_std'; idx = idx + 1;
        
        % HSV statistics names
        featureNames{idx} = 'H_mean'; idx = idx + 1;
        featureNames{idx} = 'S_mean'; idx = idx + 1;
        featureNames{idx} = 'V_mean'; idx = idx + 1;
        featureNames{idx} = 'H_std'; idx = idx + 1;
        featureNames{idx} = 'S_std'; idx = idx + 1;
        featureNames{idx} = 'V_std';
    end
end
