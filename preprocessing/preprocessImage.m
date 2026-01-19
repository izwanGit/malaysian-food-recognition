%% PREPROCESS IMAGE - Main Pre-processing Pipeline
% Applies comprehensive pre-processing to food images for analysis
%
% Syntax:
%   processedImg = preprocessImage(img)
%   processedImg = preprocessImage(img, targetSize)
%   [processedImg, originalSize] = preprocessImage(img)
%
% Inputs:
%   img        - RGB image (any size)
%   targetSize - [height, width] for output (default: [512, 512])
%
% Outputs:
%   processedImg  - Pre-processed RGB image
%   originalSize  - Original image dimensions [height, width]
%
% Processing Steps:
%   1. Resize to target dimensions (maintains consistency)
%   2. Convert to double precision for processing
%   3. Apply histogram stretching for contrast enhancement
%   4. Apply noise reduction filter
%   5. Convert back to uint8

function [processedImg, originalSize] = preprocessImage(img, targetSize)
    %% Input validation
    if nargin < 2
        targetSize = [512, 512];  % Default target size
    end
    
    % Validate image input
    if isempty(img)
        error('preprocessImage:EmptyInput', 'Input image is empty');
    end
    
    % Handle file path input
    if ischar(img) || isstring(img)
        if ~exist(img, 'file')
            error('preprocessImage:FileNotFound', 'Image file not found: %s', img);
        end
        img = imread(img);
    end
    
    % Convert grayscale to RGB if needed
    if size(img, 3) == 1
        img = repmat(img, 1, 1, 3);
    end
    
    % Store original size
    originalSize = [size(img, 1), size(img, 2)];
    
    %% Step 1: Resize image to target dimensions
    resizedImg = imresize(img, targetSize);
    
    %% Step 2: Convert to double for processing
    doubleImg = im2double(resizedImg);
    
    %% Step 3: A++ Color Correction & Enhancement
    try
        % 3a. Automatic White Balance (Gray World Assumption)
        % Corrects for indoor lighting (common in hawker centers)
        illum = illumgray(doubleImg);
        wbImg = chromadapt(doubleImg, illum, 'ColorSpace', 'linear-rgb');
    catch
        wbImg = doubleImg; % Fallback
    end
    
    % 3b. Contrast Enhancement (CLAHE in Lab Space)
    % Enhances local details without amplifying noise in flat regions
    labImg = rgb2lab(wbImg);
    L = labImg(:,:,1) / 100;
    L = adapthisteq(L, 'NumTiles', [8 8], 'ClipLimit', 0.005);
    labImg(:,:,1) = L * 100;
    enhancedImg = lab2rgb(labImg);
    
    %% Step 4: Noise Reduction & Sharpening
    % 4a. Guided Filter (Edge-preserving smoothing) - better than median
    denoisedImg = imguidedfilter(enhancedImg);
    
    % 4b. Subtle Sharpening (Makes edges crisp for Segmentation/SVM)
    sharpImg = imsharpen(denoisedImg, 'Radius', 1, 'Amount', 0.8);
    
    %% Step 5: Convert back to uint8
    processedImg = im2uint8(sharpImg);
end
