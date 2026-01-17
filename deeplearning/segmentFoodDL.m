%% SEGMENT FOOD DEEP LEARNING - DeepLabv3+ Semantic Segmentation
% Performs food segmentation using pretrained DeepLabv3+ model
%
% This script demonstrates how deep learning (DeepLabv3+) can be used
% for semantic segmentation and compared with classical methods
%
% Syntax:
%   mask = segmentFoodDL(img)
%   [mask, labeledImage] = segmentFoodDL(img)
%
% Requirements:
%   - Deep Learning Toolbox
%   - Computer Vision Toolbox
%
% Note: If DeepLabv3+ is not available, this falls back to a
%       simplified semantic segmentation approach

function [mask, labeledImage] = segmentFoodDL(img)
    %% Load or initialize model
    persistent dlModel
    
    if isempty(dlModel)
        fprintf('Initializing Deep Learning Segmentation Model...\n');
        
        % Try to load pretrained DeepLabv3+
        try
            % Check if deeplabv3plus is available
            if exist('deeplabv3plusLayers', 'file')
                % Use pretrained resnet18 backbone
                inputSize = [512 512 3];
                numClasses = 2;  % Food vs Background
                
                lgraph = deeplabv3plusLayers(inputSize, numClasses, 'resnet18');
                dlModel.type = 'deeplabv3plus';
                dlModel.lgraph = lgraph;
                dlModel.inputSize = inputSize;
                fprintf('  DeepLabv3+ architecture loaded\n');
            else
                error('DeepLabv3+ not available');
            end
        catch
            % Fallback to simpler semantic segmentation using encoder-decoder
            fprintf('  DeepLabv3+ not available, using fallback segmentation\n');
            dlModel.type = 'fallback';
            dlModel.inputSize = [512 512 3];
        end
    end
    
    %% Load image if path provided
    if ischar(img) || isstring(img)
        img = imread(img);
    end
    
    %% Preprocess image
    originalSize = [size(img, 1), size(img, 2)];
    inputSize = dlModel.inputSize;
    
    % Resize to network input size
    imgResized = imresize(img, inputSize(1:2));
    
    % Normalize to [0, 1]
    if isa(imgResized, 'uint8')
        imgResized = im2double(imgResized);
    end
    
    %% Perform segmentation based on model type
    if strcmp(dlModel.type, 'deeplabv3plus')
        % If we have a trained DeepLabv3+ model, use it
        modelPath = fullfile(fileparts(mfilename('fullpath')), '..', 'models', 'foodSegmentationDL.mat');
        
        if exist(modelPath, 'file')
            loaded = load(modelPath, 'segNet');
            segNet = loaded.segNet;
            
            % Predict
            predictedLabels = semanticseg(imgResized, segNet);
            mask = predictedLabels == 'food';
        else
            % Model not trained yet, use fallback
            fprintf('DeepLabv3+ model not trained. Using enhanced classical method.\n');
            mask = segmentFoodDL_fallback(imgResized);
        end
    else
        % Use fallback segmentation (enhanced classical with DL-inspired approach)
        mask = segmentFoodDL_fallback(imgResized);
    end
    
    %% Post-process mask
    % Morphological cleanup
    mask = imclose(mask, strel('disk', 5));
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 500);
    
    % Resize back to original size
    mask = imresize(mask, originalSize, 'nearest');
    
    %% Create labeled image for visualization
    if nargout > 1
        labeledImage = uint8(mask) + 1;  % 1 = background, 2 = food
    end
end

%% Fallback segmentation using enhanced classical methods
function mask = segmentFoodDL_fallback(img)
    % This implements an enhanced segmentation approach that
    % mimics some deep learning concepts using classical methods
    
    % Convert to different color spaces
    hsv = rgb2hsv(img);
    lab = rgb2lab(img);
    
    % Multi-scale analysis (similar to deep learning receptive fields)
    scales = [1, 0.5, 0.25];
    combinedMask = zeros(size(img, 1), size(img, 2));
    
    for s = 1:length(scales)
        scale = scales(s);
        
        if scale < 1
            scaledImg = imresize(img, scale);
            scaledHSV = imresize(hsv, scale);
        else
            scaledImg = img;
            scaledHSV = hsv;
        end
        
        % HSV-based detection
        S = scaledHSV(:,:,2);
        V = scaledHSV(:,:,3);
        
        % Food typically has color (saturation) and isn't too dark/bright
        satMask = S > 0.1 & S < 0.95;
        valMask = V > 0.15 & V < 0.95;
        
        % Exclude white backgrounds (low sat, high val)
        bgMask = ~(S < 0.1 & V > 0.8);
        
        scaledMask = satMask & valMask & bgMask;
        
        % Resize back and accumulate
        if scale < 1
            scaledMask = imresize(double(scaledMask), [size(img,1), size(img,2)]);
        else
            scaledMask = double(scaledMask);
        end
        
        combinedMask = combinedMask + scaledMask;
    end
    
    % Threshold combined multi-scale result
    mask = combinedMask >= 2;  % At least 2 scales agree
    
    % Refinement using superpixels (if available)
    try
        numSuperpixels = 200;
        [L, ~] = superpixels(img, numSuperpixels);
        
        % For each superpixel, vote based on mask coverage
        stats = regionprops(L, double(mask), 'MeanIntensity');
        refinedMask = false(size(mask));
        
        for i = 1:max(L(:))
            if stats(i).MeanIntensity > 0.5
                refinedMask(L == i) = true;
            end
        end
        
        mask = refinedMask;
    catch
        % Superpixels not available, use morphology instead
        mask = imopen(mask, strel('disk', 3));
    end
end
