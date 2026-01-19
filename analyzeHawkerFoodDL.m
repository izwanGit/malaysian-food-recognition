%% ANALYZE HAWKER FOOD DL - Deep Learning Analysis Pipeline
% Main analysis pipeline using CNN (Deep Learning) instead of SVM
%
% Syntax:
%   results = analyzeHawkerFoodDL(img)
%
% Inputs:
%   img - RGB image or path to image file
%
% Outputs:
%   results - Struct containing all analysis results

function results = analyzeHawkerFoodDL(img)
    %% Load image if path provided
    if ischar(img) || isstring(img)
        if ~exist(img, 'file')
            error('Image file not found: %s', img);
        end
        img = imread(img);
    end
    
    %% Setup
    projectRoot = fileparts(mfilename('fullpath'));
    addpath(genpath(projectRoot));
    
    %% Step 1: Preprocessing
    processedImg = preprocessImage(img);
    
    %% Step 2: Classification (CNN)
    try
        [foodClass, confidence] = classifyFoodCNN(processedImg);
    catch ME
        warning('CNN not available: %s. Falling back to SVM.', ME.message);
        [foodClass, confidence] = classifyFood(processedImg);
    end
    
    %% Step 3: Segmentation (use classical - it's reliable)
    [mask, ~, ~] = segmentFood(processedImg);
    
    %% Step 4: Create segmented image
    segmentedImg = processedImg;
    for c = 1:3
        channel = segmentedImg(:,:,c);
        channel(~mask) = 0;
        segmentedImg(:,:,c) = channel;
    end
    
    %% Step 5: Portion Estimation
    [portionRatio, portionLabel, foodArea] = estimatePortion(mask, foodClass, processedImg);
    
    %% Step 6: Calorie Calculation
    [calories, nutrition] = calculateCalories(foodClass, portionRatio);
    
    %% Package results
    results = struct();
    results.originalImage = img;
    results.processedImage = processedImg;
    results.segmentedImage = segmentedImg;
    results.foodMask = mask;
    results.foodClass = foodClass;
    results.confidence = confidence;
    results.portionRatio = portionRatio;
    results.portionLabel = portionLabel;
    results.foodArea = foodArea;
    results.calories = calories;
    results.nutrition = nutrition;
    results.classifierType = 'CNN';
end
