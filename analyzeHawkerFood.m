%% ANALYZE HAWKER FOOD - Main Analysis Pipeline
% Complete pipeline for Malaysian hawker food recognition and calorie estimation
%
% Syntax:
%   results = analyzeHawkerFood(imagePath)
%   results = analyzeHawkerFood(img)
%
% Inputs:
%   imagePath/img - Path to image file or RGB image array
%
% Outputs:
%   results - Struct containing:
%             .originalImage   - Original input image
%             .processedImage  - Pre-processed image
%             .foodClass       - Predicted food class
%             .mode            - Classifier mode used ('svm' or 'cnn')
%             .confidence      - Classification confidence
%             .mask            - Food region mask
%             .segmentedImage  - Segmentation visualization
%             .portionRatio    - Portion size ratio
%             .portionLabel    - Portion size description
%             .calories        - Estimated calories
%             .nutrition       - Full nutritional breakdown
%             .processingTime  - Total processing time (seconds)
%
% Example:
%   results = analyzeHawkerFood('path/to/nasi_lemak.jpg', 'svm');
%   results = analyzeHawkerFood('path/to/nasi_lemak.jpg', 'cnn');
%   fprintf('Food: %s (%.1f%% confident)\n', results.foodClass, results.confidence*100);
%   fprintf('Calories: %d kcal (%s portion)\n', results.calories, results.portionLabel);

function results = analyzeHawkerFood(input, mode)
    tic;  % Start timing
    
    if nargin < 2
        mode = 'svm';
    end
    
    %% Step 1: Load image
    fprintf('=== Malaysian Hawker Food Analysis ===\n');
    fprintf('Mode: %s\n\n', upper(mode));
    fprintf('Step 1: Loading image...\n');
    
    if ischar(input) || isstring(input)
        if ~exist(input, 'file')
            error('analyzeHawkerFood:FileNotFound', 'Image file not found: %s', input);
        end
        originalImage = imread(input);
        fprintf('  Loaded: %s\n', input);
    else
        originalImage = input;
        fprintf('  Using provided image array\n');
    end
    
    fprintf('  Image size: %d x %d x %d\n\n', size(originalImage, 1), ...
            size(originalImage, 2), size(originalImage, 3));
    
    %% Step 2: Pre-process image
    fprintf('Step 2: Pre-processing image...\n');
    processedImage = preprocessImage(originalImage);
    fprintf('  Applied: Resize, histogram stretch, noise filter\n\n');
    
    %% Step 3: Classify food type
    fprintf('Step 3: Classifying food type (%s)...\n', upper(mode));
    try
        [foodClass, confidence, allScores] = classifyFood(processedImage);
        fprintf('  Prediction: %s\n', strrep(foodClass, '_', ' '));
        fprintf('  Confidence: %.1f%%\n\n', confidence * 100);
    catch ME
        warning('Classification failed: %s\nUsing default class.', ME.message);
        foodClass = 'mixed_rice';
        confidence = 0.5;
        allScores = struct();
    end
    
    %% Step 4: Segment food region
    fprintf('Step 4: Segmenting food region...\n');
    [mask, labeledRegions, segmentedImage] = segmentFood(processedImage);
    foodArea = sum(mask(:));
    totalArea = numel(mask);
    fprintf('  Food coverage: %.1f%% of image\n\n', foodArea / totalArea * 100);
    
    %% Step 5: Estimate portion size
    fprintf('Step 5: Estimating portion size...\n');
    [portionRatio, portionLabel, areaPixels] = estimatePortion(mask, foodClass, processedImage);
    fprintf('  Portion ratio: %.2f\n', portionRatio);
    fprintf('  Portion label: %s\n\n', portionLabel);
    
    %% Step 6: Calculate calories
    fprintf('Step 6: Calculating calories...\n');
    [calories, nutrition] = calculateCalories(foodClass, portionRatio);
    fprintf('  Base calories: %d kcal\n', foodDatabase(foodClass).baseCalories);
    fprintf('  Adjusted calories: %d kcal\n\n', calories);
    
    %% Compile results
    processingTime = toc;
    
    results = struct();
    results.originalImage = originalImage;
    results.processedImage = processedImage;
    results.foodClass = foodClass;
    results.foodDisplayName = strrep(foodClass, '_', ' ');
    results.confidence = confidence;
    results.allScores = allScores;
    results.mask = mask;
    results.labeledRegions = labeledRegions;
    results.segmentedImage = segmentedImage;
    results.portionRatio = portionRatio;
    results.portionLabel = portionLabel;
    results.areaPixels = areaPixels;
    results.calories = calories;
    results.nutrition = nutrition;
    results.processingTime = processingTime;
    
    %% Print summary
    fprintf('=== Analysis Complete ===\n');
    fprintf('  Food: %s\n', capitalizeFirst(strrep(foodClass, '_', ' ')));
    fprintf('  Confidence: %.1f%%\n', confidence * 100);
    fprintf('  Portion: %s (%.2fx)\n', portionLabel, portionRatio);
    fprintf('  Calories: %d kcal\n', calories);
    fprintf('  Processing time: %.2f seconds\n', processingTime);
    fprintf('========================\n');
end

%% Helper function
function str = capitalizeFirst(str)
    if ~isempty(str)
        words = strsplit(str, ' ');
        for i = 1:length(words)
            if ~isempty(words{i})
                words{i}(1) = upper(words{i}(1));
            end
        end
        str = strjoin(words, ' ');
    end
end
