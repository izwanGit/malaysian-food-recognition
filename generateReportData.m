%% GENERATE REPORT DATA - Create Tables/Images for Final Report
% Generates specific images and data tables required by the Rubric
%
% This script will:
% 1. Process a sample image
% 2. Save intermediate images for "Table 1" (Segmentation)
% 3. Calculate and display metrics for "Table 2" (Texture Features)
% 4. Save visualization to 'report_output' folder

function generateReportData(imagePath)
    %% Setup
    if nargin < 1
        % Default: Try to find a sample image
        datasetPath = fullfile(fileparts(mfilename('fullpath')), 'dataset', 'train');
        % Try finding a Nasi Lemak image
        sample = dir(fullfile(datasetPath, 'nasi_lemak', '*.jpg'));
        
        if isempty(sample)
            % Fallback if dataset not linked
            error('No image provided and dataset not found. Usage: generateReportData(''path/to/image.jpg'')');
        end
        imagePath = fullfile(sample(1).folder, sample(1).name);
    end
    
    outputDir = fullfile(fileparts(mfilename('fullpath')), 'report_output');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    fprintf('=== Generating Report Data ===\n');
    fprintf('Input Image: %s\n', imagePath);
    fprintf('Output Directory: %s\n\n', outputDir);
    
    %% Load and Process
    img = imread(imagePath);
    processedImg = preprocessImage(img);
    
    %% --- GENERATE TABLE 1 DATA (Segmentation Steps) ---
    fprintf('--- TABLE 1: Segmentation Stages ---\n');
    
    % Step 1: Original (Preprocessed)
    imwrite(processedImg, fullfile(outputDir, '1_original.jpg'));
    
    % Step 2: HSV Thresholding (Equivalent to "Sobel/Gradient" step in rubric flow)
    hsvMask = hsvThreshold(processedImg);
    imwrite(hsvMask, fullfile(outputDir, '2_hsv_threshold_mask.jpg'));
    
    % Step 3: Morphological Cleaning (Dilated/Filled/Erosion)
    % We decompose morphologyClean to show intermediate steps
    se = strel('disk', 3);
    
    % 3a. Opening (Erosion -> Dilation) ~ "Erosion"
    openedMask = imopen(hsvMask, se);
    imwrite(openedMask, fullfile(outputDir, '3_morph_open.jpg'));
    
    % 3b. Closing (Dilation -> Erosion) ~ "Dilated"
    closedMask = imclose(openedMask, se);
    imwrite(closedMask, fullfile(outputDir, '4_morph_close.jpg'));
    
    % 3c. Fill Holes ~ "Filled In Holes"
    filledMask = imfill(closedMask, 'holes');
    imwrite(filledMask, fullfile(outputDir, '5_morph_filled.jpg'));
    
    % 3d. Remove Small Regions ~ "Remove Small Region"
    finalMask = bwareaopen(filledMask, 1000); % 1000px threshold
    imwrite(finalMask, fullfile(outputDir, '6_final_mask.jpg'));
    
    % Step 4: Final Segmented Image
    segmentedImg = processedImg;
    for c = 1:3
        channel = segmentedImg(:,:,c);
        channel(~finalMask) = 0; % Black background
        segmentedImg(:,:,c) = channel;
    end
    imwrite(segmentedImg, fullfile(outputDir, '7_segmented_rgb.jpg'));
    
    fprintf('Saved 7 images for Table 1 to %s\n', outputDir);
    fprintf('  1. 1_original.jpg\n');
    fprintf('  2. 2_hsv_threshold_mask.jpg\n');
    fprintf('  3. 3_morph_open.jpg\n');
    fprintf('  4. 4_morph_close.jpg\n');
    fprintf('  5. 5_morph_filled.jpg\n');
    fprintf('  6. 6_final_mask.jpg\n');
    fprintf('  7. 7_segmented_rgb.jpg\n\n');
    
    %% --- GENERATE TABLE 2 DATA (Feature Extraction) ---
    fprintf('--- TABLE 2: Texture Feature Extraction ---\n');
    
    % 1. Original
    % (Already saved as 1_original.jpg)
    
    % 2. Binarization
    % (Saved as 2_hsv_threshold_mask.jpg or 6_final_mask.jpg)
    
    % 3. Segmented Image
    % (Saved as 7_segmented_rgb.jpg)
    
    % 4. Grayscale Image
    grayImg = rgb2gray(processedImg);
    imwrite(grayImg, fullfile(outputDir, 'table2_grayscale.jpg'));
    
    % 5. Region of Interest (ROI)
    % Crop to the bounding box of the food
    stats = regionprops(finalMask, 'BoundingBox');
    if ~isempty(stats)
        % Find largest region
        areas = regionprops(finalMask, 'Area');
        [~, idx] = max([areas.Area]);
        bbox = stats(idx).BoundingBox;
        roiImg = imcrop(grayImg, bbox);
        imwrite(roiImg, fullfile(outputDir, 'table2_roi.jpg'));
    else
        roiImg = grayImg;
        warning('No object detected for ROI crop');
    end
    
    % 6. Results (Mean, Std, Smoothness)
    % Calculate on the ROI or the masked region
    maskedGray = grayImg;
    maskedGray(~finalMask) = 0;
    
    % Only consider pixels inside the mask for stats
    foodPixels = double(grayImg(finalMask)) / 255;
    
    statMean = mean(foodPixels);
    statStd = std(foodPixels);
    statSmoothness = 1 - (1 / (1 + statStd^2));
    
    fprintf('--- COPY THESE VALUES TO REPORT TABLE 2 ---\n');
    fprintf('Mean:               %.4f\n', statMean);
    fprintf('Standard Deviation: %.4f\n', statStd);
    fprintf('Smoothness:         %.4f\n', statSmoothness);
    fprintf('-------------------------------------------\n\n');
    
    fprintf('=== Done ===\n');
end
