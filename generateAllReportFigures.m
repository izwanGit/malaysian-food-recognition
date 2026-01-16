%% GENERATE ALL REPORT FIGURES - Complete Report Visualization Suite
% Generates ALL figures, tables, and diagrams required for the report
%
% This script generates:
% 1. Table 1: Image Segmentation Results (7 images per row)
% 2. Table 2: Texture Feature Extraction Results (Mean, Std, Smoothness)
% 3. Feature Distribution Visualization
% 4. Color Histogram Comparison
% 5. GLCM Texture Visualization
% 6. Confusion Matrix (if model trained)
% 7. Per-class Accuracy Bar Chart
% 8. Processing Pipeline Visualization
%
% Usage:
%   generateAllReportFigures()               % Uses sample images
%   generateAllReportFigures('path/to/img')  % Uses specific image

function generateAllReportFigures(imagePath)
    %% Setup
    projectRoot = fileparts(mfilename('fullpath'));
    addpath(genpath(projectRoot));
    
    outputDir = fullfile(projectRoot, 'report_figures');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║        GENERATING ALL REPORT FIGURES AND TABLES              ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
    fprintf('Output Directory: %s\n\n', outputDir);
    
    %% Find sample images if not provided
    if nargin < 1 || isempty(imagePath)
        datasetPath = fullfile(projectRoot, 'dataset', 'train');
        foodClasses = {'nasi_lemak', 'roti_canai', 'satay', 'laksa', ...
                       'popiah', 'kaya_toast', 'mixed_rice'};
        
        % Try to find one image from each class
        sampleImages = {};
        for i = 1:length(foodClasses)
            classPath = fullfile(datasetPath, foodClasses{i});
            images = dir(fullfile(classPath, '*.jpg'));
            if ~isempty(images)
                sampleImages{end+1} = fullfile(classPath, images(1).name); %#ok<AGROW>
            end
        end
        
        if isempty(sampleImages)
            error('No dataset found. Please provide an image path or link the dataset.');
        end
        
        imagePath = sampleImages{1};
        fprintf('Using sample image: %s\n\n', imagePath);
    end
    
    %% Load and process the main sample image
    img = imread(imagePath);
    processedImg = preprocessImage(img);
    
    %% ═══════════════════════════════════════════════════════════════════
    %% FIGURE 1: TABLE 1 - Segmentation Process Steps
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('Generating Figure 1: Segmentation Process Steps...\n');
    
    fig1 = figure('Name', 'Table 1: Segmentation Steps', ...
                  'Position', [50, 50, 1400, 400], 'Color', 'white');
    
    % Step 1: Original
    subplot(1, 7, 1);
    imshow(processedImg);
    title('(a) Original', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Step 2: HSV Color Detection
    hsvMask = hsvThreshold(processedImg);
    subplot(1, 7, 2);
    imshow(hsvMask);
    title('(b) HSV Threshold', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Step 3: Morphological Opening
    se = strel('disk', 5);
    openedMask = imopen(hsvMask, se);
    subplot(1, 7, 3);
    imshow(openedMask);
    title('(c) Opening', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Step 4: Morphological Closing
    seClose = strel('disk', 10);
    closedMask = imclose(openedMask, seClose);
    subplot(1, 7, 4);
    imshow(closedMask);
    title('(d) Closing', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Step 5: Fill Holes
    filledMask = imfill(closedMask, 'holes');
    subplot(1, 7, 5);
    imshow(filledMask);
    title('(e) Filled Holes', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Step 6: Remove Small Regions
    finalMask = bwareaopen(filledMask, 500);
    subplot(1, 7, 6);
    imshow(finalMask);
    title('(f) Clean Mask', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Step 7: Segmented Result
    segmentedImg = processedImg;
    for c = 1:3
        channel = segmentedImg(:,:,c);
        channel(~finalMask) = 0;
        segmentedImg(:,:,c) = channel;
    end
    subplot(1, 7, 7);
    imshow(segmentedImg);
    title('(g) Segmented', 'FontSize', 10, 'FontWeight', 'bold');
    
    sgtitle('Table 1: Image Segmentation Process', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig1, fullfile(outputDir, 'Table1_Segmentation_Steps.png'));
    fprintf('  Saved: Table1_Segmentation_Steps.png\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% FIGURE 2: TABLE 2 - Texture Feature Extraction
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('Generating Figure 2: Texture Feature Extraction...\n');
    
    fig2 = figure('Name', 'Table 2: Texture Features', ...
                  'Position', [50, 50, 1200, 300], 'Color', 'white');
    
    grayImg = rgb2gray(processedImg);
    
    % ROI extraction
    stats = regionprops(finalMask, 'BoundingBox', 'Area');
    if ~isempty(stats)
        [~, idx] = max([stats.Area]);
        bbox = stats(idx).BoundingBox;
        roiImg = imcrop(grayImg, bbox);
    else
        roiImg = grayImg;
    end
    
    % Calculate texture metrics
    foodPixels = double(grayImg(finalMask)) / 255;
    statMean = mean(foodPixels);
    statStd = std(foodPixels);
    statSmoothness = 1 - (1 / (1 + statStd^2));
    
    subplot(1, 5, 1);
    imshow(processedImg);
    title('(a) Original', 'FontSize', 10, 'FontWeight', 'bold');
    
    subplot(1, 5, 2);
    imshow(finalMask);
    title('(b) Binarization', 'FontSize', 10, 'FontWeight', 'bold');
    
    subplot(1, 5, 3);
    imshow(segmentedImg);
    title('(c) Segmented', 'FontSize', 10, 'FontWeight', 'bold');
    
    subplot(1, 5, 4);
    imshow(grayImg);
    title('(d) Grayscale', 'FontSize', 10, 'FontWeight', 'bold');
    
    subplot(1, 5, 5);
    imshow(roiImg);
    title(sprintf('(e) ROI\nMean: %.4f\nStd: %.4f\nSmoothness: %.4f', ...
          statMean, statStd, statSmoothness), 'FontSize', 9, 'FontWeight', 'bold');
    
    sgtitle('Table 2: Texture Feature Extraction Results', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig2, fullfile(outputDir, 'Table2_Texture_Features.png'));
    fprintf('  Saved: Table2_Texture_Features.png\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% FIGURE 3: Color Histogram Analysis
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('Generating Figure 3: Color Histogram Analysis...\n');
    
    fig3 = figure('Name', 'Color Histogram', ...
                  'Position', [50, 50, 1000, 600], 'Color', 'white');
    
    % RGB Histograms
    subplot(2, 3, 1);
    imhist(processedImg(:,:,1));
    title('Red Channel', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('Intensity'); ylabel('Frequency');
    
    subplot(2, 3, 2);
    imhist(processedImg(:,:,2));
    title('Green Channel', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('Intensity'); ylabel('Frequency');
    
    subplot(2, 3, 3);
    imhist(processedImg(:,:,3));
    title('Blue Channel', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('Intensity'); ylabel('Frequency');
    
    % HSV Histograms
    hsvImg = rgb2hsv(im2double(processedImg));
    
    subplot(2, 3, 4);
    histogram(hsvImg(:,:,1), 16, 'FaceColor', [0.8, 0.2, 0.2]);
    title('Hue', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('Value'); ylabel('Frequency');
    
    subplot(2, 3, 5);
    histogram(hsvImg(:,:,2), 16, 'FaceColor', [0.2, 0.8, 0.2]);
    title('Saturation', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('Value'); ylabel('Frequency');
    
    subplot(2, 3, 6);
    histogram(hsvImg(:,:,3), 16, 'FaceColor', [0.2, 0.2, 0.8]);
    title('Value', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('Value'); ylabel('Frequency');
    
    sgtitle('Color Feature Analysis (RGB & HSV Histograms)', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig3, fullfile(outputDir, 'Figure3_Color_Histograms.png'));
    fprintf('  Saved: Figure3_Color_Histograms.png\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% FIGURE 4: GLCM Texture Visualization
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('Generating Figure 4: GLCM Texture Features...\n');
    
    fig4 = figure('Name', 'GLCM Texture', ...
                  'Position', [50, 50, 1000, 400], 'Color', 'white');
    
    % Compute GLCM
    grayImg8 = im2uint8(grayImg);
    offsets = [0 1; -1 1; -1 0; -1 -1];
    glcms = graycomatrix(grayImg8, 'Offset', offsets, 'NumLevels', 32);
    stats = graycoprops(glcms, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    
    % Bar chart of GLCM properties
    orientations = {'0°', '45°', '90°', '135°'};
    
    subplot(1, 4, 1);
    bar(stats.Contrast, 'FaceColor', [0.2, 0.6, 0.8]);
    set(gca, 'XTickLabel', orientations);
    title('Contrast', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Value');
    
    subplot(1, 4, 2);
    bar(stats.Correlation, 'FaceColor', [0.8, 0.4, 0.2]);
    set(gca, 'XTickLabel', orientations);
    title('Correlation', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Value');
    
    subplot(1, 4, 3);
    bar(stats.Energy, 'FaceColor', [0.4, 0.8, 0.4]);
    set(gca, 'XTickLabel', orientations);
    title('Energy', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Value');
    
    subplot(1, 4, 4);
    bar(stats.Homogeneity, 'FaceColor', [0.8, 0.2, 0.6]);
    set(gca, 'XTickLabel', orientations);
    title('Homogeneity', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Value');
    
    sgtitle('GLCM Texture Features at 4 Orientations', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig4, fullfile(outputDir, 'Figure4_GLCM_Features.png'));
    fprintf('  Saved: Figure4_GLCM_Features.png\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% FIGURE 5: Complete Feature Vector Visualization
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('Generating Figure 5: Feature Vector Visualization...\n');
    
    [features, featureNames] = extractFeatures(im2double(processedImg));
    
    fig5 = figure('Name', 'Feature Vector', ...
                  'Position', [50, 50, 1200, 400], 'Color', 'white');
    
    % Color code by feature type
    colors = zeros(127, 3);
    colors(1:48, :) = repmat([0.8, 0.2, 0.2], 48, 1);     % RGB hist (red)
    colors(49:96, :) = repmat([0.2, 0.8, 0.2], 48, 1);    % HSV hist (green)
    colors(97:108, :) = repmat([0.2, 0.2, 0.8], 12, 1);   % Statistics (blue)
    colors(109:124, :) = repmat([0.8, 0.6, 0.2], 16, 1);  % GLCM (orange)
    colors(125:127, :) = repmat([0.6, 0.2, 0.8], 3, 1);   % Mean/Std/Smooth (purple)
    
    bar(features, 'FaceColor', 'flat', 'CData', colors);
    xlabel('Feature Index', 'FontSize', 12);
    ylabel('Feature Value', 'FontSize', 12);
    title('127-Dimensional Feature Vector', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Add legend
    hold on;
    h1 = bar(nan, 'FaceColor', [0.8, 0.2, 0.2]);
    h2 = bar(nan, 'FaceColor', [0.2, 0.8, 0.2]);
    h3 = bar(nan, 'FaceColor', [0.2, 0.2, 0.8]);
    h4 = bar(nan, 'FaceColor', [0.8, 0.6, 0.2]);
    h5 = bar(nan, 'FaceColor', [0.6, 0.2, 0.8]);
    legend([h1, h2, h3, h4, h5], {'RGB Histogram (48)', 'HSV Histogram (48)', ...
            'Statistics (12)', 'GLCM (16)', 'Mean/Std/Smoothness (3)'}, ...
            'Location', 'northeast');
    hold off;
    
    saveas(fig5, fullfile(outputDir, 'Figure5_Feature_Vector.png'));
    fprintf('  Saved: Figure5_Feature_Vector.png\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% FIGURE 6: K-means Ingredient Segmentation
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('Generating Figure 6: K-means Ingredient Segmentation...\n');
    
    fig6 = figure('Name', 'K-means Segmentation', ...
                  'Position', [50, 50, 1000, 300], 'Color', 'white');
    
    labeledRegions = kmeansSegment(processedImg, finalMask, 5);
    
    subplot(1, 3, 1);
    imshow(processedImg);
    title('Original Image', 'FontSize', 11, 'FontWeight', 'bold');
    
    subplot(1, 3, 2);
    imshow(label2rgb(labeledRegions, 'jet', 'k', 'shuffle'));
    title('K-means Clusters (k=5)', 'FontSize', 11, 'FontWeight', 'bold');
    
    subplot(1, 3, 3);
    % Overlay
    overlay = imoverlay(processedImg, bwperim(finalMask), [0, 1, 0]);
    imshow(overlay);
    title('Food Region Overlay', 'FontSize', 11, 'FontWeight', 'bold');
    
    sgtitle('K-means Ingredient Segmentation', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig6, fullfile(outputDir, 'Figure6_Kmeans_Segmentation.png'));
    fprintf('  Saved: Figure6_Kmeans_Segmentation.png\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% FIGURE 7: Preprocessing Steps
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('Generating Figure 7: Preprocessing Pipeline...\n');
    
    fig7 = figure('Name', 'Preprocessing', ...
                  'Position', [50, 50, 1200, 300], 'Color', 'white');
    
    % Original
    subplot(1, 5, 1);
    imshow(img);
    title(sprintf('(a) Original\n%dx%d', size(img,1), size(img,2)), ...
          'FontSize', 10, 'FontWeight', 'bold');
    
    % Resized
    resizedImg = imresize(img, [512, 512]);
    subplot(1, 5, 2);
    imshow(resizedImg);
    title('(b) Resized 512×512', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Histogram Stretch
    stretchedImg = histogramStretch(im2double(resizedImg));
    subplot(1, 5, 3);
    imshow(stretchedImg);
    title('(c) Histogram Stretch', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Noise Filter
    filteredImg = noiseFilter(stretchedImg, 'median', 3);
    subplot(1, 5, 4);
    imshow(filteredImg);
    title('(d) Median Filter', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Final
    subplot(1, 5, 5);
    imshow(processedImg);
    title('(e) Final Output', 'FontSize', 10, 'FontWeight', 'bold');
    
    sgtitle('Image Preprocessing Pipeline', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig7, fullfile(outputDir, 'Figure7_Preprocessing.png'));
    fprintf('  Saved: Figure7_Preprocessing.png\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% FIGURE 8: Calorie Database Table
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('Generating Figure 8: Calorie Database Table...\n');
    
    fig8 = figure('Name', 'Calorie Database', ...
                  'Position', [50, 50, 800, 400], 'Color', 'white');
    
    db = foodDatabase();
    foodNames = {'Nasi Lemak', 'Roti Canai', 'Satay', 'Laksa', ...
                 'Popiah', 'Kaya Toast', 'Mixed Rice'};
    fieldNames = {'nasi_lemak', 'roti_canai', 'satay', 'laksa', ...
                  'popiah', 'kaya_toast', 'mixed_rice'};
    
    data = cell(7, 4);
    for i = 1:7
        data{i, 1} = db.(fieldNames{i}).baseCalories;
        data{i, 2} = db.(fieldNames{i}).protein;
        data{i, 3} = db.(fieldNames{i}).carbs;
        data{i, 4} = db.(fieldNames{i}).fat;
    end
    
    uitable(fig8, 'Data', data, ...
            'ColumnName', {'Calories (kcal)', 'Protein (g)', 'Carbs (g)', 'Fat (g)'}, ...
            'RowName', foodNames, ...
            'Position', [50, 50, 700, 300], ...
            'FontSize', 11);
    
    annotation('textbox', [0.1, 0.85, 0.8, 0.1], ...
               'String', 'Malaysian Food Calorie Database (MyFCD)', ...
               'FontSize', 14, 'FontWeight', 'bold', ...
               'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    
    saveas(fig8, fullfile(outputDir, 'Figure8_Calorie_Database.png'));
    fprintf('  Saved: Figure8_Calorie_Database.png\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% Summary
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║                    GENERATION COMPLETE                        ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  8 Figures generated and saved to:                           ║\n');
    fprintf('║  %s\n', outputDir);
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  FILES CREATED:                                               ║\n');
    fprintf('║  1. Table1_Segmentation_Steps.png                            ║\n');
    fprintf('║  2. Table2_Texture_Features.png                              ║\n');
    fprintf('║  3. Figure3_Color_Histograms.png                             ║\n');
    fprintf('║  4. Figure4_GLCM_Features.png                                ║\n');
    fprintf('║  5. Figure5_Feature_Vector.png                               ║\n');
    fprintf('║  6. Figure6_Kmeans_Segmentation.png                          ║\n');
    fprintf('║  7. Figure7_Preprocessing.png                                ║\n');
    fprintf('║  8. Figure8_Calorie_Database.png                             ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
    
    fprintf('Texture Stats for Report Table 2:\n');
    fprintf('  Mean:       %.4f\n', statMean);
    fprintf('  Std Dev:    %.4f\n', statStd);
    fprintf('  Smoothness: %.4f\n\n', statSmoothness);
    
    close all;
end
