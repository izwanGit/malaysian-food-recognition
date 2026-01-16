%% GENERATE MULTI-CLASS REPORT - Generate Results for All 7 Food Classes
% Creates the exact table format required by the rubric for all food classes
%
% This script generates:
% - Table 1 format for ALL 7 food classes (segmentation steps)
% - Table 2 format for ALL 7 food classes (texture features)
% - Combined comparison figures
%
% Usage:
%   generateMultiClassReport()

function generateMultiClassReport()
    %% Setup
    projectRoot = fileparts(mfilename('fullpath'));
    addpath(genpath(projectRoot));
    
    outputDir = fullfile(projectRoot, 'report_figures');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    datasetPath = fullfile(projectRoot, 'dataset', 'train');
    foodClasses = {'nasi_lemak', 'roti_canai', 'satay', 'laksa', ...
                   'popiah', 'kaya_toast', 'mixed_rice'};
    displayNames = {'Nasi Lemak', 'Roti Canai', 'Satay', 'Laksa', ...
                    'Popiah', 'Kaya Toast', 'Mixed Rice'};
    
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║      GENERATING MULTI-CLASS REPORT TABLES & FIGURES          ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
    
    %% Collect sample images
    sampleImages = cell(7, 1);
    for i = 1:7
        classPath = fullfile(datasetPath, foodClasses{i});
        images = dir(fullfile(classPath, '*.jpg'));
        if ~isempty(images)
            sampleImages{i} = fullfile(classPath, images(1).name);
            fprintf('Found: %s\n', sampleImages{i});
        else
            fprintf('Missing: %s (class folder not found)\n', foodClasses{i});
        end
    end
    
    %% ═══════════════════════════════════════════════════════════════════
    %% RUBRIC TABLE 1: Segmentation Results for All Classes
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('\n--- Generating Rubric Table 1: Segmentation Results ---\n');
    
    fig1 = figure('Name', 'Table 1: Multi-Class Segmentation', ...
                  'Position', [0, 0, 1800, 1200], 'Color', 'white');
    
    % 7 rows (classes) x 7 columns (steps)
    for row = 1:7
        if isempty(sampleImages{row})
            continue;
        end
        
        img = imread(sampleImages{row});
        processedImg = preprocessImage(img);
        
        % Column 1: Original
        subplot(7, 7, (row-1)*7 + 1);
        imshow(processedImg);
        if row == 1
            title('Original', 'FontSize', 9);
        end
        if mod((row-1)*7 + 1, 7) == 1
            ylabel(displayNames{row}, 'FontSize', 10, 'FontWeight', 'bold');
        end
        
        % Column 2: HSV Threshold
        hsvMask = hsvThreshold(processedImg);
        subplot(7, 7, (row-1)*7 + 2);
        imshow(hsvMask);
        if row == 1, title('HSV Mask', 'FontSize', 9); end
        
        % Column 3: Opening
        se = strel('disk', 5);
        openedMask = imopen(hsvMask, se);
        subplot(7, 7, (row-1)*7 + 3);
        imshow(openedMask);
        if row == 1, title('Opening', 'FontSize', 9); end
        
        % Column 4: Closing
        closedMask = imclose(openedMask, strel('disk', 10));
        subplot(7, 7, (row-1)*7 + 4);
        imshow(closedMask);
        if row == 1, title('Closing', 'FontSize', 9); end
        
        % Column 5: Fill Holes
        filledMask = imfill(closedMask, 'holes');
        subplot(7, 7, (row-1)*7 + 5);
        imshow(filledMask);
        if row == 1, title('Filled', 'FontSize', 9); end
        
        % Column 6: Remove Small
        finalMask = bwareaopen(filledMask, 500);
        subplot(7, 7, (row-1)*7 + 6);
        imshow(finalMask);
        if row == 1, title('Cleaned', 'FontSize', 9); end
        
        % Column 7: Segmented
        segmentedImg = processedImg;
        for c = 1:3
            channel = segmentedImg(:,:,c);
            channel(~finalMask) = 0;
            segmentedImg(:,:,c) = channel;
        end
        subplot(7, 7, (row-1)*7 + 7);
        imshow(segmentedImg);
        if row == 1, title('Segmented', 'FontSize', 9); end
        
        fprintf('  Processed: %s\n', displayNames{row});
    end
    
    sgtitle('Table 1: Image Segmentation Results for All Food Classes', ...
            'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig1, fullfile(outputDir, 'RubricTable1_AllClasses_Segmentation.png'));
    fprintf('  Saved: RubricTable1_AllClasses_Segmentation.png\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% RUBRIC TABLE 2: Texture Features for All Classes
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('\n--- Generating Rubric Table 2: Texture Features ---\n');
    
    fig2 = figure('Name', 'Table 2: Multi-Class Texture', ...
                  'Position', [0, 0, 1600, 1200], 'Color', 'white');
    
    textureResults = zeros(7, 3);  % Mean, Std, Smoothness
    
    for row = 1:7
        if isempty(sampleImages{row})
            continue;
        end
        
        img = imread(sampleImages{row});
        processedImg = preprocessImage(img);
        
        % Get mask
        hsvMask = hsvThreshold(processedImg);
        cleanMask = morphologyClean(hsvMask);
        
        grayImg = rgb2gray(processedImg);
        
        % ROI
        stats = regionprops(cleanMask, 'BoundingBox', 'Area');
        if ~isempty(stats)
            [~, idx] = max([stats.Area]);
            bbox = stats(idx).BoundingBox;
            roiImg = imcrop(grayImg, bbox);
        else
            roiImg = grayImg;
        end
        
        % Texture stats
        foodPixels = double(grayImg(cleanMask)) / 255;
        if ~isempty(foodPixels)
            statMean = mean(foodPixels);
            statStd = std(foodPixels);
            statSmoothness = 1 - (1 / (1 + statStd^2));
        else
            statMean = 0; statStd = 0; statSmoothness = 0;
        end
        textureResults(row, :) = [statMean, statStd, statSmoothness];
        
        % Segmented image
        segmentedImg = processedImg;
        for c = 1:3
            channel = segmentedImg(:,:,c);
            channel(~cleanMask) = 0;
            segmentedImg(:,:,c) = channel;
        end
        
        % Column 1: Original
        subplot(7, 6, (row-1)*6 + 1);
        imshow(processedImg);
        if row == 1, title('Original', 'FontSize', 9); end
        ylabel(displayNames{row}, 'FontSize', 10, 'FontWeight', 'bold');
        
        % Column 2: Binarization
        subplot(7, 6, (row-1)*6 + 2);
        imshow(cleanMask);
        if row == 1, title('Binary', 'FontSize', 9); end
        
        % Column 3: Segmented
        subplot(7, 6, (row-1)*6 + 3);
        imshow(segmentedImg);
        if row == 1, title('Segmented', 'FontSize', 9); end
        
        % Column 4: Grayscale
        subplot(7, 6, (row-1)*6 + 4);
        imshow(grayImg);
        if row == 1, title('Grayscale', 'FontSize', 9); end
        
        % Column 5: ROI
        subplot(7, 6, (row-1)*6 + 5);
        imshow(roiImg);
        if row == 1, title('ROI', 'FontSize', 9); end
        
        % Column 6: Results text
        subplot(7, 6, (row-1)*6 + 6);
        axis off;
        text(0.1, 0.7, sprintf('Mean: %.4f', statMean), 'FontSize', 10);
        text(0.1, 0.5, sprintf('Std: %.4f', statStd), 'FontSize', 10);
        text(0.1, 0.3, sprintf('Smooth: %.4f', statSmoothness), 'FontSize', 10);
        if row == 1, title('Results', 'FontSize', 9); end
        
        fprintf('  %s: Mean=%.4f, Std=%.4f, Smooth=%.4f\n', ...
                displayNames{row}, statMean, statStd, statSmoothness);
    end
    
    sgtitle('Table 2: Texture Feature Extraction Results for All Food Classes', ...
            'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig2, fullfile(outputDir, 'RubricTable2_AllClasses_Texture.png'));
    fprintf('  Saved: RubricTable2_AllClasses_Texture.png\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% Summary Table (Text Format for Copy-Paste)
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
    fprintf('║           TABLE 2 VALUES - COPY TO REPORT                        ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
    fprintf('║  No. │ Food Class    │   Mean   │   Std    │ Smoothness          ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
    for i = 1:7
        fprintf('║  %d.  │ %-13s │ %.4f   │ %.4f   │ %.4f              ║\n', ...
                i, displayNames{i}, textureResults(i,1), textureResults(i,2), textureResults(i,3));
    end
    fprintf('╚══════════════════════════════════════════════════════════════════╝\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% Comparison Bar Chart
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('\n--- Generating Comparison Charts ---\n');
    
    fig3 = figure('Name', 'Texture Comparison', ...
                  'Position', [50, 50, 1000, 400], 'Color', 'white');
    
    subplot(1, 3, 1);
    bar(textureResults(:, 1), 'FaceColor', [0.2, 0.6, 0.8]);
    set(gca, 'XTickLabel', displayNames, 'XTickLabelRotation', 45);
    title('Mean Intensity', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Value');
    
    subplot(1, 3, 2);
    bar(textureResults(:, 2), 'FaceColor', [0.8, 0.4, 0.2]);
    set(gca, 'XTickLabel', displayNames, 'XTickLabelRotation', 45);
    title('Standard Deviation', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Value');
    
    subplot(1, 3, 3);
    bar(textureResults(:, 3), 'FaceColor', [0.4, 0.8, 0.4]);
    set(gca, 'XTickLabel', displayNames, 'XTickLabelRotation', 45);
    title('Smoothness', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Value');
    
    sgtitle('Texture Feature Comparison Across Food Classes', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig3, fullfile(outputDir, 'Figure_TextureComparison.png'));
    fprintf('  Saved: Figure_TextureComparison.png\n');
    
    %% Done
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║                    ALL FIGURES GENERATED                      ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
    
    close all;
end
