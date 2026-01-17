%% GENERATE EXTENDED REPORT FIGURES - Complete Figure Suite for A++ Report
% Generates 20+ individual figures for maximum report flexibility
%
% Usage:
%   generateExtendedFigures()           % Uses sample images
%   generateExtendedFigures(imagePath)  % Uses specific image

function generateExtendedFigures(imagePath)
    %% Setup
    projectRoot = fileparts(mfilename('fullpath'));
    addpath(genpath(projectRoot));
    
    outputDir = fullfile(projectRoot, 'report_figures');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║    GENERATING 20+ EXTENDED FIGURES FOR A++ REPORT            ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
    
    %% Find sample image
    if nargin < 1
        datasetPath = fullfile(projectRoot, 'dataset', 'train', 'nasi_lemak');
        images = dir(fullfile(datasetPath, '*.jpg'));
        if ~isempty(images)
            imagePath = fullfile(datasetPath, images(1).name);
        else
            error('No dataset found. Provide image path.');
        end
    end
    
    img = imread(imagePath);
    processedImg = preprocessImage(img);
    
    figNum = 1;
    
    %% ═══════════════════════════════════════════════════════════════════
    %% INDIVIDUAL PREPROCESSING FIGURES
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('--- Preprocessing Figures ---\n');
    
    % Fig 1: Original Image Only
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    imshow(img);
    title('Original Image', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_Original.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 2: Resized Image
    resized = imresize(img, [512, 512]);
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    imshow(resized);
    title('Resized (512×512)', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_Resized.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 3: Histogram Stretched
    stretched = histogramStretch(im2double(resized));
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    imshow(stretched);
    title('Histogram Stretched', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_HistogramStretch.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 4: Noise Filtered
    filtered = noiseFilter(stretched, 'median', 3);
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    imshow(filtered);
    title('Median Filtered', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_MedianFiltered.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 5: Final Preprocessed
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    imshow(processedImg);
    title('Preprocessed Result', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_Preprocessed.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 6: Before/After Comparison
    f = figure('Position', [100,100,800,400], 'Color', 'w');
    subplot(1,2,1); imshow(img); title('Before', 'FontSize', 12, 'FontWeight', 'bold');
    subplot(1,2,2); imshow(processedImg); title('After Preprocessing', 'FontSize', 12, 'FontWeight', 'bold');
    sgtitle('Preprocessing Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_PreprocessingComparison.png', figNum))); figNum = figNum + 1;
    close(f);
    
    fprintf('  Saved 6 preprocessing figures\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% INDIVIDUAL SEGMENTATION FIGURES
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('--- Segmentation Figures ---\n');
    
    % Get masks
    hsvMask = hsvThreshold(processedImg);
    se = strel('disk', 5);
    openedMask = imopen(hsvMask, se);
    closedMask = imclose(openedMask, strel('disk', 10));
    filledMask = imfill(closedMask, 'holes');
    finalMask = bwareaopen(filledMask, 500);
    
    % Fig 7: HSV Mask
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    imshow(hsvMask);
    title('HSV Color Threshold', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_HSVMask.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 8: Morphological Opening
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    imshow(openedMask);
    title('Morphological Opening', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_MorphOpen.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 9: Morphological Closing
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    imshow(closedMask);
    title('Morphological Closing', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_MorphClose.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 10: Filled Holes
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    imshow(filledMask);
    title('Holes Filled', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_HolesFilled.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 11: Final Clean Mask
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    imshow(finalMask);
    title('Final Binary Mask', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_FinalMask.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 12: Segmented RGB
    segmentedImg = processedImg;
    for c = 1:3
        ch = segmentedImg(:,:,c);
        ch(~finalMask) = 0;
        segmentedImg(:,:,c) = ch;
    end
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    imshow(segmentedImg);
    title('Segmented Image', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_SegmentedRGB.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 13: Boundary Overlay
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    overlay = imoverlay(processedImg, bwperim(finalMask), [0, 1, 0]);
    imshow(overlay);
    title('Segmentation Overlay', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_BoundaryOverlay.png', figNum))); figNum = figNum + 1;
    close(f);
    
    fprintf('  Saved 7 segmentation figures\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% K-MEANS FIGURES
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('--- K-means Clustering Figures ---\n');
    
    % Fig 14: K-means k=3
    labels3 = kmeansSegment(processedImg, finalMask, 3);
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    imshow(label2rgb(labels3, 'jet', 'k', 'shuffle'));
    title('K-means (k=3)', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_Kmeans_k3.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 15: K-means k=5
    labels5 = kmeansSegment(processedImg, finalMask, 5);
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    imshow(label2rgb(labels5, 'jet', 'k', 'shuffle'));
    title('K-means (k=5)', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_Kmeans_k5.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 16: K-means Comparison
    f = figure('Position', [100,100,900,300], 'Color', 'w');
    subplot(1,3,1); imshow(processedImg); title('Original', 'FontSize', 11);
    subplot(1,3,2); imshow(label2rgb(labels3, 'jet', 'k')); title('K=3 Clusters', 'FontSize', 11);
    subplot(1,3,3); imshow(label2rgb(labels5, 'jet', 'k')); title('K=5 Clusters', 'FontSize', 11);
    sgtitle('K-means Clustering Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_KmeansComparison.png', figNum))); figNum = figNum + 1;
    close(f);
    
    fprintf('  Saved 3 k-means figures\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% COLOR ANALYSIS FIGURES
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('--- Color Analysis Figures ---\n');
    
    % Fig 17: RGB Channels
    f = figure('Position', [100,100,900,300], 'Color', 'w');
    subplot(1,3,1); imshow(processedImg(:,:,1)); title('Red Channel', 'FontSize', 11);
    subplot(1,3,2); imshow(processedImg(:,:,2)); title('Green Channel', 'FontSize', 11);
    subplot(1,3,3); imshow(processedImg(:,:,3)); title('Blue Channel', 'FontSize', 11);
    sgtitle('RGB Channel Separation', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_RGBChannels.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 18: HSV Channels
    hsvImg = rgb2hsv(im2double(processedImg));
    f = figure('Position', [100,100,900,300], 'Color', 'w');
    subplot(1,3,1); imshow(hsvImg(:,:,1)); title('Hue', 'FontSize', 11);
    subplot(1,3,2); imshow(hsvImg(:,:,2)); title('Saturation', 'FontSize', 11);
    subplot(1,3,3); imshow(hsvImg(:,:,3)); title('Value', 'FontSize', 11);
    sgtitle('HSV Channel Separation', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_HSVChannels.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 19: RGB Histograms
    f = figure('Position', [100,100,900,300], 'Color', 'w');
    subplot(1,3,1); imhist(processedImg(:,:,1)); title('Red Histogram', 'FontSize', 11);
    subplot(1,3,2); imhist(processedImg(:,:,2)); title('Green Histogram', 'FontSize', 11);
    subplot(1,3,3); imhist(processedImg(:,:,3)); title('Blue Histogram', 'FontSize', 11);
    sgtitle('RGB Histogram Analysis', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_RGBHistograms.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 20: HSV Histograms
    f = figure('Position', [100,100,900,300], 'Color', 'w');
    subplot(1,3,1); histogram(hsvImg(:,:,1), 16, 'FaceColor', [0.8, 0.2, 0.2]); title('Hue', 'FontSize', 11);
    subplot(1,3,2); histogram(hsvImg(:,:,2), 16, 'FaceColor', [0.2, 0.8, 0.2]); title('Saturation', 'FontSize', 11);
    subplot(1,3,3); histogram(hsvImg(:,:,3), 16, 'FaceColor', [0.2, 0.2, 0.8]); title('Value', 'FontSize', 11);
    sgtitle('HSV Histogram Analysis', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_HSVHistograms.png', figNum))); figNum = figNum + 1;
    close(f);
    
    fprintf('  Saved 4 color analysis figures\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% TEXTURE ANALYSIS FIGURES
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('--- Texture Analysis Figures ---\n');
    
    grayImg = rgb2gray(processedImg);
    
    % Fig 21: Grayscale Image
    f = figure('Position', [100,100,400,400], 'Color', 'w');
    imshow(grayImg);
    title('Grayscale Image', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_Grayscale.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 22: GLCM Visualization
    grayImg8 = im2uint8(grayImg);
    offsets = [0 1; -1 1; -1 0; -1 -1];
    glcms = graycomatrix(grayImg8, 'Offset', offsets, 'NumLevels', 32);
    stats = graycoprops(glcms, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    
    f = figure('Position', [100,100,1000,250], 'Color', 'w');
    orientations = categorical({'0°', '45°', '90°', '135°'});
    
    subplot(1,4,1); bar(orientations, stats.Contrast, 'FaceColor', [0.2, 0.6, 0.8]);
    title('Contrast', 'FontSize', 11); ylabel('Value');
    
    subplot(1,4,2); bar(orientations, stats.Correlation, 'FaceColor', [0.8, 0.4, 0.2]);
    title('Correlation', 'FontSize', 11); ylabel('Value');
    
    subplot(1,4,3); bar(orientations, stats.Energy, 'FaceColor', [0.4, 0.8, 0.4]);
    title('Energy', 'FontSize', 11); ylabel('Value');
    
    subplot(1,4,4); bar(orientations, stats.Homogeneity, 'FaceColor', [0.8, 0.2, 0.6]);
    title('Homogeneity', 'FontSize', 11); ylabel('Value');
    
    sgtitle('GLCM Properties at 4 Orientations', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_GLCMProperties.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 23: GLCM Matrix Visualization
    f = figure('Position', [100,100,1000,250], 'Color', 'w');
    for i = 1:4
        subplot(1,4,i);
        imagesc(glcms(:,:,i));
        colormap(hot);
        title(sprintf('%d°', (i-1)*45), 'FontSize', 11);
        axis square;
    end
    sgtitle('GLCM Matrices at Different Orientations', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_GLCMMatrices.png', figNum))); figNum = figNum + 1;
    close(f);
    
    fprintf('  Saved 3 texture analysis figures\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% FEATURE VECTOR FIGURES
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('--- Feature Vector Figures ---\n');
    
    [features, ~] = extractFeatures(im2double(processedImg));
    
    % Fig 24: Full Feature Vector
    f = figure('Position', [100,100,1200,400], 'Color', 'w');
    bar(features, 'FaceColor', [0.3, 0.6, 0.8]);
    xlabel('Feature Index'); ylabel('Value');
    title('127-Dimensional Feature Vector', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_FeatureVector.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 25: Feature Breakdown (Pie Chart)
    f = figure('Position', [100,100,500,400], 'Color', 'w');
    pie([108, 16, 3], {'Color Features (108)', 'GLCM Features (16)', 'Statistical (3)'});
    title('Feature Composition (127 Total)', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_FeatureComposition.png', figNum))); figNum = figNum + 1;
    close(f);
    
    fprintf('  Saved 2 feature vector figures\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% PORTION ESTIMATION FIGURES
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('--- Portion Estimation Figures ---\n');
    
    [portionRatio, portionLabel, areaPixels] = estimatePortion(finalMask, 'nasi_lemak');
    
    % Fig 26: Portion Visualization
    f = figure('Position', [100,100,800,400], 'Color', 'w');
    subplot(1,2,1);
    imshow(finalMask);
    title(sprintf('Food Area: %d pixels', areaPixels), 'FontSize', 12, 'FontWeight', 'bold');
    
    subplot(1,2,2);
    bar(portionRatio, 'FaceColor', [0.2, 0.8, 0.4]);
    hold on;
    yline(1, 'r--', 'LineWidth', 2);
    hold off;
    ylabel('Portion Ratio');
    title(sprintf('Portion: %s (%.2fx)', portionLabel, portionRatio), 'FontSize', 12, 'FontWeight', 'bold');
    ylim([0, 2]);
    
    sgtitle('Portion Size Estimation', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_PortionEstimation.png', figNum))); figNum = figNum + 1;
    close(f);
    
    fprintf('  Saved 1 portion estimation figure\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% CALORIE FIGURES
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('--- Calorie & Nutrition Figures ---\n');
    
    db = foodDatabase();
    
    % Fig 27: Calorie Comparison Bar Chart
    foods = {'nasi_lemak', 'roti_canai', 'satay', 'laksa', 'popiah', 'kaya_toast', 'mixed_rice'};
    names = {'Nasi Lemak', 'Roti Canai', 'Satay', 'Laksa', 'Popiah', 'Kaya Toast', 'Mixed Rice'};
    calories = zeros(7,1);
    for i = 1:7
        calories(i) = db.(foods{i}).baseCalories;
    end
    
    f = figure('Position', [100,100,800,400], 'Color', 'w');
    barh(categorical(names), calories, 'FaceColor', [0.9, 0.4, 0.2]);
    xlabel('Calories (kcal)'); ylabel('Food');
    title('Calorie Comparison of Malaysian Hawker Foods', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_CalorieComparison.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 28: Macronutrient Comparison
    protein = zeros(7,1); carbs = zeros(7,1); fat = zeros(7,1);
    for i = 1:7
        protein(i) = db.(foods{i}).protein;
        carbs(i) = db.(foods{i}).carbs;
        fat(i) = db.(foods{i}).fat;
    end
    
    f = figure('Position', [100,100,900,400], 'Color', 'w');
    data = [protein, carbs, fat];
    bar(categorical(names), data);
    ylabel('Grams (g)');
    legend({'Protein', 'Carbohydrates', 'Fat'}, 'Location', 'northwest');
    title('Macronutrient Comparison', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_MacronutrientComparison.png', figNum))); figNum = figNum + 1;
    close(f);
    
    % Fig 29: Single Food Nutrition Pie
    f = figure('Position', [100,100,500,400], 'Color', 'w');
    pie([db.nasi_lemak.protein, db.nasi_lemak.carbs, db.nasi_lemak.fat], ...
        {sprintf('Protein\n%.0fg', db.nasi_lemak.protein), ...
         sprintf('Carbs\n%.0fg', db.nasi_lemak.carbs), ...
         sprintf('Fat\n%.0fg', db.nasi_lemak.fat)});
    title('Nasi Lemak Macronutrient Breakdown', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_NasiLemakPie.png', figNum))); figNum = figNum + 1;
    close(f);
    
    fprintf('  Saved 3 calorie/nutrition figures\n');
    
    %% ═══════════════════════════════════════════════════════════════════
    %% SUMMARY
    %% ═══════════════════════════════════════════════════════════════════
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║                    GENERATION COMPLETE                        ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Total Figures Generated: %d                                  ║\n', figNum-1);
    fprintf('║  Output Directory: report_figures/                           ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
    
    fprintf('Categories:\n');
    fprintf('  - Preprocessing: 6 figures\n');
    fprintf('  - Segmentation: 7 figures\n');
    fprintf('  - K-means: 3 figures\n');
    fprintf('  - Color Analysis: 4 figures\n');
    fprintf('  - Texture Analysis: 3 figures\n');
    fprintf('  - Feature Vector: 2 figures\n');
    fprintf('  - Portion Estimation: 1 figure\n');
    fprintf('  - Calorie/Nutrition: 3 figures\n');
    fprintf('  TOTAL: 29 figures\n\n');
end
