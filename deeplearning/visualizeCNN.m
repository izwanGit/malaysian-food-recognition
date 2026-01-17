%% VISUALIZE CNN - Generate CNN Training Visualizations for Report
% Creates figures showing CNN architecture, training curves, and comparisons
%
% Usage:
%   visualizeCNN()

function visualizeCNN()
    %% Setup
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    outputDir = fullfile(projectRoot, 'report_figures');
    
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║        GENERATING CNN VISUALIZATION FIGURES                  ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
    
    figNum = 30;  % Continue from previous figures
    
    %% Figure: CNN vs SVM Comparison Architecture
    f = figure('Position', [100,100,1200,500], 'Color', 'w');
    
    % SVM Side
    subplot(1,2,1);
    axis off;
    title('Traditional Machine Learning (SVM)', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Draw SVM flow
    rectangle('Position', [0.1, 0.7, 0.2, 0.15], 'FaceColor', [0.9, 0.95, 1], 'EdgeColor', 'k', 'LineWidth', 2);
    text(0.2, 0.775, 'Image', 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    
    annotation('arrow', [0.17, 0.17], [0.7, 0.6]);
    
    rectangle('Position', [0.05, 0.45, 0.3, 0.15], 'FaceColor', [0.8, 0.9, 1], 'EdgeColor', 'k', 'LineWidth', 2);
    text(0.2, 0.525, {'Preprocessing', '+ Feature Extraction', '(127 features)'}, 'HorizontalAlignment', 'center', 'FontSize', 9);
    
    annotation('arrow', [0.17, 0.17], [0.45, 0.35]);
    
    rectangle('Position', [0.1, 0.2, 0.2, 0.15], 'FaceColor', [0.7, 0.85, 1], 'EdgeColor', 'k', 'LineWidth', 2);
    text(0.2, 0.275, {'SVM', 'Classifier'}, 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    
    annotation('arrow', [0.17, 0.17], [0.2, 0.1]);
    
    rectangle('Position', [0.1, 0, 0.2, 0.1], 'FaceColor', [0.5, 0.8, 0.5], 'EdgeColor', 'k', 'LineWidth', 2);
    text(0.2, 0.05, 'Prediction', 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'w');
    
    xlim([0, 0.4]);
    ylim([0, 1]);
    
    % CNN Side
    subplot(1,2,2);
    axis off;
    title('Deep Learning (CNN - ResNet18)', 'FontSize', 14, 'FontWeight', 'bold');
    
    rectangle('Position', [0.1, 0.7, 0.2, 0.15], 'FaceColor', [1, 0.95, 0.9], 'EdgeColor', 'k', 'LineWidth', 2);
    text(0.2, 0.775, 'Image', 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    
    annotation('arrow', [0.67, 0.67], [0.7, 0.6]);
    
    rectangle('Position', [0.05, 0.3, 0.3, 0.3], 'FaceColor', [1, 0.85, 0.7], 'EdgeColor', [0.8, 0.4, 0.2], 'LineWidth', 2);
    text(0.2, 0.52, {'ResNet18', '(18 layers)', '', 'Conv → BatchNorm → ReLU', '↓', 'Global Avg Pool', '↓', 'FC Layer (7 classes)'}, ...
         'HorizontalAlignment', 'center', 'FontSize', 8);
    
    annotation('arrow', [0.67, 0.67], [0.3, 0.2]);
    
    rectangle('Position', [0.1, 0.05, 0.2, 0.15], 'FaceColor', [0.5, 0.8, 0.5], 'EdgeColor', 'k', 'LineWidth', 2);
    text(0.2, 0.125, 'Prediction', 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Color', 'w');
    
    xlim([0, 0.4]);
    ylim([0, 1]);
    
    sgtitle('Machine Learning vs Deep Learning Architecture', 'FontSize', 16, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_SVM_vs_CNN_Architecture.png', figNum))); figNum = figNum + 1;
    close(f);
    fprintf('Saved: Fig30_SVM_vs_CNN_Architecture.png\n');
    
    %% Figure: CNN Layer Breakdown
    f = figure('Position', [100,100,1000,400], 'Color', 'w');
    
    layers = {'Input', 'Conv1', 'Block1', 'Block2', 'Block3', 'Block4', 'AvgPool', 'FC', 'Softmax'};
    sizes = [224, 112, 56, 28, 14, 7, 1, 1, 1];
    channels = [3, 64, 64, 128, 256, 512, 512, 7, 7];
    
    colors = [
        0.9, 0.95, 1.0;   % Input
        0.7, 0.85, 1.0;   % Conv
        0.5, 0.75, 0.95;  % Block1
        0.4, 0.65, 0.9;   % Block2
        0.3, 0.55, 0.85;  % Block3
        0.2, 0.45, 0.8;   % Block4
        0.15, 0.35, 0.7;  % AvgPool
        0.8, 0.6, 0.3;    % FC
        0.5, 0.8, 0.5;    % Softmax
    ];
    
    for i = 1:length(layers)
        barWidth = 0.8;
        x = i;
        h = sizes(i) / 224;  % Normalize height
        
        rectangle('Position', [x-barWidth/2, 0, barWidth, h], ...
                  'FaceColor', colors(i,:), 'EdgeColor', 'k', 'LineWidth', 1.5);
        
        text(x, h + 0.05, sprintf('%s\n%dx%d\n%dch', layers{i}, sizes(i), sizes(i), channels(i)), ...
             'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
    end
    
    xlim([0, 10]);
    ylim([0, 1.3]);
    xlabel('Layer', 'FontSize', 12);
    ylabel('Spatial Size (normalized)', 'FontSize', 12);
    title('ResNet18 Architecture Layer Breakdown', 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'XTick', 1:9, 'XTickLabel', layers);
    
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_ResNet18_Layers.png', figNum))); figNum = figNum + 1;
    close(f);
    fprintf('Saved: Fig31_ResNet18_Layers.png\n');
    
    %% Figure: Transfer Learning Concept
    f = figure('Position', [100,100,900,400], 'Color', 'w');
    
    subplot(1,2,1);
    axis off;
    title('Original ResNet18', 'FontSize', 12, 'FontWeight', 'bold');
    
    ypos = [0.9, 0.7, 0.5, 0.3, 0.1];
    labels = {'ImageNet Features', 'Conv Layers (frozen)', 'Global Avg Pool', 'FC: 1000 classes', 'ImageNet Classes'};
    colors = {[0.7, 0.85, 1], [0.6, 0.8, 0.95], [0.5, 0.75, 0.9], [0.9, 0.7, 0.7], [0.8, 0.6, 0.6]};
    
    for i = 1:5
        rectangle('Position', [0.1, ypos(i)-0.08, 0.8, 0.14], 'FaceColor', colors{i}, 'EdgeColor', 'k');
        text(0.5, ypos(i), labels{i}, 'HorizontalAlignment', 'center', 'FontSize', 10);
    end
    xlim([0, 1]);
    ylim([0, 1]);
    
    subplot(1,2,2);
    axis off;
    title('Transfer Learning (Ours)', 'FontSize', 12, 'FontWeight', 'bold');
    
    labels2 = {'Food Features', 'Conv Layers (fine-tuned)', 'Global Avg Pool', 'FC: 7 classes', 'Food Classes'};
    colors2 = {[0.7, 1, 0.7], [0.6, 0.95, 0.6], [0.5, 0.9, 0.5], [1, 0.9, 0.5], [0.9, 0.8, 0.4]};
    
    for i = 1:5
        rectangle('Position', [0.1, ypos(i)-0.08, 0.8, 0.14], 'FaceColor', colors2{i}, 'EdgeColor', 'k');
        text(0.5, ypos(i), labels2{i}, 'HorizontalAlignment', 'center', 'FontSize', 10);
    end
    xlim([0, 1]);
    ylim([0, 1]);
    
    sgtitle('Transfer Learning: Adapting Pretrained Network', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_TransferLearning.png', figNum))); figNum = figNum + 1;
    close(f);
    fprintf('Saved: Fig32_TransferLearning.png\n');
    
    %% Figure: Data Augmentation Examples
    f = figure('Position', [100,100,1000,300], 'Color', 'w');
    
    % Create sample transformations on a simple image
    testImg = rand(100, 100, 3);
    
    subplot(1,5,1);
    imshow(testImg);
    title('Original', 'FontSize', 10);
    
    subplot(1,5,2);
    imshow(imrotate(testImg, 15, 'crop'));
    title('Rotation', 'FontSize', 10);
    
    subplot(1,5,3);
    imshow(fliplr(testImg));
    title('Horizontal Flip', 'FontSize', 10);
    
    subplot(1,5,4);
    imshow(imresize(testImg, 1.1));
    title('Scale', 'FontSize', 10);
    
    subplot(1,5,5);
    imshow(testImg .* 1.2);
    title('Brightness', 'FontSize', 10);
    
    sgtitle('Data Augmentation Techniques', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_DataAugmentation.png', figNum))); figNum = figNum + 1;
    close(f);
    fprintf('Saved: Fig33_DataAugmentation.png\n');
    
    %% Summary
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║               CNN FIGURES GENERATED                          ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Fig30: SVM vs CNN Architecture                              ║\n');
    fprintf('║  Fig31: ResNet18 Layer Breakdown                             ║\n');
    fprintf('║  Fig32: Transfer Learning Concept                            ║\n');
    fprintf('║  Fig33: Data Augmentation                                    ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
end
