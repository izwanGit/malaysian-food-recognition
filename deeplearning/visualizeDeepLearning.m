%% VISUALIZE DEEP LEARNING - Complete DL Visualization Suite
% Generates all deep learning figures for the report
%
% Generates:
%   - ResNet18 vs DeepLabv3+ comparison
%   - Architecture diagrams
%   - Classical vs DL segmentation comparison
%
% Usage:
%   visualizeDeepLearning()

function visualizeDeepLearning()
    %% Setup
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    outputDir = fullfile(projectRoot, 'report_figures');
    
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║      GENERATING DEEP LEARNING FIGURES FOR REPORT             ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
    
    figNum = 30;
    
    %% Figure 30: ResNet18 Classification Architecture
    f = figure('Position', [100,100,1000,500], 'Color', 'w');
    
    % Draw ResNet18 blocks
    blockNames = {'Input', 'Conv1', 'Res Block 1', 'Res Block 2', 'Res Block 3', 'Res Block 4', 'Avg Pool', 'FC', 'Output'};
    blockColors = [
        0.9, 0.95, 1.0;
        0.7, 0.85, 1.0;
        0.5, 0.75, 0.95;
        0.4, 0.7, 0.9;
        0.3, 0.6, 0.85;
        0.2, 0.5, 0.8;
        0.4, 0.7, 0.5;
        0.9, 0.7, 0.4;
        0.5, 0.8, 0.5;
    ];
    
    for i = 1:length(blockNames)
        x = (i-1) * 0.11 + 0.05;
        rectangle('Position', [x, 0.3, 0.09, 0.4], 'FaceColor', blockColors(i,:), ...
                  'EdgeColor', 'k', 'LineWidth', 1.5, 'Curvature', 0.1);
        text(x + 0.045, 0.5, blockNames{i}, 'HorizontalAlignment', 'center', ...
             'FontSize', 9, 'FontWeight', 'bold', 'Rotation', 90);
        
        if i < length(blockNames)
            annotation('arrow', [x+0.09, x+0.11], [0.5, 0.5], 'Color', [0.3, 0.3, 0.3]);
        end
    end
    
    axis off;
    xlim([0, 1]);
    ylim([0, 1]);
    title('ResNet18 Architecture for Food Classification', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Add labels
    text(0.5, 0.15, 'Input: 224×224×3 → Output: 7 Food Classes', 'HorizontalAlignment', 'center', 'FontSize', 11);
    text(0.5, 0.85, 'Transfer Learning: Pretrained on ImageNet (1.2M images)', 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontStyle', 'italic');
    
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_ResNet18_Architecture.png', figNum))); figNum = figNum + 1;
    close(f);
    fprintf('Saved: Fig30_ResNet18_Architecture.png\n');
    
    %% Figure 31: DeepLabv3+ Segmentation Architecture
    f = figure('Position', [100,100,1200,500], 'Color', 'w');
    
    % Encoder
    subplot(1,2,1);
    axis off;
    title('Encoder (ResNet18 Backbone)', 'FontSize', 12, 'FontWeight', 'bold');
    
    encoderBlocks = {'Input\n512×512', 'Conv1\n256×256', 'Res1\n128×128', 'Res2\n64×64', 'Res3\n32×32', 'Res4\n16×16'};
    for i = 1:length(encoderBlocks)
        y = 1 - i * 0.15;
        rectangle('Position', [0.2, y, 0.6, 0.12], 'FaceColor', [0.6, 0.8, 1], 'EdgeColor', 'k');
        text(0.5, y + 0.06, encoderBlocks{i}, 'HorizontalAlignment', 'center', 'FontSize', 9);
    end
    xlim([0, 1]);
    ylim([0, 1]);
    
    % ASPP and Decoder
    subplot(1,2,2);
    axis off;
    title('ASPP + Decoder', 'FontSize', 12, 'FontWeight', 'bold');
    
    % ASPP module
    rectangle('Position', [0.1, 0.6, 0.8, 0.3], 'FaceColor', [1, 0.9, 0.7], 'EdgeColor', 'k', 'LineWidth', 2);
    text(0.5, 0.85, 'ASPP Module', 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
    text(0.5, 0.72, 'Atrous Conv (rate 6, 12, 18) + Global Pooling', 'HorizontalAlignment', 'center', 'FontSize', 9);
    
    % Decoder
    decoderBlocks = {'1×1 Conv', 'Upsample 4×', 'Concatenate', 'Upsample 4×', 'Output\n512×512'};
    for i = 1:length(decoderBlocks)
        y = 0.55 - i * 0.11;
        rectangle('Position', [0.2, y, 0.6, 0.09], 'FaceColor', [0.7, 1, 0.7], 'EdgeColor', 'k');
        text(0.5, y + 0.045, decoderBlocks{i}, 'HorizontalAlignment', 'center', 'FontSize', 9);
    end
    xlim([0, 1]);
    ylim([0, 1]);
    
    sgtitle('DeepLabv3+ Architecture for Food Segmentation', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_DeepLabv3_Architecture.png', figNum))); figNum = figNum + 1;
    close(f);
    fprintf('Saved: Fig31_DeepLabv3_Architecture.png\n');
    
    %% Figure 32: Classical vs Deep Learning Comparison
    f = figure('Position', [100,100,1200,400], 'Color', 'w');
    
    % Classical Pipeline
    subplot(1,2,1);
    axis off;
    title('Classical Image Processing', 'FontSize', 12, 'FontWeight', 'bold');
    
    classicalSteps = {'RGB Image', 'Preprocessing', 'HSV Threshold', 'Morphology', 'K-means', 'Final Mask'};
    for i = 1:length(classicalSteps)
        y = 1 - i * 0.15;
        rectangle('Position', [0.2, y, 0.6, 0.12], 'FaceColor', [0.8, 0.9, 1], 'EdgeColor', 'k');
        text(0.5, y + 0.06, classicalSteps{i}, 'HorizontalAlignment', 'center', 'FontSize', 10);
        if i < length(classicalSteps)
            annotation('arrow', [0.38, 0.38], [y-0.01, y-0.02]);
        end
    end
    xlim([0, 1]);
    ylim([0, 1]);
    
    % Deep Learning Pipeline
    subplot(1,2,2);
    axis off;
    title('Deep Learning (DeepLabv3+)', 'FontSize', 12, 'FontWeight', 'bold');
    
    dlSteps = {'RGB Image', 'Resize 512×512', 'DeepLabv3+', 'Softmax', 'Argmax', 'Segmentation Mask'};
    for i = 1:length(dlSteps)
        y = 1 - i * 0.15;
        rectangle('Position', [0.2, y, 0.6, 0.12], 'FaceColor', [1, 0.9, 0.8], 'EdgeColor', 'k');
        text(0.5, y + 0.06, dlSteps{i}, 'HorizontalAlignment', 'center', 'FontSize', 10);
    end
    xlim([0, 1]);
    ylim([0, 1]);
    
    sgtitle('Segmentation: Classical vs Deep Learning Approach', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_Classical_vs_DL.png', figNum))); figNum = figNum + 1;
    close(f);
    fprintf('Saved: Fig32_Classical_vs_DL.png\n');
    
    %% Figure 33: Complete System Architecture
    f = figure('Position', [100,100,1400,600], 'Color', 'w');
    axis off;
    
    % Title
    text(0.5, 0.95, 'Complete System Architecture: Classical + Deep Learning', ...
         'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Input
    rectangle('Position', [0.02, 0.4, 0.1, 0.2], 'FaceColor', [0.9, 0.95, 1], 'EdgeColor', 'k', 'LineWidth', 2);
    text(0.07, 0.5, {'Input', 'Image'}, 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Preprocessing
    rectangle('Position', [0.15, 0.4, 0.12, 0.2], 'FaceColor', [0.8, 0.9, 1], 'EdgeColor', 'k', 'LineWidth', 2);
    text(0.21, 0.5, {'Preprocessing', '(CLAHE +', 'Filtering)'}, 'HorizontalAlignment', 'center', 'FontSize', 9);
    
    % Branch arrow
    annotation('arrow', [0.27, 0.32], [0.5, 0.65]);
    annotation('arrow', [0.27, 0.32], [0.5, 0.35]);
    
    % Classification Branch (Top)
    rectangle('Position', [0.32, 0.6, 0.15, 0.25], 'FaceColor', [0.7, 0.85, 1], 'EdgeColor', [0, 0.4, 0.8], 'LineWidth', 2);
    text(0.395, 0.82, 'CLASSIFICATION', 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0, 0.3, 0.6]);
    text(0.395, 0.72, {'SVM (127 features)', 'or', 'ResNet18 (CNN)'}, 'HorizontalAlignment', 'center', 'FontSize', 9);
    
    % Segmentation Branch (Bottom)
    rectangle('Position', [0.32, 0.15, 0.15, 0.25], 'FaceColor', [1, 0.9, 0.8], 'EdgeColor', [0.8, 0.4, 0], 'LineWidth', 2);
    text(0.395, 0.37, 'SEGMENTATION', 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.6, 0.3, 0]);
    text(0.395, 0.27, {'HSV + Morphology', 'or', 'DeepLabv3+'}, 'HorizontalAlignment', 'center', 'FontSize', 9);
    
    % Arrows to next stage
    annotation('arrow', [0.47, 0.52], [0.72, 0.55]);
    annotation('arrow', [0.47, 0.52], [0.28, 0.45]);
    
    % Portion Estimation
    rectangle('Position', [0.52, 0.35, 0.12, 0.3], 'FaceColor', [0.8, 1, 0.8], 'EdgeColor', 'k', 'LineWidth', 2);
    text(0.58, 0.5, {'Portion', 'Estimation', '', 'Region Props', 'Area Ratio'}, 'HorizontalAlignment', 'center', 'FontSize', 9);
    
    % Arrow
    annotation('arrow', [0.64, 0.69], [0.5, 0.5]);
    
    % Calorie Calculation
    rectangle('Position', [0.69, 0.35, 0.12, 0.3], 'FaceColor', [1, 1, 0.7], 'EdgeColor', 'k', 'LineWidth', 2);
    text(0.75, 0.5, {'Calorie', 'Calculation', '', 'MyFCD', 'Database'}, 'HorizontalAlignment', 'center', 'FontSize', 9);
    
    % Arrow
    annotation('arrow', [0.81, 0.86], [0.5, 0.5]);
    
    % Output
    rectangle('Position', [0.86, 0.35, 0.12, 0.3], 'FaceColor', [0.5, 0.8, 0.5], 'EdgeColor', 'k', 'LineWidth', 2);
    text(0.92, 0.5, {'OUTPUT', '', 'Food Type', 'Calories', 'Nutrients'}, 'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', 'Color', 'w');
    
    xlim([0, 1]);
    ylim([0, 1]);
    
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_Complete_System_Architecture.png', figNum))); figNum = figNum + 1;
    close(f);
    fprintf('Saved: Fig33_Complete_System_Architecture.png\n');
    
    %% Figure 34: Training Process
    f = figure('Position', [100,100,1000,400], 'Color', 'w');
    
    subplot(1,2,1);
    % Simulated training curve
    epochs = 1:20;
    trainLoss = 2.5 * exp(-0.15 * epochs) + 0.1 + 0.05*rand(1,20);
    valLoss = 2.5 * exp(-0.12 * epochs) + 0.15 + 0.08*rand(1,20);
    
    plot(epochs, trainLoss, 'b-', 'LineWidth', 2); hold on;
    plot(epochs, valLoss, 'r--', 'LineWidth', 2);
    xlabel('Epoch'); ylabel('Loss');
    title('Training Progress (Example)', 'FontWeight', 'bold');
    legend('Training Loss', 'Validation Loss', 'Location', 'northeast');
    grid on;
    
    subplot(1,2,2);
    % Accuracy curve
    trainAcc = 1 - 0.8*exp(-0.2*epochs) + 0.02*rand(1,20);
    valAcc = 1 - 0.85*exp(-0.18*epochs) + 0.03*rand(1,20);
    
    plot(epochs, trainAcc*100, 'b-', 'LineWidth', 2); hold on;
    plot(epochs, valAcc*100, 'r--', 'LineWidth', 2);
    xlabel('Epoch'); ylabel('Accuracy (%)');
    title('Accuracy Over Training', 'FontWeight', 'bold');
    legend('Training', 'Validation', 'Location', 'southeast');
    grid on;
    ylim([0, 100]);
    
    sgtitle('Deep Learning Training Visualization', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, sprintf('Fig%02d_Training_Curves.png', figNum))); figNum = figNum + 1;
    close(f);
    fprintf('Saved: Fig34_Training_Curves.png\n');
    
    %% Summary
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║          DEEP LEARNING FIGURES GENERATED                     ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Fig30: ResNet18 Classification Architecture                 ║\n');
    fprintf('║  Fig31: DeepLabv3+ Segmentation Architecture                 ║\n');
    fprintf('║  Fig32: Classical vs Deep Learning Comparison                ║\n');
    fprintf('║  Fig33: Complete System Architecture                         ║\n');
    fprintf('║  Fig34: Training Curves Example                              ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
end
