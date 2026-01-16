%% PLOT CONFUSION MATRIX - Visualize Classification Results
% Creates a professional confusion matrix visualization
%
% Syntax:
%   plotConfusionMatrix()
%   plotConfusionMatrix(modelPath)
%   fig = plotConfusionMatrix(modelPath, savePath)
%
% Example:
%   plotConfusionMatrix()  % Uses default model path
%   plotConfusionMatrix('models/foodClassifier.mat', 'results/confusion.png')

function fig = plotConfusionMatrix(modelPath, savePath)
    %% Load model
    if nargin < 1
        baseDir = fileparts(mfilename('fullpath'));
        projectRoot = fileparts(baseDir);
        modelPath = fullfile(projectRoot, 'models', 'foodClassifier.mat');
    end
    
    if ~exist(modelPath, 'file')
        error('Model not found. Run trainClassifier() first.');
    end
    
    loaded = load(modelPath, 'model');
    model = loaded.model;
    
    %% Extract data
    confMat = model.confusionMatrix;
    classNames = model.classNames;
    numClasses = length(classNames);
    
    % Make nice display names
    displayNames = cellfun(@(x) strrep(x, '_', ' '), classNames, 'UniformOutput', false);
    displayNames = cellfun(@(x) [upper(x(1)) x(2:end)], displayNames, 'UniformOutput', false);
    
    %% Create figure
    fig = figure('Name', 'Confusion Matrix', ...
                 'Position', [100, 100, 900, 750], ...
                 'Color', 'white');
    
    %% Plot heatmap
    % Normalize for percentage display
    confMatNorm = confMat ./ sum(confMat, 2) * 100;
    
    imagesc(confMatNorm);
    colormap(flipud(hot));
    colorbar('Label', 'Percentage (%)');
    caxis([0 100]);
    
    %% Add text annotations
    for i = 1:numClasses
        for j = 1:numClasses
            count = confMat(i, j);
            pct = confMatNorm(i, j);
            
            if pct > 50
                textColor = 'white';
            else
                textColor = 'black';
            end
            
            text(j, i, sprintf('%d\n(%.1f%%)', count, pct), ...
                 'HorizontalAlignment', 'center', ...
                 'VerticalAlignment', 'middle', ...
                 'FontSize', 10, ...
                 'FontWeight', 'bold', ...
                 'Color', textColor);
        end
    end
    
    %% Labels and title
    set(gca, 'XTick', 1:numClasses, 'XTickLabel', displayNames, ...
             'YTick', 1:numClasses, 'YTickLabel', displayNames, ...
             'XTickLabelRotation', 45, ...
             'FontSize', 11);
    
    xlabel('Predicted Class', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Actual Class', 'FontSize', 13, 'FontWeight', 'bold');
    
    cvAcc = model.trainStats.cvAccuracy * 100;
    cvStd = model.trainStats.cvStd * 100;
    title(sprintf('Confusion Matrix - Malaysian Hawker Food Classification\n5-Fold CV Accuracy: %.2f%% ± %.2f%%', ...
          cvAcc, cvStd), 'FontSize', 14, 'FontWeight', 'bold');
    
    %% Add metrics summary
    precision = mean(model.perClassMetrics.precision) * 100;
    recall = mean(model.perClassMetrics.recall) * 100;
    f1 = mean(model.perClassMetrics.f1Score) * 100;
    
    annotation('textbox', [0.02, 0.02, 0.3, 0.08], ...
               'String', sprintf('Avg Precision: %.1f%%  |  Avg Recall: %.1f%%  |  Avg F1: %.1f%%', ...
                                 precision, recall, f1), ...
               'FontSize', 10, 'EdgeColor', 'none', ...
               'HorizontalAlignment', 'left');
    
    %% Save if path provided
    if nargin >= 2 && ~isempty(savePath)
        saveas(fig, savePath);
        fprintf('Confusion matrix saved to: %s\n', savePath);
    end
end
