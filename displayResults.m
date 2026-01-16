%% DISPLAY RESULTS - Visualize Analysis Results
% Creates a comprehensive visualization of food analysis results
%
% Syntax:
%   displayResults(results)
%   fig = displayResults(results)
%   fig = displayResults(results, 'SavePath', 'output.png')
%
% Inputs:
%   results  - Results struct from analyzeHawkerFood()
%   SavePath - Optional path to save the figure
%
% Outputs:
%   fig - Figure handle

function fig = displayResults(results, varargin)
    %% Parse inputs
    p = inputParser;
    addRequired(p, 'results', @isstruct);
    addParameter(p, 'SavePath', '', @ischar);
    parse(p, results, varargin{:});
    
    savePath = p.Results.SavePath;
    
    %% Create figure
    fig = figure('Name', 'Hawker Food Analysis Results', ...
                 'Position', [100, 100, 1200, 700], ...
                 'Color', 'white');
    
    %% Subplot 1: Original Image
    subplot(2, 3, 1);
    imshow(results.originalImage);
    title('Original Image', 'FontSize', 12, 'FontWeight', 'bold');
    
    %% Subplot 2: Pre-processed Image
    subplot(2, 3, 2);
    imshow(results.processedImage);
    title('Pre-processed Image', 'FontSize', 12, 'FontWeight', 'bold');
    
    %% Subplot 3: Segmentation
    subplot(2, 3, 3);
    imshow(results.segmentedImage);
    title('Food Segmentation', 'FontSize', 12, 'FontWeight', 'bold');
    
    %% Subplot 4: Food Mask
    subplot(2, 3, 4);
    imshow(results.mask);
    title(sprintf('Food Mask (%.1f%% coverage)', ...
          sum(results.mask(:)) / numel(results.mask) * 100), ...
          'FontSize', 12, 'FontWeight', 'bold');
    
    %% Subplot 5: Ingredient Regions
    subplot(2, 3, 5);
    if isfield(results, 'labeledRegions') && ~isempty(results.labeledRegions)
        labelOverlay = label2rgb(results.labeledRegions, 'jet', 'k', 'shuffle');
        imshow(labelOverlay);
    else
        imshow(results.mask);
    end
    title('Ingredient Regions', 'FontSize', 12, 'FontWeight', 'bold');
    
    %% Subplot 6: Results Summary
    subplot(2, 3, 6);
    axis off;
    
    % Create text box with results
    nutrition = results.nutrition;
    
    textContent = {
        sprintf('\\bf{%s}', upper(strrep(results.foodClass, '_', ' '))), ...
        '', ...
        sprintf('Confidence: %.1f%%', results.confidence * 100), ...
        sprintf('Portion: %s (%.2fx)', results.portionLabel, results.portionRatio), ...
        '', ...
        '--- Nutritional Information ---', ...
        '', ...
        sprintf('\\bf{Calories: %d kcal}', results.calories), ...
        sprintf('  (%.0f%% Daily Value)', nutrition.caloriesDV), ...
        '', ...
        sprintf('Protein: %.1f g (%.0f%% DV)', nutrition.protein, nutrition.proteinDV), ...
        sprintf('Carbohydrates: %.1f g (%.0f%% DV)', nutrition.carbs, nutrition.carbsDV), ...
        sprintf('Fat: %.1f g (%.0f%% DV)', nutrition.fat, nutrition.fatDV), ...
        '', ...
        sprintf('Reference: %s', nutrition.referenceServing), ...
        '', ...
        sprintf('Processing time: %.2f s', results.processingTime)
    };
    
    text(0.1, 0.95, textContent, 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'FontSize', 11, ...
         'Interpreter', 'tex', 'FontName', 'FixedWidth');
    
    title('Analysis Results', 'FontSize', 12, 'FontWeight', 'bold');
    
    %% Add overall title
    sgtitle(['Malaysian Hawker Food Analysis - ', ...
             strrep(results.foodClass, '_', ' ')], ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    %% Save if requested
    if ~isempty(savePath)
        saveas(fig, savePath);
        fprintf('Results saved to: %s\n', savePath);
    end
end
