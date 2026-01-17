%% PLOT REAL TRAINING CURVES
% Extracts REAL training data from trained models and creates accurate plots

% Setup
outputDir = 'report_figures';
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

fprintf('\n=== GENERATING REAL TRAINING CURVES ===\n\n');

%% CNN Training Curves (REAL DATA)
fprintf('Loading CNN model...\n');
load('models/foodCNN.mat');
ti = cnnModel.trainInfo;

f = figure('Position', [100,100,1200,500], 'Color', 'w');

subplot(1,2,1);
plot(ti.TrainingLoss, 'b-', 'LineWidth', 1.5);
hold on;
valIdx = find(~isnan(ti.ValidationLoss));
plot(valIdx, ti.ValidationLoss(valIdx), 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Iteration', 'FontSize', 12);
ylabel('Loss', 'FontSize', 12);
title('CNN Training Loss', 'FontSize', 14, 'FontWeight', 'bold');
legend('Training Loss', 'Validation Loss', 'Location', 'northeast');
grid on;

subplot(1,2,2);
plot(ti.TrainingAccuracy, 'b-', 'LineWidth', 1.5);
hold on;
valIdx = find(~isnan(ti.ValidationAccuracy));
plot(valIdx, ti.ValidationAccuracy(valIdx), 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Iteration', 'FontSize', 12);
ylabel('Accuracy (%)', 'FontSize', 12);
title(sprintf('CNN Training Accuracy (Final: %.2f%%)', ti.FinalValidationAccuracy), 'FontSize', 14, 'FontWeight', 'bold');
legend('Training Accuracy', 'Validation Accuracy', 'Location', 'southeast');
grid on;

sgtitle(sprintf('CNN (Custom) Training Progress - %s', cnnModel.trainDate), 'FontSize', 16, 'FontWeight', 'bold');
saveas(f, fullfile(outputDir, 'Fig35_CNN_Training_Curves_REAL.png'));
close(f);
fprintf('Saved: Fig35_CNN_Training_Curves_REAL.png\n');

%% DeepLabv3+ Training Curves (REAL DATA)
fprintf('Loading DeepLabv3+ model...\n');
load('models/foodSegmentationDL.mat');
ti2 = segModel.trainInfo;

if isstruct(ti2) && isfield(ti2, 'TrainingLoss')
    f = figure('Position', [100,100,1200,500], 'Color', 'w');
    
    subplot(1,2,1);
    plot(ti2.TrainingLoss, 'b-', 'LineWidth', 1.5);
    hold on;
    valIdx = find(~isnan(ti2.ValidationLoss));
    if ~isempty(valIdx)
        plot(valIdx, ti2.ValidationLoss(valIdx), 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
        legend('Training Loss', 'Validation Loss', 'Location', 'northeast');
    else
        legend('Training Loss', 'Location', 'northeast');
    end
    xlabel('Iteration', 'FontSize', 12);
    ylabel('Loss', 'FontSize', 12);
    title('DeepLabv3+ Training Loss', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    
    subplot(1,2,2);
    plot(ti2.TrainingAccuracy, 'b-', 'LineWidth', 1.5);
    hold on;
    valIdx = find(~isnan(ti2.ValidationAccuracy));
    if ~isempty(valIdx)
        plot(valIdx, ti2.ValidationAccuracy(valIdx), 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
        legend('Training Accuracy', 'Validation Accuracy', 'Location', 'southeast');
    else
        legend('Training Accuracy', 'Location', 'southeast');
    end
    xlabel('Iteration', 'FontSize', 12);
    ylabel('Accuracy (%)', 'FontSize', 12);
    
    if isfield(ti2, 'FinalValidationAccuracy')
        title(sprintf('DeepLabv3+ Training Accuracy (Final: %.2f%%)', ti2.FinalValidationAccuracy), 'FontSize', 14, 'FontWeight', 'bold');
    else
        title('DeepLabv3+ Training Accuracy', 'FontSize', 14, 'FontWeight', 'bold');
    end
    grid on;
    
    sgtitle(sprintf('DeepLabv3+ Training Progress - %s', segModel.trainDate), 'FontSize', 16, 'FontWeight', 'bold');
    saveas(f, fullfile(outputDir, 'Fig36_DeepLabv3_Training_Curves_REAL.png'));
    close(f);
    fprintf('Saved: Fig36_DeepLabv3_Training_Curves_REAL.png\n');
else
    fprintf('DeepLabv3+ trainInfo does not contain training curves data.\n');
    fprintf('trainInfo type: %s\n', class(ti2));
end

fprintf('\n=== DONE - REAL TRAINING CURVES SAVED ===\n');
