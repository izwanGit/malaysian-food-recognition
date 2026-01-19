%% GENERATE CNN REPORT FIGURES
% Creates professional visualizations for academic report
% Run this AFTER training completes

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║   GENERATING CNN REPORT FIGURES                            ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

projectRoot = pwd;
modelPath = fullfile(projectRoot, 'models', 'foodCNN.mat');
testPath = fullfile(projectRoot, 'dataset', 'test');
outputDir = fullfile(projectRoot, 'report_figures', 'cnn');

% Create output directory
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% Load trained model
fprintf('Loading trained CNN model...\n');
if ~exist(modelPath, 'file')
    error('Model not found! Run training first.');
end
load(modelPath, 'trainedNet', 'classNames', 'accuracy');
fprintf('Model loaded. Training accuracy: %.2f%%\n\n', accuracy);

%% Load test data
fprintf('Loading test dataset...\n');
imdsTest = imageDatastore(testPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest.Labels = categorical(imdsTest.Labels);

% Get input size from network
inputSize = trainedNet.Layers(1).InputSize;

% Prepare test data with proper preprocessing
imdsTest.ReadFcn = @(loc) preprocessTestImage(loc, inputSize(1:2));

fprintf('Test set: %d images\n\n', numel(imdsTest.Files));

%% 1. CONFUSION MATRIX
fprintf('─── Figure 1: Confusion Matrix ───\n');
predictedLabels = classify(trainedNet, imdsTest);
actualLabels = imdsTest.Labels;

figure('Position', [100 100 800 700], 'Color', 'white');
cm = confusionchart(actualLabels, predictedLabels);
cm.Title = 'SqueezeNet Classification Confusion Matrix';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
cm.FontSize = 12;
saveas(gcf, fullfile(outputDir, 'confusion_matrix.png'));
fprintf('Saved: confusion_matrix.png\n\n');

%% 2. PER-CLASS ACCURACY BAR CHART
fprintf('─── Figure 2: Per-Class Accuracy ───\n');
classes = categories(actualLabels);
numClasses = numel(classes);
classAccuracy = zeros(numClasses, 1);

for i = 1:numClasses
    idx = actualLabels == classes{i};
    classAccuracy(i) = sum(predictedLabels(idx) == actualLabels(idx)) / sum(idx) * 100;
end

figure('Position', [100 100 900 500], 'Color', 'white');
b = bar(classAccuracy, 'FaceColor', 'flat');
% Color gradient from red (low) to green (high)
for i = 1:numClasses
    if classAccuracy(i) >= 80
        b.CData(i,:) = [0.2 0.7 0.3]; % Green
    elseif classAccuracy(i) >= 60
        b.CData(i,:) = [0.9 0.7 0.1]; % Yellow
    else
        b.CData(i,:) = [0.8 0.2 0.2]; % Red
    end
end
set(gca, 'XTickLabel', strrep(classes, '_', ' '), 'XTickLabelRotation', 45);
ylabel('Accuracy (%)');
xlabel('Food Class');
title('Per-Class Classification Accuracy (SqueezeNet)');
ylim([0 100]);
grid on;
% Add value labels on bars
for i = 1:numClasses
    text(i, classAccuracy(i) + 2, sprintf('%.1f%%', classAccuracy(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end
saveas(gcf, fullfile(outputDir, 'per_class_accuracy.png'));
fprintf('Saved: per_class_accuracy.png\n\n');

%% 3. SAMPLE PREDICTIONS GRID
fprintf('─── Figure 3: Sample Predictions ───\n');
figure('Position', [100 100 1200 800], 'Color', 'white');

% Get 12 random test images
numSamples = 12;
randIdx = randperm(numel(imdsTest.Files), min(numSamples, numel(imdsTest.Files)));

for i = 1:numel(randIdx)
    subplot(3, 4, i);
    img = readimage(imdsTest, randIdx(i));
    imshow(img);
    
    actual = string(actualLabels(randIdx(i)));
    predicted = string(predictedLabels(randIdx(i)));
    
    if actual == predicted
        titleColor = [0 0.5 0]; % Green for correct
        symbol = '✓';
    else
        titleColor = [0.8 0 0]; % Red for wrong
        symbol = '✗';
    end
    
    title(sprintf('%s %s\n(Actual: %s)', symbol, strrep(predicted, '_', ' '), ...
        strrep(actual, '_', ' ')), 'Color', titleColor, 'FontSize', 9);
end
sgtitle('Sample Predictions: SqueezeNet Transfer Learning', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(outputDir, 'sample_predictions.png'));
fprintf('Saved: sample_predictions.png\n\n');

%% 4. MODEL SUMMARY
fprintf('─── Figure 4: Model Architecture Summary ───\n');
figure('Position', [100 100 600 400], 'Color', 'white');

% Create summary text
summaryText = {
    '╔═══════════════════════════════════════╗';
    '║    SQUEEZENET TRANSFER LEARNING       ║';
    '╠═══════════════════════════════════════╣';
    sprintf('║  Base Model: SqueezeNet v1.1          ║');
    sprintf('║  Input Size: %d x %d x %d             ║', inputSize(1), inputSize(2), inputSize(3));
    sprintf('║  Output Classes: %d                    ║', numClasses);
    sprintf('║  Total Layers: %d                     ║', numel(trainedNet.Layers));
    '╠═══════════════════════════════════════╣';
    sprintf('║  Test Accuracy: %.2f%%               ║', mean(predictedLabels == actualLabels) * 100);
    sprintf('║  Test Images: %d                     ║', numel(imdsTest.Files));
    '╚═══════════════════════════════════════╝';
};

text(0.5, 0.5, summaryText, 'FontName', 'Courier', 'FontSize', 11, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
axis off;
title('Model Performance Summary', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(outputDir, 'model_summary.png'));
fprintf('Saved: model_summary.png\n\n');

%% 5. SVM vs CNN COMPARISON (if SVM model exists)
svmModelPath = fullfile(projectRoot, 'models', 'foodClassifier.mat');
if exist(svmModelPath, 'file')
    fprintf('─── Figure 5: SVM vs CNN Comparison ───\n');
    
    % Assume SVM accuracy from earlier training (you may need to adjust)
    svmAccuracy = 79.56; % From earlier SVM training
    cnnAccuracy = mean(predictedLabels == actualLabels) * 100;
    
    figure('Position', [100 100 600 500], 'Color', 'white');
    data = [svmAccuracy, cnnAccuracy];
    b = bar(data, 0.6, 'FaceColor', 'flat');
    b.CData(1,:) = [0.3 0.5 0.8]; % Blue for SVM
    b.CData(2,:) = [0.2 0.7 0.3]; % Green for CNN
    set(gca, 'XTickLabel', {'SVM (Classical)', 'SqueezeNet (Deep Learning)'});
    ylabel('Accuracy (%)');
    title('Classification Accuracy: SVM vs Deep Learning');
    ylim([0 100]);
    grid on;
    
    % Add value labels
    for i = 1:2
        text(i, data(i) + 2, sprintf('%.2f%%', data(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
    end
    
    saveas(gcf, fullfile(outputDir, 'svm_vs_cnn_comparison.png'));
    fprintf('Saved: svm_vs_cnn_comparison.png\n\n');
end

%% Final Summary
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('ALL FIGURES SAVED TO: %s\n', outputDir);
fprintf('═══════════════════════════════════════════════════════════════\n\n');

% List all generated files
files = dir(fullfile(outputDir, '*.png'));
fprintf('Generated files:\n');
for i = 1:numel(files)
    fprintf('  • %s\n', files(i).name);
end

close all;
fprintf('\nDone! Copy these images to your report.\n');

%% Helper Function
function img = preprocessTestImage(loc, targetSize)
    img = imread(loc);
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    elseif size(img, 3) == 4
        img = img(:,:,1:3);
    end
    img = imresize(img, targetSize);
    img = uint8(img);
end
