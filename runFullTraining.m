%% RUN FULL TRAINING - Split dataset and train A++ model
% Simplified version for terminal run

fprintf('=== A++ MODEL TRAINING PIPELINE ===\n');

baseDir = '/Applications/MATLAB_R2025a.app/toolbox/images/imdata/CSC566_MINI GROUP PROJECT_HAWKER FOOD CALORIE_TEAMONE';
cd(baseDir);

% Add subfolders manually
addpath(fullfile(baseDir, 'classification'));
addpath(fullfile(baseDir, 'preprocessing'));
addpath(fullfile(baseDir, 'features'));
addpath(fullfile(baseDir, 'segmentation'));
addpath(fullfile(baseDir, 'portion'));
addpath(fullfile(baseDir, 'calories'));
addpath(fullfile(baseDir, 'models'));

rehash;

fprintf('STEP 1: Splitting dataset...\n');
splitDataset(0.2);

fprintf('STEP 2: Training model...\n');
model = trainClassifier();

fprintf('\nSTEP 3: Generating Visualizations...\n');
% Save confusion matrix
resultsFile = fullfile(baseDir, 'results', 'confusion_matrix.png');
plotConfusionMatrix(fullfile(baseDir, 'models', 'foodClassifier.mat'), resultsFile);

% Save augmentation samples
visualizeAugmentation();

fprintf('\nTRAINING COMPLETE\n');
fprintf('CV Accuracy: %.2f%%\n', model.trainStats.cvAccuracy * 100);
fprintf('Train Accuracy: %.2f%%\n', model.trainStats.trainAccuracy * 100);
fprintf('Model saved!\n');
