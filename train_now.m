% Run Training Script - Inline version
baseDir = '/Applications/MATLAB_R2025a.app/toolbox/images/imdata/CSC566_MINI GROUP PROJECT_HAWKER FOOD CALORIE_TEAMONE';
cd(baseDir);

% Add all subdirectories to path
addpath(fullfile(baseDir, 'classification'));
addpath(fullfile(baseDir, 'preprocessing'));
addpath(fullfile(baseDir, 'features'));
addpath(fullfile(baseDir, 'segmentation'));
addpath(fullfile(baseDir, 'portion'));
addpath(fullfile(baseDir, 'calories'));

disp('Paths added. Starting training...');
disp('=== STEP 1: SPLIT DATASET ===');
splitDataset(0.2);
disp('=== STEP 2: TRAIN MODEL ===');
model = trainClassifier();
fprintf('CV Accuracy: %.2f%%\n', model.trainStats.cvAccuracy*100);
disp('DONE!');
exit;
