%% DEBUG FEATURES - Benchmark Feature Extraction
fprintf('=== DEBUGGING FEATURE EXTRACTION ===\n');

% Find a sample image
imgDir = fullfile(pwd, 'dataset', 'train');
files = dir(fullfile(imgDir, '*', '*.jpg'));
if isempty(files)
    files = dir(fullfile(imgDir, '*', '*.png'));
end

if isempty(files)
    error('No images found for debugging!');
end

imgPath = fullfile(files(1).folder, files(1).name);
fprintf('Testing on image: %s\n', imgPath);

% Load image
tic;
img = imread(imgPath);
t_load = toc;
fprintf('Load time: %.4f s\n', t_load);

% Preprocess
tic;
processed = preprocessImage(img);
t_process = toc;
fprintf('Preprocess time: %.4f s\n', t_process);

% Extract Features (The suspect)
tic;
[features, names] = extractFeatures(processed);
t_extract = toc;
fprintf('Feature Extraction Time: %.4f s\n', t_extract);
fprintf('Feature Dimension: %d\n', length(features));

% HOG Check
hogIndices = contains(names, 'HOG');
fprintf('Number of HOG features: %d\n', sum(hogIndices));

% Estimate total time for 11,200 images
totalTime = t_extract * 11200 / 60;
fprintf('Estimated total extraction time: %.2f minutes\n', totalTime);

fprintf('=== DEBUG COMPLETE ===\n');
