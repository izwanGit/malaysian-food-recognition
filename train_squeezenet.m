%% TRAIN SQUEEZENET - Deep Learning Script
% No function line to avoid path resolution issues
rehash;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║   DEEP LEARNING: Transfer Learning with SqueezeNet         ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

% Set Paths
projectRoot = pwd;
trainPath = fullfile(projectRoot, 'dataset', 'train');
testPath = fullfile(projectRoot, 'dataset', 'test');

addpath(genpath(projectRoot));

%% Load Pre-trained Network
fprintf('Loading pre-trained SqueezeNet...\n');
try
    net = squeezenet;
catch ME
    fprintf('Error loading squeezenet: %s\n', ME.message);
    error('Could not load squeezenet. This is unexpected as it was detected earlier.');
end

inputSize = net.Layers(1).InputSize;
fprintf('Input size: %d x %d x %d\n\n', inputSize(1), inputSize(2), inputSize(3));

%% Prepare Image Datastores
fprintf('─── PHASE 1: Loading Dataset ───\n\n');

imdsTrain = imageDatastore(trainPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Ensure consistent labels
imdsTrain.Labels = categorical(imdsTrain.Labels);

if exist(testPath, 'dir')
    imdsTest = imageDatastore(testPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    imdsTest.Labels = categorical(imdsTest.Labels);
else
    [imdsTrain, imdsTest] = splitEachLabel(imdsTrain, 0.8, 'randomized');
end

fprintf('Training set: %d images (%d classes)\n', numel(imdsTrain.Files), numel(categories(imdsTrain.Labels)));
fprintf('Test set: %d images (%d classes)\n', numel(imdsTest.Files), numel(categories(imdsTest.Labels)));

%% Data Augmentation
fprintf('─── PHASE 2: Data Augmentation ───\n\n');

augmenter = imageDataAugmenter( ...
    'RandRotation', [-20, 20], ...
    'RandXReflection', true);

augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter);

% Robust resizing for validation data
imdsTest.ReadFcn = @(loc) imresize(im2double(imread(loc)), inputSize(1:2));
% Ensure 3-channel (some images might be grayscale)
imdsTest.ReadFcn = @(loc) gray2rgb_and_resize(loc, inputSize(1:2));

fprintf('Input size: %d x %d\n', inputSize(1), inputSize(2));

%% Network Modification
fprintf('─── PHASE 3: Network Modification ───\n\n');

numClasses = numel(categories(imdsTrain.Labels));
lgraph = layerGraph(net);

% SqueezeNet doesn't have a simple FC layer at the end. 
% It uses a 1x1 convolution (conv10) followed by pooling.
% We replace 'conv10' with a new 1x1 convolution for our number of classes.

newConvLayer = convolution2dLayer([1 1], numClasses, ...
    'Name', 'new_conv10', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

lgraph = replaceLayer(lgraph, 'conv10', newConvLayer);

% Replace classification layer
newClassLayer = classificationLayer('Name', 'food_output');
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);

%% Training Options
fprintf('─── PHASE 4: Training Configuration ───\n\n');

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ... % Smaller batch for stable gradients
    'MaxEpochs', 10, ...
    'InitialLearnRate', 0.0003, ... % Lower LR for fine-tuning pre-trained weights
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 5, ...
    'ValidationData', imdsTest, ...
    'ValidationFrequency', 50, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'none', ...
    'ExecutionEnvironment', 'cpu'); % Single CPU for reliability


%% Train
fprintf('─── PHASE 5: Training ───\n\n');
trainedNet = trainNetwork(augTrain, lgraph, options);

%% Evaluate
fprintf('\n─── PHASE 6: Evaluation ───\n\n');
predictedLabels = classify(trainedNet, imdsTest);
actualLabels = imdsTest.Labels;
accuracy = sum(predictedLabels == actualLabels) / numel(actualLabels) * 100;
fprintf('TEST ACCURACY: %.2f%%\n\n', accuracy);

%% Save
modelsPath = fullfile(projectRoot, 'models');
if ~exist(modelsPath, 'dir'), mkdir(modelsPath); end
classNames = categories(imdsTrain.Labels);
save(fullfile(modelsPath, 'foodCNN.mat'), 'trainedNet', 'classNames', 'accuracy');

fprintf('DONE!\n');

%% Helper Function
function img = gray2rgb_and_resize(loc, targetSize)
    img = imread(loc);
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    elseif size(img, 3) == 4
        img = img(:,:,1:3);
    end
    img = imresize(img, targetSize);
    % SqueezeNet training expects [0, 255] range for its pre-trained mean subtraction
    img = uint8(img); 
end

exit;
