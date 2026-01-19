%% TRAIN RESNET ONNX - Transfer Learning from Open Source Model
% This script imports an ONNX ResNet-18 model and fine-tunes it.
% It bypasses the official MATLAB Download restrictions.

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║   DEEP LEARNING: Transfer Learning with ResNet-18 (ONNX)   ║\n');
╚════════════════════════════════════════════════════════════╝\n\n');

% Set Paths
projectRoot = pwd;
trainPath = fullfile(projectRoot, 'dataset', 'train');
testPath = fullfile(projectRoot, 'dataset', 'test');
modelPath = fullfile(projectRoot, 'models', 'resnet18-v2-7.onnx');

addpath(genpath(projectRoot));

%% Load ONNX Network
fprintf('Importing ResNet-18 from ONNX model...\n');
if ~exist(modelPath, 'file')
    error('ONNX model file not found at: %s\nDownloading might still be in progress.', modelPath);
end

try
    % Use importONNXNetwork to load the pre-trained weights
    % Note: ONNX variants have different input sizes, typically 224x224
    net_onnx = importONNXNetwork(modelPath, 'OutputLayerType', 'classification');
    
    % Get input size
    inputSize = [224 224 3]; % Standard for ResNet ONNX
    fprintf('Model imported successfully.\n\n');
catch ME
    fprintf('Error importing ONNX: %s\n', ME.message);
    if contains(ME.message, 'Support Package')
        error('ONNX Support Package missing. Please run SqueezeNet or install the ONNX converter.');
    else
        rethrow(ME);
    end
end

%% Prepare Image Datastores
fprintf('─── PHASE 1: Loading Dataset ───\n\n');

imdsTrain = imageDatastore(trainPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTrain.Labels = categorical(imdsTrain.Labels);

if exist(testPath, 'dir')
    imdsTest = imageDatastore(testPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    imdsTest.Labels = categorical(imdsTest.Labels);
else
    [imdsTrain, imdsTest] = splitEachLabel(imdsTrain, 0.8, 'randomized');
end

fprintf('Training set: %d images (%d classes)\n', numel(imdsTrain.Files), numel(categories(imdsTrain.Labels)));

%% Data Augmentation
fprintf('─── PHASE 2: Data Augmentation ───\n\n');

augmenter = imageDataAugmenter('RandRotation', [-20, 20], 'RandXReflection', true);

augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', augmenter);
imdsTest.ReadFcn = @(loc) imresize(cat(3, im2single(imread(loc)), im2single(imread(loc)), im2single(imread(loc))), inputSize(1:2));
% Correcting ReadFcn logic
imdsTest.ReadFcn = @(loc) preprocess_image_onnx(loc, inputSize(1:2));

%% Modify Network
fprintf('─── PHASE 3: Network Modification ───\n\n');

numClasses = numel(categories(imdsTrain.Labels));
lgraph = layerGraph(net_onnx);

% resnet18-v2-7.onnx typical structure:
% It might use different layer names than the MATLAB version.
% We usually need to find 'resnetv22_dense0_fwd' (last FC) and its output.

% Let's inspect the layers briefly to find the replacement targets
layerNames = {lgraph.Layers.Name};
fprintf('Last 5 layers:\n');
disp(layerNames(end-4:end)');

% Target the final fully connected and output layers
% For resnet18-v2-7, it's often 'resnetv22_dense0_fwd'
try
    % Automatic detection of final layer
    lastFcName = '';
    for i = length(lgraph.Layers):-1:1
        if isa(lgraph.Layers(i), 'nnet.cnn.layer.FullyConnectedLayer')
            lastFcName = lgraph.Layers(i).Name;
            break;
        end
    end
    
    if isempty(lastFcName)
        error('Could not find FullyConnectedLayer in ONNX model automatically.');
    end
    
    newFcLayer = fullyConnectedLayer(numClasses, 'Name', 'fc_food', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
    lgraph = replaceLayer(lgraph, lastFcName, newFcLayer);
    
    % Replace classification output layer
    newClassLayer = classificationLayer('Name', 'food_output');
    lgraph = replaceLayer(lgraph, lgraph.Layers(end).Name, newClassLayer);
catch ME
    fprintf('Layer replacement failed: %s\n', ME.message);
    error('Ensure ONNX model topology is compatible.');
end

%% Training Options
fprintf('─── PHASE 4: Training Configuration ───\n\n');

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 0.0001, ... % Lower LR for ONNX fine-tuning
    'ValidationData', imdsTest, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'none', ...
    'ExecutionEnvironment', 'auto');

%% Train
fprintf('─── PHASE 5: Training ───\n\n');
trainedNet = trainNetwork(augTrain, lgraph, options);

%% Save
modelsPath = fullfile(projectRoot, 'models');
if ~exist(modelsPath, 'dir'), mkdir(modelsPath); end
classNames = categories(imdsTrain.Labels);
accuracy = 0; % Calculated later
save(fullfile(modelsPath, 'foodCNN_resnet.mat'), 'trainedNet', 'classNames');
fprintf('Model saved as foodCNN_resnet.mat\nDone!\n');

%% Helper
function img = preprocess_image_onnx(loc, targetSize)
    img = imread(loc);
    if size(img, 3) == 1, img = cat(3, img, img, img); end
    img = imresize(img, targetSize);
    img = im2single(img);
end
exit;
