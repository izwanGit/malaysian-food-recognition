%% TRAIN CNN - Transfer Learning with ResNet-18
% Fine-tunes a pre-trained CNN for Malaysian hawker food classification
%
% This script implements Deep Learning as required by the project guidelines.
% It uses Transfer Learning to leverage pre-trained features from ImageNet.
%
% Reference: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
%
% Usage:
%   trainCNN()  % Uses default paths
%   trainCNN('dataset/train', 'dataset/test')

function net = trainCNN(trainPath, testPath)
    %% Default paths
    if nargin < 1
        baseDir = fileparts(mfilename('fullpath'));
        projectRoot = fileparts(baseDir);
        trainPath = fullfile(projectRoot, 'dataset', 'train');
    end
    if nargin < 2
        testPath = fullfile(projectRoot, 'dataset', 'test');
    end
    
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║   DEEP LEARNING: Transfer Learning with ResNet-18          ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    
    %% Load Pre-trained Network
    fprintf('Loading pre-trained ResNet-18...\n');
    net = resnet18;
    inputSize = net.Layers(1).InputSize;
    fprintf('Input size: %d x %d x %d\n\n', inputSize(1), inputSize(2), inputSize(3));
    
    %% Prepare Image Datastores
    fprintf('─── PHASE 1: Loading Dataset ───\n\n');
    
    % Training data with augmentation
    imdsTrain = imageDatastore(trainPath, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    
    % Test data (no augmentation)
    if exist(testPath, 'dir')
        imdsTest = imageDatastore(testPath, ...
            'IncludeSubfolders', true, ...
            'LabelSource', 'foldernames');
        fprintf('Test set: %d images\n', numel(imdsTest.Files));
    else
        % Split training data if no test set
        [imdsTrain, imdsTest] = splitEachLabel(imdsTrain, 0.8, 'randomized');
    end
    
    fprintf('Training set: %d images\n', numel(imdsTrain.Files));
    fprintf('Classes: %s\n\n', strjoin(string(categories(imdsTrain.Labels)), ', '));
    
    %% Data Augmentation
    fprintf('─── PHASE 2: Data Augmentation ───\n\n');
    
    % Augmentation for training (resize + random transforms)
    augmenter = imageDataAugmenter( ...
        'RandRotation', [-20, 20], ...
        'RandXReflection', true, ...
        'RandYReflection', false, ...
        'RandXScale', [0.9, 1.1], ...
        'RandYScale', [0.9, 1.1]);
    
    augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
        'DataAugmentation', augmenter, ...
        'ColorPreprocessing', 'gray2rgb');
    
    % No augmentation for test (just resize)
    augTest = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
        'ColorPreprocessing', 'gray2rgb');
    
    fprintf('Augmentation: Rotation ±20°, Horizontal Flip, Scale 0.9-1.1x\n\n');
    
    %% Modify Network for Transfer Learning
    fprintf('─── PHASE 3: Network Modification ───\n\n');
    
    numClasses = numel(categories(imdsTrain.Labels));
    fprintf('Modifying final layers for %d classes...\n', numClasses);
    
    % Get layer graph
    lgraph = layerGraph(net);
    
    % Replace final fully connected layer
    newFcLayer = fullyConnectedLayer(numClasses, ...
        'Name', 'fc_food', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10);
    lgraph = replaceLayer(lgraph, 'fc1000', newFcLayer);
    
    % Replace classification layer
    newClassLayer = classificationLayer('Name', 'food_output');
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);
    
    fprintf('Done. New layers: fc_food (%d units) -> food_output\n\n', numClasses);
    
    %% Training Options
    fprintf('─── PHASE 4: Training Configuration ───\n\n');
    
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 32, ...
        'MaxEpochs', 10, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 5, ...
        'ValidationData', augTest, ...
        'ValidationFrequency', 50, ...
        'ValidationPatience', 5, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', 'auto');
    
    fprintf('Optimizer: SGD with Momentum\n');
    fprintf('Learning Rate: 0.001 (drop 10x every 5 epochs)\n');
    fprintf('Batch Size: 32\n');
    fprintf('Max Epochs: 10\n\n');
    
    %% Train the Network
    fprintf('─── PHASE 5: Training ───\n\n');
    fprintf('Starting training (this may take 20-40 minutes)...\n\n');
    
    trainedNet = trainNetwork(augTrain, lgraph, options);
    
    %% Evaluate on Test Set
    fprintf('\n─── PHASE 6: Evaluation ───\n\n');
    
    predictedLabels = classify(trainedNet, augTest);
    actualLabels = imdsTest.Labels;
    
    accuracy = sum(predictedLabels == actualLabels) / numel(actualLabels) * 100;
    fprintf('TEST ACCURACY: %.2f%%\n\n', accuracy);
    
    % Confusion Matrix
    confMat = confusionmat(actualLabels, predictedLabels);
    classNames = categories(imdsTrain.Labels);
    
    fprintf('Confusion Matrix:\n');
    fprintf('                 ');
    for c = 1:numClasses
        fprintf('%8s ', classNames{c}(1:min(8,end)));
    end
    fprintf('\n');
    fprintf('─────────────────');
    fprintf(repmat('─────────', 1, numClasses));
    fprintf('\n');
    
    for i = 1:numClasses
        fprintf('%12s  │ ', classNames{i}(1:min(12,end)));
        for j = 1:numClasses
            if i == j
                fprintf('%7d* ', confMat(i,j));
            else
                fprintf('%8d ', confMat(i,j));
            end
        end
        fprintf('\n');
    end
    fprintf('\n');
    
    %% Save Model
    fprintf('─── PHASE 7: Saving Model ───\n\n');
    
    modelsPath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'models');
    if ~exist(modelsPath, 'dir')
        mkdir(modelsPath);
    end
    
    modelFile = fullfile(modelsPath, 'foodCNN.mat');
    save(modelFile, 'trainedNet', 'classNames', 'accuracy', 'confMat');
    fprintf('Model saved to: %s\n\n', modelFile);
    
    %% Summary
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║              DEEP LEARNING TRAINING COMPLETE               ║\n');
    fprintf('╠════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Test Accuracy: %5.2f%%                                    ║\n', accuracy);
    fprintf('║  Model: ResNet-18 (Transfer Learning)                      ║\n');
    fprintf('║  Classes: %d                                                ║\n', numClasses);
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    
    net = trainedNet;
end
