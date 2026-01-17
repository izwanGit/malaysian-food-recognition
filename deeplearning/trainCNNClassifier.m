%% TRAIN CNN CLASSIFIER - Deep Learning Food Classification with Transfer Learning
% Trains a CNN classifier using transfer learning with pretrained ResNet18
%
% This demonstrates how image preprocessing improves deep learning performance
%
% Syntax:
%   net = trainCNNClassifier()
%   net = trainCNNClassifier(datasetPath)
%   net = trainCNNClassifier(datasetPath, options)
%
% Requirements:
%   - Deep Learning Toolbox
%   - Deep Learning Toolbox Model for ResNet-18 Network
%
% Inputs:
%   datasetPath - Path to dataset with class subfolders
%   options     - Optional struct with training options
%
% Outputs:
%   net - Trained CNN network

function net = trainCNNClassifier(datasetPath, options)
    %% Check for Deep Learning Toolbox
    if ~license('test', 'Neural_Network_Toolbox')
        error('Deep Learning Toolbox is required. Please install it.');
    end
    
    %% Default parameters
    if nargin < 1
        projectRoot = fileparts(mfilename('fullpath'));
        datasetPath = fullfile(projectRoot, '..', 'dataset', 'train');
    end
    
    if nargin < 2
        options = struct();
    end
    
    % Training options
    maxEpochs = getfield_default(options, 'maxEpochs', 10);
    miniBatchSize = getfield_default(options, 'miniBatchSize', 32);
    initialLearnRate = getfield_default(options, 'initialLearnRate', 0.0001);
    validationSplit = getfield_default(options, 'validationSplit', 0.2);
    usePreprocessing = getfield_default(options, 'usePreprocessing', true);
    
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║     CNN FOOD CLASSIFIER - Transfer Learning with ResNet18    ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
    
    %% Load pretrained ResNet18
    fprintf('Loading pretrained ResNet18...\n');
    
    try
        baseNet = resnet18;
    catch
        error(['ResNet18 not available. Install with:\n' ...
               '  >> matlab.addons.install("resnet18")']);
    end
    
    inputSize = baseNet.Layers(1).InputSize;
    fprintf('  Input size: %dx%dx%d\n', inputSize(1), inputSize(2), inputSize(3));
    
    %% Create image datastore
    fprintf('\nLoading dataset: %s\n', datasetPath);
    
    imds = imageDatastore(datasetPath, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    
    numClasses = numel(categories(imds.Labels));
    numImages = numel(imds.Files);
    
    fprintf('  Found %d images in %d classes\n', numImages, numClasses);
    fprintf('  Classes: %s\n', strjoin(categories(imds.Labels), ', '));
    
    % Count per class
    labelCounts = countEachLabel(imds);
    disp(labelCounts);
    
    %% Split into training and validation
    [imdsTrain, imdsVal] = splitEachLabel(imds, 1-validationSplit, 'randomized');
    
    fprintf('\n  Training images: %d\n', numel(imdsTrain.Files));
    fprintf('  Validation images: %d\n', numel(imdsVal.Files));
    
    %% Data augmentation
    fprintf('\nSetting up data augmentation...\n');
    
    if usePreprocessing
        % Apply our custom preprocessing + standard augmentation
        augmentedTrainDS = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
            'DataAugmentation', imageDataAugmenter(...
                'RandRotation', [-20, 20], ...
                'RandXReflection', true, ...
                'RandYReflection', false, ...
                'RandXScale', [0.9, 1.1], ...
                'RandYScale', [0.9, 1.1]));
        
        augmentedValDS = augmentedImageDatastore(inputSize(1:2), imdsVal);
        fprintf('  Using data augmentation with custom preprocessing\n');
    else
        augmentedTrainDS = augmentedImageDatastore(inputSize(1:2), imdsTrain);
        augmentedValDS = augmentedImageDatastore(inputSize(1:2), imdsVal);
        fprintf('  Using basic resize only\n');
    end
    
    %% Modify network for transfer learning
    fprintf('\nModifying network for %d classes...\n', numClasses);
    
    % Get layer graph
    lgraph = layerGraph(baseNet);
    
    % Find and replace the fully connected layer
    fcLayerName = 'fc1000';  % ResNet18's final FC layer
    
    % New layers for our classes
    newFCLayer = fullyConnectedLayer(numClasses, ...
        'Name', 'fc_food', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10);
    
    newClassLayer = classificationLayer('Name', 'foodClassOutput');
    
    % Replace layers
    lgraph = replaceLayer(lgraph, fcLayerName, newFCLayer);
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);
    
    fprintf('  Replaced final layers for food classification\n');
    
    %% Training options
    fprintf('\nConfiguring training options...\n');
    
    trainingOptions = trainingOptions('adam', ...
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', miniBatchSize, ...
        'InitialLearnRate', initialLearnRate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 5, ...
        'ValidationData', augmentedValDS, ...
        'ValidationFrequency', 30, ...
        'ValidationPatience', 5, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', 'auto');
    
    fprintf('  Epochs: %d\n', maxEpochs);
    fprintf('  Mini-batch size: %d\n', miniBatchSize);
    fprintf('  Initial learning rate: %.6f\n', initialLearnRate);
    
    %% Train the network
    fprintf('\n');
    fprintf('═══════════════════════════════════════════════════════════════\n');
    fprintf('                    TRAINING CNN                              \n');
    fprintf('═══════════════════════════════════════════════════════════════\n\n');
    
    tic;
    [net, trainInfo] = trainNetwork(augmentedTrainDS, lgraph, trainingOptions);
    trainingTime = toc;
    
    fprintf('\n  Training completed in %.1f seconds\n', trainingTime);
    
    %% Evaluate on validation set
    fprintf('\nEvaluating on validation set...\n');
    
    predictedLabels = classify(net, augmentedValDS);
    actualLabels = imdsVal.Labels;
    
    valAccuracy = mean(predictedLabels == actualLabels);
    fprintf('  Validation Accuracy: %.2f%%\n', valAccuracy * 100);
    
    % Confusion matrix
    confMat = confusionmat(actualLabels, predictedLabels);
    fprintf('\nConfusion Matrix:\n');
    disp(confMat);
    
    %% Save the trained network
    modelsPath = fullfile(fileparts(mfilename('fullpath')), '..', 'models');
    if ~exist(modelsPath, 'dir')
        mkdir(modelsPath);
    end
    
    modelFile = fullfile(modelsPath, 'foodCNN.mat');
    
    cnnModel = struct();
    cnnModel.net = net;
    cnnModel.classNames = categories(imds.Labels);
    cnnModel.inputSize = inputSize;
    cnnModel.trainInfo = trainInfo;
    cnnModel.valAccuracy = valAccuracy;
    cnnModel.confusionMatrix = confMat;
    cnnModel.trainDate = datestr(now);
    cnnModel.trainingTime = trainingTime;
    
    save(modelFile, 'cnnModel', '-v7.3');
    
    fprintf('\nModel saved to: %s\n', modelFile);
    
    %% Summary
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║                 CNN TRAINING COMPLETE                         ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Validation Accuracy: %5.2f%%                                ║\n', valAccuracy*100);
    fprintf('║  Training Time: %.1f seconds                                ║\n', trainingTime);
    fprintf('║  Model: ResNet18 Transfer Learning                          ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
end

%% Helper function
function value = getfield_default(s, field, default)
    if isfield(s, field)
        value = s.(field);
    else
        value = default;
    end
end
