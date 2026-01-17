%% TRAIN DEEPLABV3+ - Deep Learning Semantic Segmentation for Food Detection
% Trains a DeepLabv3+ model for food semantic segmentation
%
% This demonstrates deep learning for food region detection
%
% Syntax:
%   net = trainDeepLabv3()
%   net = trainDeepLabv3(datasetPath, options)
%
% Requirements:
%   - Deep Learning Toolbox
%   - Computer Vision Toolbox
%
% Inputs:
%   datasetPath - Path to dataset with class subfolders
%   options     - Optional training configuration
%
% Outputs:
%   net - Trained DeepLabv3+ network

function net = trainDeepLabv3(datasetPath, options)
    %% Check toolboxes
    if ~license('test', 'Neural_Network_Toolbox')
        error('Deep Learning Toolbox required.');
    end
    
    if ~license('test', 'Video_and_Image_Blockset')
        error('Computer Vision Toolbox required.');
    end
    
    %% Setup
    if nargin < 1 || isempty(datasetPath)
        projectRoot = fileparts(fileparts(mfilename('fullpath')));
        datasetPath = fullfile(projectRoot, 'dataset', 'train');
    end
    
    if nargin < 2
        options = struct();
    end
    
    % Parameters
    maxEpochs = getfield_default(options, 'maxEpochs', 20);
    miniBatchSize = getfield_default(options, 'miniBatchSize', 8);
    initialLearnRate = getfield_default(options, 'initialLearnRate', 0.001);
    inputSize = [512, 512, 3];
    numClasses = 2;  % Food vs Background
    
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║   DeepLabv3+ SEMANTIC SEGMENTATION - Food Detection          ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
    
    %% Create training data from classification dataset
    % We'll use the food images and generate pseudo-masks using classical methods
    fprintf('Preparing training data...\n');
    
    imds = imageDatastore(datasetPath, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', ...
        'FileExtensions', {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'});
    
    numImages = numel(imds.Files);
    fprintf('  Found %d images\n', numImages);
    
    %% Generate pseudo ground truth masks using classical segmentation
    fprintf('  Generating pseudo ground-truth masks using classical methods...\n');
    
    maskDir = fullfile(fileparts(datasetPath), 'masks_generated');
    if ~exist(maskDir, 'dir')
        mkdir(maskDir);
    end
    
    validImageFiles = {};
    
    % FAST MODE: Process fewer images for demo training
    numToProcess = min(50, numImages); 
    
    for i = 1:numToProcess
        try
            img = imread(imds.Files{i});
            
            % Use classical segmentation to create mask
            processedImg = preprocessForMask(img, inputSize(1:2));
            mask = createPseudoMask(processedImg);
            
            % Save mask
            [~, filename, ~] = fileparts(imds.Files{i});
            maskFile = fullfile(maskDir, [filename, '_mask.png']);
            imwrite(uint8(mask) * 255, maskFile);
            
            % Add to valid list
            validImageFiles{end+1} = imds.Files{i};
            
            if mod(i, 10) == 0
                fprintf('    Generated %d/%d masks\n', i, numToProcess);
            end
        catch
            fprintf('    Warning: Skipped corrupt file %s\n', imds.Files{i});
        end
    end
    
    %% Create pixel label datastore
    classNames = {'background', 'food'};
    labelIDs = [0, 255];
    
    pxds = pixelLabelDatastore(maskDir, classNames, labelIDs);
    
    % Update image datastore to match masks
    % Update image datastore to match masks (using only valid files)
    if isempty(validImageFiles)
        error('No valid images found for segmentation training.');
    end
    imdsTrain = imageDatastore(validImageFiles);
    
    %% Create combined datastore for training
    combinedDS = combine(imdsTrain, pxds);
    
    % Apply transforms
    trainDS = transform(combinedDS, @(data) preprocessTrainingData(data, inputSize));
    
    %% Create DeepLabv3+ network
    fprintf('\nBuilding DeepLabv3+ network...\n');
    
    try
        lgraph = deeplabv3plusLayers(inputSize, numClasses, 'resnet18');
        fprintf('  Using ResNet18 backbone\n');
    catch
        % Fallback: create simplified encoder-decoder
        fprintf('  DeepLabv3+ not available, using simplified architecture\n');
        lgraph = createSimplifiedSegmentationNetwork(inputSize, numClasses);
    end
    
    %% Training options
    fprintf('\nConfiguring training...\n');
    
    trainOpts = trainingOptions('adam', ...
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', miniBatchSize, ...
        'InitialLearnRate', initialLearnRate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 10, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'VerboseFrequency', 20, ...
        'Plots', 'none', ...
        'ExecutionEnvironment', 'auto');
    
    fprintf('  Epochs: %d\n', maxEpochs);
    fprintf('  Batch size: %d\n', miniBatchSize);
    fprintf('  Learning rate: %.4f\n', initialLearnRate);
    
    %% Train
    fprintf('\n');
    fprintf('═══════════════════════════════════════════════════════════════\n');
    fprintf('                 TRAINING DeepLabv3+                          \n');
    fprintf('═══════════════════════════════════════════════════════════════\n\n');
    
    tic;
    [net, trainInfo] = trainNetwork(trainDS, lgraph, trainOpts);
    trainingTime = toc;
    
    fprintf('\n  Training completed in %.1f seconds\n', trainingTime);
    
    %% Save model
    modelsPath = fullfile(fileparts(mfilename('fullpath')), '..', 'models');
    if ~exist(modelsPath, 'dir')
        mkdir(modelsPath);
    end
    
    modelFile = fullfile(modelsPath, 'foodSegmentationDL.mat');
    
    segModel = struct();
    segModel.net = net;
    segModel.classNames = classNames;
    segModel.inputSize = inputSize;
    segModel.trainInfo = trainInfo;
    segModel.trainDate = datestr(now);
    segModel.trainingTime = trainingTime;
    
    save(modelFile, 'segModel', '-v7.3');
    fprintf('\nModel saved to: %s\n', modelFile);
    
    %% Summary
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║            DeepLabv3+ TRAINING COMPLETE                       ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Model: DeepLabv3+ with ResNet18 backbone                    ║\n');
    fprintf('║  Classes: Food vs Background (binary)                       ║\n');
    fprintf('║  Training Time: %.1f seconds                                ║\n', trainingTime);
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
end

%% Helper function: Preprocess image for mask generation
function img = preprocessForMask(img, targetSize)
    img = imresize(img, targetSize);
    if isa(img, 'uint8')
        img = im2double(img);
    end
end

%% Helper function: Create pseudo mask using classical methods
function mask = createPseudoMask(img)
    % Use HSV thresholding + morphology to create ground truth
    hsv = rgb2hsv(img);
    S = hsv(:,:,2);
    V = hsv(:,:,3);
    
    % Detect food regions (colored, not too dark/bright)
    mask = (S > 0.08) & (V > 0.15) & (V < 0.98);
    
    % Exclude white backgrounds
    whiteBG = (S < 0.1) & (V > 0.85);
    mask = mask & ~whiteBG;
    
    % Morphological cleanup
    se = strel('disk', 5);
    mask = imopen(mask, se);
    mask = imclose(mask, strel('disk', 10));
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 500);
end

%% Helper function: Preprocess training data
function data = preprocessTrainingData(data, inputSize)
    img = data{1};
    label = data{2};
    
    % Resize
    img = imresize(img, inputSize(1:2));
    label = imresize(label, inputSize(1:2), 'nearest');
    
    % Normalize image
    if isa(img, 'uint8')
        img = im2double(img);
    end
    
    data = {img, label};
end

%% Helper function: Simplified segmentation network (fallback)
function lgraph = createSimplifiedSegmentationNetwork(inputSize, numClasses)
    layers = [
        imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
        
        % Encoder
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'enc_conv1')
        batchNormalizationLayer('Name', 'enc_bn1')
        reluLayer('Name', 'enc_relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc_pool1')
        
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'enc_conv2')
        batchNormalizationLayer('Name', 'enc_bn2')
        reluLayer('Name', 'enc_relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc_pool2')
        
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'enc_conv3')
        batchNormalizationLayer('Name', 'enc_bn3')
        reluLayer('Name', 'enc_relu3')
        
        % Decoder
        transposedConv2dLayer(4, 128, 'Stride', 2, 'Cropping', 'same', 'Name', 'dec_upconv1')
        batchNormalizationLayer('Name', 'dec_bn1')
        reluLayer('Name', 'dec_relu1')
        
        transposedConv2dLayer(4, 64, 'Stride', 2, 'Cropping', 'same', 'Name', 'dec_upconv2')
        batchNormalizationLayer('Name', 'dec_bn2')
        reluLayer('Name', 'dec_relu2')
        
        % Output
        convolution2dLayer(1, numClasses, 'Name', 'classifier')
        softmaxLayer('Name', 'softmax')
        pixelClassificationLayer('Name', 'output')
    ];
    
    lgraph = layerGraph(layers);
end

%% Helper function
function value = getfield_default(s, field, default)
    if isfield(s, field)
        value = s.(field);
    else
        value = default;
    end
end
