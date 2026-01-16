%% TRAIN CLASSIFIER - Train SVM Food Classifier with Cross-Validation
% Trains a multi-class SVM classifier on Malaysian hawker food images
% with k-fold cross-validation and confusion matrix for evaluation
%
% Syntax:
%   model = trainClassifier()
%   model = trainClassifier(datasetPath)
%   model = trainClassifier(datasetPath, maxImagesPerClass)
%
% Inputs:
%   datasetPath       - Path to training dataset (default: 'dataset/train')
%   maxImagesPerClass - Maximum images per class for training (default: 200)
%
% Outputs:
%   model - Trained classifier structure with fields:
%           .classifier     - fitcecoc SVM model
%           .classNames     - Cell array of class names
%           .featureNames   - Cell array of feature names
%           .featureMean    - Mean for normalization
%           .featureStd     - Std for normalization
%           .trainStats     - Training statistics including CV accuracy
%           .confusionMat   - Confusion matrix from cross-validation

function model = trainClassifier(datasetPath, maxImagesPerClass)
    %% Default parameters
    if nargin < 1
        baseDir = fileparts(mfilename('fullpath'));
        projectRoot = fileparts(baseDir);
        datasetPath = fullfile(projectRoot, 'dataset', 'train');
    end
    if nargin < 2
        maxImagesPerClass = 200;  % Limit for faster training
    end
    
    %% Define food classes
    classNames = {'nasi_lemak', 'roti_canai', 'satay', 'laksa', ...
                  'popiah', 'kaya_toast', 'mixed_rice'};
    numClasses = length(classNames);
    
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║     MALAYSIAN HAWKER FOOD CLASSIFIER - TRAINING            ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    fprintf('Dataset path: %s\n', datasetPath);
    fprintf('Max images per class: %d\n', maxImagesPerClass);
    fprintf('Number of classes: %d\n', numClasses);
    fprintf('Classes: %s\n\n', strjoin(classNames, ', '));
    
    %% Collect features from all classes
    fprintf('─── PHASE 1: Feature Extraction ───\n\n');
    
    allFeatures = [];
    allLabels = [];
    featureNames = [];
    samplesPerClass = zeros(1, numClasses);
    
    for c = 1:numClasses
        className = classNames{c};
        classPath = fullfile(datasetPath, className);
        
        fprintf('[%d/%d] Processing: %s\n', c, numClasses, className);
        
        % Get image files
        imageFiles = [dir(fullfile(classPath, '*.jpg')); ...
                      dir(fullfile(classPath, '*.jpeg')); ...
                      dir(fullfile(classPath, '*.png'))];
        
        if isempty(imageFiles)
            warning('No images found for class: %s', className);
            continue;
        end
        
        % Limit number of images
        numImages = min(length(imageFiles), maxImagesPerClass);
        rng(42);  % Reproducible random selection
        imageFiles = imageFiles(randperm(length(imageFiles), numImages));
        
        fprintf('      Loading %d images...\n', numImages);
        
        classFeatures = zeros(numImages, 127);  % Preallocate (108 color + 19 texture)
        validCount = 0;
        
        for i = 1:numImages
            try
                imagePath = fullfile(classPath, imageFiles(i).name);
                
                % Load and preprocess image
                img = imread(imagePath);
                processedImg = preprocessImage(img);
                
                % Extract features
                [features, names] = extractFeatures(processedImg);
                
                validCount = validCount + 1;
                classFeatures(validCount, :) = features;
                
                if isempty(featureNames)
                    featureNames = names;
                end
                
            catch ME
                % Silent fail for individual images
            end
            
            % Progress indicator
            if mod(i, 100) == 0
                fprintf('      Progress: %d/%d\n', i, numImages);
            end
        end
        
        % Trim to valid features only
        classFeatures = classFeatures(1:validCount, :);
        classLabels = repmat({className}, validCount, 1);
        samplesPerClass(c) = validCount;
        
        % Append to all features
        allFeatures = [allFeatures; classFeatures]; %#ok<AGROW>
        allLabels = [allLabels; classLabels]; %#ok<AGROW>
        
        fprintf('      Completed: %d valid images\n\n', validCount);
    end
    
    fprintf('Total training samples: %d\n', size(allFeatures, 1));
    fprintf('Feature dimensions: %d\n\n', size(allFeatures, 2));
    
    % Display samples per class
    fprintf('Samples per class:\n');
    for c = 1:numClasses
        fprintf('  %-12s: %d samples\n', classNames{c}, samplesPerClass(c));
    end
    fprintf('\n');
    
    %% Normalize features
    fprintf('─── PHASE 2: Feature Normalization ───\n\n');
    fprintf('Applying Z-score normalization...\n');
    
    featureMean = mean(allFeatures, 1);
    featureStd = std(allFeatures, 0, 1);
    featureStd(featureStd == 0) = 1;  % Avoid division by zero
    normalizedFeatures = (allFeatures - featureMean) ./ featureStd;
    
    fprintf('Done.\n\n');
    
    %% Cross-Validation
    fprintf('─── PHASE 3: Cross-Validation (5-Fold) ───\n\n');
    
    numFolds = 5;
    cv = cvpartition(allLabels, 'KFold', numFolds);
    
    cvAccuracies = zeros(numFolds, 1);
    allPredictions = cell(size(allLabels));
    
    for fold = 1:numFolds
        fprintf('Fold %d/%d: ', fold, numFolds);
        
        % Split data
        trainIdx = cv.training(fold);
        testIdx = cv.test(fold);
        
        XTrain = normalizedFeatures(trainIdx, :);
        yTrain = allLabels(trainIdx);
        XTest = normalizedFeatures(testIdx, :);
        yTest = allLabels(testIdx);
        
        % Train SVM
        svmTemplate = templateSVM('KernelFunction', 'rbf', ...
                                  'KernelScale', 'auto', ...
                                  'BoxConstraint', 1, ...
                                  'Standardize', false);
        
        cvClassifier = fitcecoc(XTrain, yTrain, ...
                                'Learners', svmTemplate, ...
                                'Coding', 'onevsall', ...
                                'ClassNames', classNames);
        
        % Predict
        predictions = predict(cvClassifier, XTest);
        allPredictions(testIdx) = predictions;
        
        % Calculate accuracy
        cvAccuracies(fold) = sum(strcmp(predictions, yTest)) / length(yTest);
        fprintf('Accuracy = %.2f%%\n', cvAccuracies(fold) * 100);
    end
    
    meanCVAccuracy = mean(cvAccuracies);
    stdCVAccuracy = std(cvAccuracies);
    fprintf('\nCross-Validation Results:\n');
    fprintf('  Mean Accuracy: %.2f%% ± %.2f%%\n\n', meanCVAccuracy * 100, stdCVAccuracy * 100);
    
    %% Confusion Matrix
    fprintf('─── PHASE 4: Confusion Matrix ───\n\n');
    
    % Convert to numeric for confusion matrix
    labelToNum = containers.Map(classNames, 1:numClasses);
    actualNumeric = cellfun(@(x) labelToNum(x), allLabels);
    predictedNumeric = cellfun(@(x) labelToNum(x), allPredictions);
    
    confMat = confusionmat(actualNumeric, predictedNumeric);
    
    % Display confusion matrix
    fprintf('Confusion Matrix:\n');
    fprintf('                 ');
    for c = 1:numClasses
        fprintf('%8s ', classNames{c}(1:min(8,end)));
    end
    fprintf('\n');
    fprintf('─────────────────');
    fprintf('─────────');
    fprintf(repmat('─────────', 1, numClasses-1));
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
    
    % Per-class metrics
    fprintf('Per-Class Performance:\n');
    fprintf('─────────────────────────────────────────────────\n');
    fprintf('%-15s %10s %10s %10s\n', 'Class', 'Precision', 'Recall', 'F1-Score');
    fprintf('─────────────────────────────────────────────────\n');
    
    precisions = zeros(numClasses, 1);
    recalls = zeros(numClasses, 1);
    f1Scores = zeros(numClasses, 1);
    
    for c = 1:numClasses
        tp = confMat(c, c);
        fp = sum(confMat(:, c)) - tp;
        fn = sum(confMat(c, :)) - tp;
        
        precision = tp / (tp + fp + eps);
        recall = tp / (tp + fn + eps);
        f1 = 2 * precision * recall / (precision + recall + eps);
        
        precisions(c) = precision;
        recalls(c) = recall;
        f1Scores(c) = f1;
        
        fprintf('%-15s %9.2f%% %9.2f%% %9.2f%%\n', ...
                classNames{c}, precision*100, recall*100, f1*100);
    end
    fprintf('─────────────────────────────────────────────────\n');
    fprintf('%-15s %9.2f%% %9.2f%% %9.2f%%\n', ...
            'AVERAGE', mean(precisions)*100, mean(recalls)*100, mean(f1Scores)*100);
    fprintf('\n');
    
    %% Train Final Model on All Data
    fprintf('─── PHASE 5: Training Final Model ───\n\n');
    fprintf('Training on all %d samples...\n', size(normalizedFeatures, 1));
    
    svmTemplate = templateSVM('KernelFunction', 'rbf', ...
                              'KernelScale', 'auto', ...
                              'BoxConstraint', 1, ...
                              'Standardize', false);
    
    classifier = fitcecoc(normalizedFeatures, allLabels, ...
                          'Learners', svmTemplate, ...
                          'Coding', 'onevsall', ...
                          'ClassNames', classNames);
    
    % Training accuracy (sanity check)
    trainPredictions = predict(classifier, normalizedFeatures);
    trainAccuracy = sum(strcmp(trainPredictions, allLabels)) / length(allLabels);
    fprintf('Training accuracy: %.2f%%\n\n', trainAccuracy * 100);
    
    %% Save Model
    fprintf('─── PHASE 6: Saving Model ───\n\n');
    
    model.classifier = classifier;
    model.classNames = classNames;
    model.featureNames = featureNames;
    model.featureMean = featureMean;
    model.featureStd = featureStd;
    model.trainStats.numSamples = size(allFeatures, 1);
    model.trainStats.numFeatures = size(allFeatures, 2);
    model.trainStats.samplesPerClass = samplesPerClass;
    model.trainStats.trainAccuracy = trainAccuracy;
    model.trainStats.cvAccuracy = meanCVAccuracy;
    model.trainStats.cvStd = stdCVAccuracy;
    model.trainStats.trainDate = datestr(now);
    model.confusionMatrix = confMat;
    model.perClassMetrics.precision = precisions;
    model.perClassMetrics.recall = recalls;
    model.perClassMetrics.f1Score = f1Scores;
    
    % Save to file
    modelsPath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'models');
    if ~exist(modelsPath, 'dir')
        mkdir(modelsPath);
    end
    modelFile = fullfile(modelsPath, 'foodClassifier.mat');
    save(modelFile, 'model');
    
    fprintf('Model saved to: %s\n\n', modelFile);
    
    %% Summary
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║                    TRAINING COMPLETE                        ║\n');
    fprintf('╠════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Cross-Validation Accuracy: %5.2f%% ± %.2f%%                ║\n', meanCVAccuracy*100, stdCVAccuracy*100);
    fprintf('║  Training Accuracy:         %5.2f%%                         ║\n', trainAccuracy*100);
    fprintf('║  Average F1-Score:          %5.2f%%                         ║\n', mean(f1Scores)*100);
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
end
