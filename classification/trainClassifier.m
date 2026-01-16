%% TRAIN CLASSIFIER - Train SVM Food Classifier
% Trains a multi-class SVM classifier on Malaysian hawker food images
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
%           .classifier   - fitcecoc SVM model
%           .classNames   - Cell array of class names
%           .featureNames - Cell array of feature names
%           .trainStats   - Training statistics

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
    
    fprintf('=== Training Food Classifier ===\n\n');
    fprintf('Dataset path: %s\n', datasetPath);
    fprintf('Max images per class: %d\n', maxImagesPerClass);
    fprintf('Number of classes: %d\n\n', numClasses);
    
    %% Collect features from all classes
    allFeatures = [];
    allLabels = [];
    featureNames = [];
    
    for c = 1:numClasses
        className = classNames{c};
        classPath = fullfile(datasetPath, className);
        
        fprintf('Processing class %d/%d: %s\n', c, numClasses, className);
        
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
        imageFiles = imageFiles(randperm(length(imageFiles), numImages));
        
        fprintf('  Loading %d images...\n', numImages);
        
        classFeatures = zeros(numImages, 124);  % Preallocate
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
                fprintf('  Warning: Failed to process %s - %s\n', ...
                        imageFiles(i).name, ME.message);
            end
            
            % Progress indicator
            if mod(i, 50) == 0
                fprintf('  Processed %d/%d images\n', i, numImages);
            end
        end
        
        % Trim to valid features only
        classFeatures = classFeatures(1:validCount, :);
        classLabels = repmat({className}, validCount, 1);
        
        % Append to all features
        allFeatures = [allFeatures; classFeatures]; %#ok<AGROW>
        allLabels = [allLabels; classLabels]; %#ok<AGROW>
        
        fprintf('  Completed: %d valid images\n\n', validCount);
    end
    
    fprintf('Total training samples: %d\n', size(allFeatures, 1));
    fprintf('Feature dimensions: %d\n\n', size(allFeatures, 2));
    
    %% Normalize features
    fprintf('Normalizing features...\n');
    featureMean = mean(allFeatures, 1);
    featureStd = std(allFeatures, 0, 1);
    featureStd(featureStd == 0) = 1;  % Avoid division by zero
    normalizedFeatures = (allFeatures - featureMean) ./ featureStd;
    
    %% Train SVM classifier with One-vs-All strategy
    fprintf('Training SVM classifier (this may take a few minutes)...\n');
    
    % Create SVM template with RBF kernel
    svmTemplate = templateSVM('KernelFunction', 'rbf', ...
                              'KernelScale', 'auto', ...
                              'BoxConstraint', 1, ...
                              'Standardize', false);  % Already normalized
    
    % Train multi-class classifier using ECOC (Error-Correcting Output Codes)
    classifier = fitcecoc(normalizedFeatures, allLabels, ...
                          'Learners', svmTemplate, ...
                          'Coding', 'onevsall', ...
                          'ClassNames', classNames);
    
    %% Evaluate on training set (for sanity check)
    fprintf('Evaluating training accuracy...\n');
    predictions = predict(classifier, normalizedFeatures);
    trainAccuracy = sum(strcmp(predictions, allLabels)) / length(allLabels);
    fprintf('Training accuracy: %.2f%%\n\n', trainAccuracy * 100);
    
    %% Save model
    model.classifier = classifier;
    model.classNames = classNames;
    model.featureNames = featureNames;
    model.featureMean = featureMean;
    model.featureStd = featureStd;
    model.trainStats.numSamples = size(allFeatures, 1);
    model.trainStats.numFeatures = size(allFeatures, 2);
    model.trainStats.trainAccuracy = trainAccuracy;
    model.trainStats.trainDate = datestr(now);
    
    % Save to file
    modelsPath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'models');
    if ~exist(modelsPath, 'dir')
        mkdir(modelsPath);
    end
    modelFile = fullfile(modelsPath, 'foodClassifier.mat');
    save(modelFile, 'model');
    
    fprintf('Model saved to: %s\n', modelFile);
    fprintf('=== Training Complete ===\n');
end
