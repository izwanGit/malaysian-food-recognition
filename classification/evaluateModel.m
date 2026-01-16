%% EVALUATE MODEL - Comprehensive Model Evaluation
% Evaluates the trained classifier on test images
%
% Syntax:
%   results = evaluateModel()
%   results = evaluateModel(testPath)
%
% Outputs:
%   results - Struct with accuracy, per-class metrics, confusion matrix

function results = evaluateModel(testPath)
    %% Setup paths
    if nargin < 1
        baseDir = fileparts(mfilename('fullpath'));
        projectRoot = fileparts(baseDir);
        testPath = fullfile(projectRoot, 'dataset', 'test');
    end
    
    modelPath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'models', 'foodClassifier.mat');
    
    if ~exist(modelPath, 'file')
        error('Model not found. Run trainClassifier() first.');
    end
    
    %% Load model
    loaded = load(modelPath, 'model');
    model = loaded.model;
    classNames = model.classNames;
    numClasses = length(classNames);
    
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║           MODEL EVALUATION ON TEST SET                      ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    
    %% Collect test samples
    allFeatures = [];
    allLabels = [];
    
    for c = 1:numClasses
        className = classNames{c};
        classPath = fullfile(testPath, className);
        
        if ~exist(classPath, 'dir')
            fprintf('Warning: No test folder for %s\n', className);
            continue;
        end
        
        imageFiles = [dir(fullfile(classPath, '*.jpg')); ...
                      dir(fullfile(classPath, '*.jpeg')); ...
                      dir(fullfile(classPath, '*.png'))];
        
        fprintf('Testing %s: %d images\n', className, length(imageFiles));
        
        for i = 1:length(imageFiles)
            try
                imagePath = fullfile(classPath, imageFiles(i).name);
                img = imread(imagePath);
                processedImg = preprocessImage(img);
                [features, ~] = extractFeatures(processedImg);
                
                allFeatures = [allFeatures; features]; %#ok<AGROW>
                allLabels = [allLabels; {className}]; %#ok<AGROW>
            catch
                % Skip failed images
            end
        end
    end
    
    if isempty(allFeatures)
        error('No test images found in: %s', testPath);
    end
    
    fprintf('\nTotal test samples: %d\n\n', size(allFeatures, 1));
    
    %% Normalize and predict
    normalizedFeatures = (allFeatures - model.featureMean) ./ model.featureStd;
    predictions = predict(model.classifier, normalizedFeatures);
    
    %% Calculate metrics
    accuracy = sum(strcmp(predictions, allLabels)) / length(allLabels);
    
    % Confusion matrix
    labelToNum = containers.Map(classNames, 1:numClasses);
    actualNumeric = cellfun(@(x) labelToNum(x), allLabels);
    predictedNumeric = cellfun(@(x) labelToNum(x), predictions);
    confMat = confusionmat(actualNumeric, predictedNumeric);
    
    % Per-class metrics
    precisions = zeros(numClasses, 1);
    recalls = zeros(numClasses, 1);
    f1Scores = zeros(numClasses, 1);
    
    for c = 1:numClasses
        tp = confMat(c, c);
        fp = sum(confMat(:, c)) - tp;
        fn = sum(confMat(c, :)) - tp;
        
        precisions(c) = tp / (tp + fp + eps);
        recalls(c) = tp / (tp + fn + eps);
        f1Scores(c) = 2 * precisions(c) * recalls(c) / (precisions(c) + recalls(c) + eps);
    end
    
    %% Display results
    fprintf('─────────────────────────────────────────────────\n');
    fprintf('%-15s %10s %10s %10s\n', 'Class', 'Precision', 'Recall', 'F1-Score');
    fprintf('─────────────────────────────────────────────────\n');
    
    for c = 1:numClasses
        fprintf('%-15s %9.2f%% %9.2f%% %9.2f%%\n', ...
                classNames{c}, precisions(c)*100, recalls(c)*100, f1Scores(c)*100);
    end
    
    fprintf('─────────────────────────────────────────────────\n');
    fprintf('%-15s %9.2f%% %9.2f%% %9.2f%%\n', ...
            'AVERAGE', mean(precisions)*100, mean(recalls)*100, mean(f1Scores)*100);
    fprintf('\n');
    
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║  TEST ACCURACY: %5.2f%%                                     ║\n', accuracy*100);
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    
    %% Return results
    results.accuracy = accuracy;
    results.confusionMatrix = confMat;
    results.precision = precisions;
    results.recall = recalls;
    results.f1Score = f1Scores;
    results.predictions = predictions;
    results.actualLabels = allLabels;
end
