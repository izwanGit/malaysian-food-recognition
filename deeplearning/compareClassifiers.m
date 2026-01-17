%% COMPARE CLASSIFIERS - Compare SVM vs CNN Performance
% Compares traditional SVM classifier with Deep Learning CNN
%
% This script demonstrates how image preprocessing improves both approaches
%
% Syntax:
%   results = compareClassifiers()
%   results = compareClassifiers(testPath)

function results = compareClassifiers(testPath)
    %% Setup
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(projectRoot));
    
    if nargin < 1
        testPath = fullfile(projectRoot, 'dataset', 'test');
    end
    
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║        CLASSIFIER COMPARISON: SVM vs CNN (Deep Learning)     ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
    
    %% Check models exist
    svmModelPath = fullfile(projectRoot, 'models', 'foodClassifier.mat');
    cnnModelPath = fullfile(projectRoot, 'models', 'foodCNN.mat');
    
    hasSVM = exist(svmModelPath, 'file');
    hasCNN = exist(cnnModelPath, 'file');
    
    if ~hasSVM && ~hasCNN
        error('No models found. Train at least one classifier first.');
    end
    
    %% Get test images
    classNames = {'nasi_lemak', 'roti_canai', 'satay', 'laksa', ...
                  'popiah', 'kaya_toast', 'mixed_rice'};
    
    testImages = {};
    testLabels = {};
    
    for i = 1:length(classNames)
        classPath = fullfile(testPath, classNames{i});
        if ~exist(classPath, 'dir')
            continue;
        end
        
        images = dir(fullfile(classPath, '*.jpg'));
        for j = 1:min(20, length(images))  % Max 20 per class
            testImages{end+1} = fullfile(classPath, images(j).name);
            testLabels{end+1} = classNames{i};
        end
    end
    
    numTest = length(testImages);
    fprintf('Found %d test images\n\n', numTest);
    
    if numTest == 0
        error('No test images found in: %s', testPath);
    end
    
    %% Evaluate SVM
    if hasSVM
        fprintf('--- Evaluating SVM Classifier ---\n');
        
        svmPredictions = cell(numTest, 1);
        svmConfidences = zeros(numTest, 1);
        svmTimes = zeros(numTest, 1);
        
        for i = 1:numTest
            tic;
            img = imread(testImages{i});
            processedImg = preprocessImage(img);
            [pred, conf] = classifyFood(processedImg);
            svmTimes(i) = toc;
            
            svmPredictions{i} = pred;
            svmConfidences(i) = conf;
            
            if mod(i, 20) == 0
                fprintf('  Processed %d/%d images\n', i, numTest);
            end
        end
        
        svmCorrect = sum(strcmp(svmPredictions, testLabels'));
        svmAccuracy = svmCorrect / numTest;
        svmAvgTime = mean(svmTimes);
        
        fprintf('  SVM Accuracy: %.2f%% (%d/%d)\n', svmAccuracy*100, svmCorrect, numTest);
        fprintf('  Average inference time: %.3f seconds\n\n', svmAvgTime);
    else
        svmAccuracy = 0;
        svmAvgTime = 0;
        fprintf('SVM model not found, skipping...\n\n');
    end
    
    %% Evaluate CNN
    if hasCNN
        fprintf('--- Evaluating CNN Classifier (Deep Learning) ---\n');
        
        cnnPredictions = cell(numTest, 1);
        cnnConfidences = zeros(numTest, 1);
        cnnTimes = zeros(numTest, 1);
        
        for i = 1:numTest
            tic;
            [pred, conf] = classifyFoodCNN(testImages{i});
            cnnTimes(i) = toc;
            
            cnnPredictions{i} = pred;
            cnnConfidences(i) = conf;
            
            if mod(i, 20) == 0
                fprintf('  Processed %d/%d images\n', i, numTest);
            end
        end
        
        cnnCorrect = sum(strcmp(cnnPredictions, testLabels'));
        cnnAccuracy = cnnCorrect / numTest;
        cnnAvgTime = mean(cnnTimes);
        
        fprintf('  CNN Accuracy: %.2f%% (%d/%d)\n', cnnAccuracy*100, cnnCorrect, numTest);
        fprintf('  Average inference time: %.3f seconds\n\n', cnnAvgTime);
    else
        cnnAccuracy = 0;
        cnnAvgTime = 0;
        fprintf('CNN model not found, skipping...\n\n');
    end
    
    %% Comparison Summary
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║                    COMPARISON RESULTS                         ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Metric              │    SVM        │    CNN (DL)           ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    
    if hasSVM && hasCNN
        fprintf('║  Accuracy            │  %5.2f%%       │  %5.2f%%               ║\n', svmAccuracy*100, cnnAccuracy*100);
        fprintf('║  Inference Time      │  %.3fs       │  %.3fs               ║\n', svmAvgTime, cnnAvgTime);
        fprintf('║  Features            │  127 manual   │  Auto-learned         ║\n');
        fprintf('║  Training Data Needed│  Medium       │  Large                ║\n');
    elseif hasSVM
        fprintf('║  SVM Accuracy: %5.2f%%                                      ║\n', svmAccuracy*100);
    else
        fprintf('║  CNN Accuracy: %5.2f%%                                      ║\n', cnnAccuracy*100);
    end
    
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
    
    %% Return results
    results = struct();
    results.numTestImages = numTest;
    
    if hasSVM
        results.svmAccuracy = svmAccuracy;
        results.svmAvgTime = svmAvgTime;
    end
    
    if hasCNN
        results.cnnAccuracy = cnnAccuracy;
        results.cnnAvgTime = cnnAvgTime;
    end
end
