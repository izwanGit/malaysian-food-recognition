%% EVALUATE SVM ON TEST SET
% Evaluates the trained SVM model on the test dataset for fair comparison with CNN

function evaluateSVMOnTest()
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║      SVM MODEL EVALUATION ON TEST SET                      ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    
    %% Load trained model
    baseDir = fileparts(mfilename('fullpath'));
    modelPath = fullfile(baseDir, 'models', 'foodClassifier.mat');
    
    if ~exist(modelPath, 'file')
        error('Model not found: %s\nRun trainClassifier() first.', modelPath);
    end
    
    fprintf('Loading trained SVM model...\n');
    loaded = load(modelPath, 'model');
    model = loaded.model;
    fprintf('Model loaded successfully!\n\n');
    
    %% Define test path
    testPath = fullfile(baseDir, 'dataset', 'test');
    classNames = model.classNames;
    numClasses = length(classNames);
    
    fprintf('Test dataset: %s\n', testPath);
    fprintf('Classes: %d\n\n', numClasses);
    
    %% Extract features from test set
    fprintf('─── Extracting Features from Test Set ───\n\n');
    
    allFeatures = [];
    allLabels = {};
    
    for c = 1:numClasses
        className = classNames{c};
        classPath = fullfile(testPath, className);
        
        if ~exist(classPath, 'dir')
            fprintf('[!] Skipping missing class: %s\n', className);
            continue;
        end
        
        imageFiles = [dir(fullfile(classPath, '*.jpg')); ...
                      dir(fullfile(classPath, '*.jpeg')); ...
                      dir(fullfile(classPath, '*.png'))];
        
        fprintf('[%d/%d] %s: %d images... ', c, numClasses, className, length(imageFiles));
        
        classFeatures = [];
        validCount = 0;
        
        for i = 1:length(imageFiles)
            try
                imgPath = fullfile(classPath, imageFiles(i).name);
                img = imread(imgPath);
                
                % Preprocess
                processedImg = preprocessImage(img);
                
                % Extract features
                features = extractFeatures(processedImg);
                
                if ~isempty(features) && all(isfinite(features))
                    classFeatures = [classFeatures; features];
                    validCount = validCount + 1;
                end
            catch
                % Skip problematic images
            end
        end
        
        fprintf('%d valid\n', validCount);
        
        % Add to dataset
        allFeatures = [allFeatures; classFeatures];
        allLabels = [allLabels; repmat({className}, validCount, 1)];
    end
    
    fprintf('\nTotal test samples: %d\n\n', size(allFeatures, 1));
    
    %% Normalize features using training stats
    % Ensure feature dimensions match (truncate if needed)
    expectedDim = length(model.featureMean);
    if size(allFeatures, 2) > expectedDim
        fprintf('Truncating features from %d to %d dimensions...\n', size(allFeatures, 2), expectedDim);
        allFeatures = allFeatures(:, 1:expectedDim);
    elseif size(allFeatures, 2) < expectedDim
        error('Feature dimension mismatch: got %d, expected %d', size(allFeatures, 2), expectedDim);
    end
    
    normalizedFeatures = (allFeatures - model.featureMean) ./ model.featureStd;
    
    %% Predict
    fprintf('─── Running Predictions ───\n\n');
    predictions = predict(model.classifier, normalizedFeatures);
    
    %% Calculate accuracy
    correct = sum(strcmp(predictions, allLabels));
    total = length(allLabels);
    accuracy = correct / total * 100;
    
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║                    TEST RESULTS                            ║\n');
    fprintf('╠════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Total Test Samples:  %5d                                ║\n', total);
    fprintf('║  Correct Predictions: %5d                                ║\n', correct);
    fprintf('║  TEST ACCURACY:       %5.2f%%                              ║\n', accuracy);
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    
    %% Confusion Matrix
    fprintf('─── Confusion Matrix ───\n\n');
    
    labelToNum = containers.Map(classNames, 1:numClasses);
    actualNumeric = cellfun(@(x) labelToNum(x), allLabels);
    predictedNumeric = cellfun(@(x) labelToNum(x), predictions);
    
    confMat = confusionmat(actualNumeric, predictedNumeric);
    
    % Per-class accuracy
    fprintf('Per-Class Performance:\n');
    fprintf('%-15s %10s %10s\n', 'Class', 'Correct', 'Accuracy');
    fprintf('─────────────────────────────────────\n');
    
    for c = 1:numClasses
        classTotal = sum(confMat(c, :));
        classCorrect = confMat(c, c);
        classAcc = classCorrect / classTotal * 100;
        fprintf('%-15s %10d %9.1f%%\n', classNames{c}, classCorrect, classAcc);
    end
    fprintf('─────────────────────────────────────\n');
    fprintf('%-15s %10d %9.1f%%\n', 'OVERALL', correct, accuracy);
    fprintf('\n');
    
    %% Comparison with CNN
    fprintf('═══════════════════════════════════════════════════════════════\n');
    fprintf('  MODEL COMPARISON (Test Set)\n');
    fprintf('═══════════════════════════════════════════════════════════════\n');
    fprintf('  CNN (SqueezeNet):     83.00%%\n');
    fprintf('  SVM (Classical):      %.2f%%\n', accuracy);
    fprintf('  Difference:           %.2f%%\n', 83.00 - accuracy);
    fprintf('═══════════════════════════════════════════════════════════════\n\n');
end
