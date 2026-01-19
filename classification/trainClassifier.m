function model = trainClassifier(datasetPath, maxImagesPerClass, useDeepFeatures)
    %% Default parameters
    if nargin < 1
        baseDir = fileparts(mfilename('fullpath'));
        projectRoot = fileparts(baseDir);
        datasetPath = fullfile(projectRoot, 'dataset', 'train');
    end
    if nargin < 2
        maxImagesPerClass = 1000;  % Process up to 1000 images per class
    end
    if nargin < 3
        useDeepFeatures = true;  % Use SqueezeNet features by default for A++ quality
    end
    
    %% Define food classes
    classNames = {'kaya_toast', 'laksa', 'mixed_rice', 'nasi_lemak', ...
                  'popiah', 'roti_canai', 'satay'};
    numClasses = length(classNames);
    
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║     MALAYSIAN HAWKER FOOD CLASSIFIER - A++ OPTIMIZED       ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
    fprintf('Dataset path: %s\n', datasetPath);
    if useDeepFeatures
        featureTypeLabel = 'Deep Features (SqueezeNet)';
    else
        featureTypeLabel = 'Hand-Crafted Features (Color+Texture+HOG)';
    end
    fprintf('Feature Type: %s\n', featureTypeLabel);
    
    %% Load Deep Model if needed
    if useDeepFeatures
        fprintf('Loading SqueezeNet for feature extraction...\n');
        net = squeezenet;
        inputSize = net.Layers(1).InputSize;
    end
    
    %% Collect features from all classes
    fprintf('\n─── PHASE 1: Feature Extraction ───\n\n');
    
    allFeatures = [];
    allLabels = [];
    featureNames = [];
    samplesPerClass = zeros(1, numClasses);
    
    for c = 1:numClasses
        className = classNames{c};
        classPath = fullfile(datasetPath, className);
        
        if ~exist(classPath, 'dir')
            fprintf('[!] Skipping missing class directory: %s\n', classPath);
            continue;
        end
        
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
        rng(42);  
        imageFiles = imageFiles(randperm(length(imageFiles), numImages));
        
        classFeatures = [];
        validCount = 0;
        
        for i = 1:numImages
            try
                imagePath = fullfile(classPath, imageFiles(i).name);
                img = imread(imagePath);
                
                if useDeepFeatures
                    % Deep Features (SqueezeNet)
                    if size(img, 3) == 1, img = cat(3,img,img,img); end
                    if size(img, 3) == 4, img = img(:,:,1:3); end
                    imgResized = imresize(img, inputSize(1:2));
                    % Extract activations from fire9-concat (high-level visual features)
                    feat = activations(net, imgResized, 'fire9-concat', 'OutputAs', 'rows');
                    features = feat;
                    if isempty(featureNames)
                        featureNames = arrayfun(@(x) sprintf('DeepFeat_%d', x), 1:length(features), 'UniformOutput', false);
                    end
                else
                    % Hand-Crafted Features
                    processedImg = preprocessImage(img);
                    [features, names] = extractFeatures(processedImg);
                    if isempty(featureNames)
                        featureNames = names;
                    end
                end
                
                validCount = validCount + 1;
                classFeatures(validCount, :) = features;
                
            catch
                % Skip corrupted images
                continue;
            end
            
            if mod(i, 200) == 0
                fprintf('      Progress: %d/%d\n', i, numImages);
            end
        end
        
        if validCount > 0
            allFeatures = [allFeatures; classFeatures(1:validCount, :)]; %#ok<AGROW>
            allLabels = [allLabels; repmat({className}, validCount, 1)]; %#ok<AGROW>
            samplesPerClass(c) = validCount;
        end
        
        fprintf('      Completed: %d valid images\n\n', validCount);
    end
    
    if isempty(allFeatures)
        error('No features were extracted. Check dataset path and image formats.');
    end
    
    %% Normalize features
    fprintf('─── PHASE 2: Feature Normalization ───\n\n');
    featureMean = mean(allFeatures, 1);
    featureStd = std(allFeatures, 0, 1);
    featureStd(featureStd == 0) = 1; 
    X = (allFeatures - featureMean) ./ featureStd;
    Y = categorical(allLabels);
    
    %% Bayesian Hyperparameter Optimization
    fprintf('─── PHASE 3: A++ Bayesian Hyperparameter Tuning ───\n\n');
    fprintf('Searching for optimal SVM BoxConstraint and KernelScale...\n');
    
    % Use Bayesian Optimization to find the best SVM parameters
    t = templateSVM('Standardize', false, 'KernelFunction', 'rbf');
    
    classifier = fitcecoc(X, Y, ...
        'Learners', t, ...
        'Coding', 'onevsall', ...
        'ClassNames', classNames, ...
        'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
        'HyperparameterOptimizationOptions', struct(...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'MaxObjectiveEvaluations', 20, ... 
            'Verbose', 0, ...
            'ShowPlots', false));
            
    %% Cross-Validation Results
    fprintf('\n─── PHASE 4: Model Evaluation ───\n\n');
    cvModel = crossval(classifier, 'KFold', 5);
    cvAccuracy = 1 - kfoldLoss(cvModel);
    
    fprintf('Final Cross-Validation Accuracy: %.2f%%\n', cvAccuracy * 100);
    
    % Detailed metrics
    [YPred, ~] = kfoldPredict(cvModel);
    confMat = confusionmat(Y, YPred);
    
    %% Save Model
    fprintf('\n─── PHASE 5: Saving Model ───\n\n');
    
    model.classifier = classifier;
    model.classNames = classNames;
    model.featureNames = featureNames;
    model.featureMean = featureMean;
    model.featureStd = featureStd;
    model.useDeepFeatures = useDeepFeatures;
    model.trainStats.numSamples = size(X, 1);
    model.trainStats.numFeatures = size(X, 2);
    model.trainStats.cvAccuracy = cvAccuracy;
    model.trainStats.trainDate = datestr(now);
    model.confusionMatrix = confMat;
    
    modelsPath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'models');
    if ~exist(modelsPath, 'dir'), mkdir(modelsPath); end
    
    if useDeepFeatures
        modelFile = fullfile(modelsPath, 'foodClassifier_hybrid.mat');
    else
        modelFile = fullfile(modelsPath, 'foodClassifier.mat');
    end
    save(modelFile, 'model');
    
    fprintf('A++ optimized model saved to: %s\n', modelFile);
end
