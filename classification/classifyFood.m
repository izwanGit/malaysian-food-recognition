%% CLASSIFY FOOD - Predict Food Class from Image
% Classifies a food image into one of the Malaysian hawker food classes
%
% Syntax:
%   [predictedClass, confidence] = classifyFood(img)
%   [predictedClass, confidence, allScores] = classifyFood(img)
%   [predictedClass, confidence, allScores] = classifyFood(img, modelPath, mode)
%
% Inputs:
%   img       - RGB image or path to image file
%   modelPath - Path to trained model file (optional)
%   mode      - 'svm' (default) or 'cnn'
%
% Outputs:
%   predictedClass - Predicted food class name (string)
%   confidence     - Confidence score (0-1)
%   allScores      - Scores for all classes

function [predictedClass, confidence, allScores] = classifyFood(img, modelPath, mode)
    %% Default parameters
    if nargin < 3
        mode = 'svm';
    end
    
    if nargin < 2 || isempty(modelPath)
        baseDir = fileparts(mfilename('fullpath'));
        projectRoot = fileparts(baseDir);
        if strcmpi(mode, 'cnn')
            modelPath = fullfile(projectRoot, 'models', 'foodCNN.mat');
        elseif strcmpi(mode, 'hybrid')
            modelPath = fullfile(projectRoot, 'models', 'foodClassifier_hybrid.mat');
        else
            modelPath = fullfile(projectRoot, 'models', 'foodClassifier.mat');
        end
    end
    
    %% Load model
    persistent cachedSVM cachedCNN cachedHybrid;
    
    if strcmpi(mode, 'cnn')
        % Load CNN model
        if isempty(cachedCNN) || ~strcmp(cachedCNN.path, modelPath)
            if ~exist(modelPath, 'file')
                error('classifyFood:ModelNotFound', ...
                      'CNN Model not found: %s\nRun trainCNN() first.', modelPath);
            end
            loaded = load(modelPath, 'trainedNet', 'classNames');
            cachedCNN.net = loaded.trainedNet;
            cachedCNN.classNames = loaded.classNames;
            cachedCNN.path = modelPath;
        end
        model = cachedCNN;
    elseif strcmpi(mode, 'hybrid')
        % Load Hybrid SVM model (trained with deep features)
        if isempty(cachedHybrid) || ~strcmp(cachedHybrid.path, modelPath)
            if ~exist(modelPath, 'file')
                error('classifyFood:ModelNotFound', ...
                      'Hybrid Model not found: %s\nRun trainClassifier(datasetPath, maxImages, true) first.', modelPath);
            end
            loaded = load(modelPath, 'model');
            cachedHybrid.model = loaded.model;
            cachedHybrid.path = modelPath;
        end
        model = cachedHybrid.model;
    else
        % Load SVM model
        if isempty(cachedSVM) || ~strcmp(cachedSVM.path, modelPath)
            if ~exist(modelPath, 'file')
                error('classifyFood:ModelNotFound', ...
                      'SVM Model not found: %s\nRun trainClassifier() first.', modelPath);
            end
            loaded = load(modelPath, 'model');
            cachedSVM.model = loaded.model;
            cachedSVM.path = modelPath;
        end
        model = cachedSVM.model;
    end
    
    %% Load and preprocess image
    if ischar(img) || isstring(img)
        if ~exist(img, 'file')
            error('classifyFood:ImageNotFound', 'Image file not found: %s', img);
        end
        img = imread(img);
    end
    
    % Preprocess is common for both
    processedImg = preprocessImage(img);
    
    %% Classification Logic
    if strcmpi(mode, 'cnn')
        % --- Deep Learning (CNN) ---
        inputSize = model.net.Layers(1).InputSize;
        resizedImg = imresize(processedImg, inputSize(1:2));
        
        [predictedLabel, scores] = classify(model.net, resizedImg);
        
        predictedClass = char(predictedLabel);
        confidence = max(scores);
        
        % Normalize scores if needed (already softmax from CNN)
        probScores = scores;
        classNames = model.classNames;
        
    elseif strcmpi(mode, 'hybrid')
        % --- Hybrid: SqueezeNet Features + SVM Classifier ---
        persistent hybridNet hybridInputSize;
        if isempty(hybridNet)
            hybridNet = squeezenet;
            hybridInputSize = hybridNet.Layers(1).InputSize;
        end
        imgResized = imresize(img, hybridInputSize(1:2));
        if size(imgResized, 3) == 1, imgResized = cat(3,imgResized,imgResized,imgResized); end
        if size(imgResized, 3) == 4, imgResized = imgResized(:,:,1:3); end
        features = activations(hybridNet, imgResized, 'fire9-concat', 'OutputAs', 'rows');
        
        normalizedFeatures = (features - model.featureMean) ./ model.featureStd;
        
        [predictedClass, scores] = predict(model.classifier, normalizedFeatures);
        
        if iscell(predictedClass)
            predictedClass = predictedClass{1};
        end
        
        % Convert SVM scores to probabilities (Temperature Scaling)
        temperature = 0.12; 
        scaledScores = scores / temperature;
        probScores = exp(scaledScores - max(scaledScores));
        probScores = probScores / sum(probScores);
        
        confidence = max(probScores);
        classNames = model.classNames;
        
    else
        % --- Classical Machine Learning (SVM) ---
        % Hand-Crafted Features (Color + Texture + HOG)
        features = extractFeatures(processedImg);
        
        normalizedFeatures = (features - model.featureMean) ./ model.featureStd;
        
        [predictedClass, scores] = predict(model.classifier, normalizedFeatures);
        
        if iscell(predictedClass)
            predictedClass = predictedClass{1};
        end
        
        % Convert SVM scores to probabilities (Temperature Scaling)
        temperature = 0.12; 
        scaledScores = scores / temperature;
        probScores = exp(scaledScores - max(scaledScores));
        probScores = probScores / sum(probScores);
        
        confidence = max(probScores);
        classNames = model.classNames;
    end
    
    %% Return all scores
    if nargout > 2
        allScores = struct();
        for i = 1:length(classNames)
            fieldName = matlab.lang.makeValidName(classNames{i});
            allScores.(fieldName) = probScores(i);
        end
    end
end
