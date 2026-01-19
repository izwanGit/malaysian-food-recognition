%% CLASSIFY FOOD - Predict Food Class from Image
% Classifies a food image into one of the Malaysian hawker food classes
%
% Syntax:
%   [predictedClass, confidence] = classifyFood(img)
%   [predictedClass, confidence, allScores] = classifyFood(img)
%   [predictedClass, confidence, allScores] = classifyFood(img, modelPath)
%
% Inputs:
%   img       - RGB image or path to image file
%   modelPath - Path to trained model file (optional)
%
% Outputs:
%   predictedClass - Predicted food class name (string)
%   confidence     - Confidence score (0-1)
%   allScores      - Scores for all classes

function [predictedClass, confidence, allScores] = classifyFood(img, modelPath)
    %% Load model
    persistent cachedModel;
    
    if nargin < 2
        baseDir = fileparts(mfilename('fullpath'));
        projectRoot = fileparts(baseDir);
        modelPath = fullfile(projectRoot, 'models', 'foodClassifier.mat');
    end
    
    % Load model if not cached or different path
    if isempty(cachedModel) || ~strcmp(cachedModel.path, modelPath)
        if ~exist(modelPath, 'file')
            error('classifyFood:ModelNotFound', ...
                  'Model file not found: %s\nRun trainClassifier() first.', modelPath);
        end
        loaded = load(modelPath, 'model');
        cachedModel.model = loaded.model;
        cachedModel.path = modelPath;
    end
    
    model = cachedModel.model;
    
    %% Load and preprocess image
    if ischar(img) || isstring(img)
        if ~exist(img, 'file')
            error('classifyFood:ImageNotFound', 'Image file not found: %s', img);
        end
        img = imread(img);
    end
    
    % Preprocess
    processedImg = preprocessImage(img);
    
    %% Extract features
    features = extractFeatures(processedImg);
    
    %% Normalize features using training statistics
    normalizedFeatures = (features - model.featureMean) ./ model.featureStd;
    
    %% Predict class
    [predictedClass, scores] = predict(model.classifier, normalizedFeatures);
    
    % Handle cell output
    if iscell(predictedClass)
        predictedClass = predictedClass{1};
    end
    
    %% Calculate confidence
    % For SVM, scores are negative loss values - convert to probabilities
    % Using softmax-like transformation
    if all(scores <= 0)
        % Negative loss scores - apply softmax with temperature scaling
        % Temperature < 1 makes distribution more peaked (higher max confidence)
        temperature = 0.3;  % Lower = higher confidence for winner
        scaledScores = scores / temperature;
        probScores = exp(scaledScores - max(scaledScores));  % Numerical stability
        probScores = probScores / sum(probScores);
    else
        % Positive scores - apply softmax with temperature
        temperature = 0.3;
        scaledScores = scores / temperature;
        probScores = exp(scaledScores - max(scaledScores));
        probScores = probScores / sum(probScores);
    end
    
    % Find predicted class score
    classIdx = find(strcmp(model.classNames, predictedClass), 1);
    if ~isempty(classIdx)
        confidence = probScores(classIdx);
    else
        confidence = max(probScores);
    end
    
    % Ensure confidence is in valid range
    confidence = max(0, min(1, confidence));
    
    %% Return all scores if requested
    if nargout > 2
        allScores = struct();
        for i = 1:length(model.classNames)
            % Clean field name (replace spaces/special chars)
            fieldName = matlab.lang.makeValidName(model.classNames{i});
            allScores.(fieldName) = probScores(i);
        end
    end
end
