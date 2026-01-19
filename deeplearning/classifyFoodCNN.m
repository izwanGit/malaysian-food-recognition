%% CLASSIFY FOOD CNN - Deep Learning Food Classification
% Classifies food images using trained CNN model
%
% Syntax:
%   [foodClass, confidence] = classifyFoodCNN(img)
%   [foodClass, confidence, allScores] = classifyFoodCNN(img)
%
% Inputs:
%   img - RGB image or path to image file
%
% Outputs:
%   foodClass  - Predicted food class name
%   confidence - Confidence score (0-1)
%   allScores  - Scores for all classes

function [foodClass, confidence, allScores] = classifyFoodCNN(img)
    persistent cnnModel
    
    %% Load model if not cached
    if isempty(cnnModel)
        modelPath = fullfile(fileparts(mfilename('fullpath')), '..', 'models', 'foodCNN.mat');
        
        if ~exist(modelPath, 'file')
            error(['CNN model not found. Train first with:\n' ...
                   '  >> trainCNNClassifier()']);
        end
        
        fprintf('Loading CNN model...\n');
        loaded = load(modelPath);
        
        % Build cnnModel struct from loaded variables
        cnnModel = struct();
        cnnModel.net = loaded.trainedNet;
        cnnModel.classNames = loaded.classNames;
        cnnModel.valAccuracy = loaded.accuracy;
        cnnModel.inputSize = cnnModel.net.Layers(1).InputSize;
        
        fprintf('  Model loaded (%.2f%% validation accuracy)\n', cnnModel.valAccuracy * 100);
    end
    
    %% Load image if path provided
    if ischar(img) || isstring(img)
        if ~exist(img, 'file')
            error('Image file not found: %s', img);
        end
        img = imread(img);
    end
    
    %% Preprocess image
    % Resize to network input size
    inputSize = cnnModel.inputSize;
    imgResized = imresize(img, inputSize(1:2));
    
    % Ensure RGB
    if size(imgResized, 3) == 1
        imgResized = repmat(imgResized, 1, 1, 3);
    end
    
    %% Classify
    [predictedLabel, scores] = classify(cnnModel.net, imgResized);
    
    %% Get results
    foodClass = char(predictedLabel);
    confidence = max(scores);
    
    %% Build all scores structure
    if nargout > 2
        allScores = struct();
        classNames = cnnModel.classNames;
        for i = 1:length(classNames)
            fieldName = classNames{i};
            allScores.(fieldName) = scores(i);
        end
    end
end
