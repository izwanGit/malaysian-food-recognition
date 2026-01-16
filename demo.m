%% DEMO - Quick Demonstration Script
% Run this script to see the system in action
%
% Usage:
%   1. Open MATLAB
%   2. Navigate to project folder
%   3. Run: demo

function demo()
    clc;
    fprintf('========================================\n');
    fprintf('Malaysian Hawker Food Recognition Demo\n');
    fprintf('========================================\n\n');
    
    % Get project path
    demoDir = fileparts(mfilename('fullpath'));
    if isempty(demoDir)
        demoDir = pwd;
    end
    
    % Add paths
    addpath(genpath(demoDir));
    
    %% Step 1: Run project setup
    fprintf('Step 1: Running project setup...\n');
    projectSetup();
    fprintf('\n');
    
    %% Step 2: Find a sample image
    fprintf('Step 2: Finding sample image...\n');
    
    foodClasses = {'nasi_lemak', 'roti_canai', 'satay', 'laksa', ...
                   'popiah', 'kaya_toast', 'mixed_rice'};
    
    sampleImage = '';
    sampleClass = '';
    
    for i = 1:length(foodClasses)
        classPath = fullfile(demoDir, 'dataset', 'train', foodClasses{i});
        images = dir(fullfile(classPath, '*.jpg'));
        if isempty(images)
            images = dir(fullfile(classPath, '*.jpeg'));
        end
        if isempty(images)
            images = dir(fullfile(classPath, '*.png'));
        end
        
        if ~isempty(images)
            sampleImage = fullfile(classPath, images(1).name);
            sampleClass = foodClasses{i};
            break;
        end
    end
    
    if isempty(sampleImage)
        fprintf('  No sample images found in dataset.\n');
        fprintf('  Please ensure the dataset is properly linked.\n');
        return;
    end
    
    fprintf('  Found: %s (%s)\n\n', sampleClass, images(1).name);
    
    %% Step 3: Check if classifier exists
    fprintf('Step 3: Checking classifier...\n');
    modelPath = fullfile(demoDir, 'models', 'foodClassifier.mat');
    
    if ~exist(modelPath, 'file')
        fprintf('  Classifier not found. Training is required.\n');
        fprintf('  To train, run: trainClassifier()\n');
        fprintf('  This may take several minutes.\n\n');
        
        answer = input('  Train classifier now? (y/n): ', 's');
        if lower(answer) == 'y'
            fprintf('\n  Training classifier (this may take 5-10 minutes)...\n');
            trainClassifier();
            fprintf('\n');
        else
            fprintf('\n  Skipping classifier training.\n');
            fprintf('  Demo will continue with manual analysis.\n\n');
        end
    else
        fprintf('  Classifier found!\n\n');
    end
    
    %% Step 4: Run analysis
    fprintf('Step 4: Analyzing sample image...\n\n');
    
    try
        results = analyzeHawkerFood(sampleImage);
        
        %% Step 5: Display results
        fprintf('\nStep 5: Displaying results...\n');
        displayResults(results);
        
    catch ME
        % If classifier fails, show manual analysis
        fprintf('\nNote: Classifier prediction failed. Showing processing only.\n');
        fprintf('Error: %s\n\n', ME.message);
        
        % Show images manually
        figure('Name', 'Demo Results', 'Position', [100 100 1000 400]);
        
        subplot(1, 3, 1);
        imshow(imread(sampleImage));
        title('Original Image');
        
        subplot(1, 3, 2);
        processedImg = preprocessImage(imread(sampleImage));
        imshow(processedImg);
        title('Pre-processed');
        
        subplot(1, 3, 3);
        [mask, ~, segImg] = segmentFood(processedImg);
        imshow(segImg);
        title(sprintf('Segmented (%.1f%% coverage)', sum(mask(:))/numel(mask)*100));
    end
    
    fprintf('\n========================================\n');
    fprintf('Demo Complete!\n');
    fprintf('========================================\n\n');
    fprintf('Next steps:\n');
    fprintf('  1. Run "HawkerFoodCalorieApp" to open the GUI\n');
    fprintf('  2. Run "trainClassifier()" to train the full classifier\n');
    fprintf('  3. Run test scripts in the "tests" folder\n');
end
