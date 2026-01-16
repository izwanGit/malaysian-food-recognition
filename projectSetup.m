%% PROJECT SETUP - Malaysian Hawker Food Recognition System
% This script initializes the project environment and verifies the dataset
% Run this script first before using any other functions
%
% CSC566 Mini Group Project - Team One
% Members:
%   - Muhammad Izwan bin Ahmad (2024938885)
%   - Ahmad Azfar Hakimi bin Mohammad Fauzy (2024544727)
%   - Afiq Danial bin Mohd Asrinnihar (2024974673)
%   - Alimi bin Ruzi (2024568765)

function projectSetup()
    clc;
    fprintf('========================================\n');
    fprintf('Malaysian Hawker Food Recognition System\n');
    fprintf('CSC566 Mini Group Project - Team One\n');
    fprintf('========================================\n\n');
    
    %% Get project root directory
    projectRoot = fileparts(mfilename('fullpath'));
    
    %% Add all subdirectories to MATLAB path
    fprintf('Setting up MATLAB paths...\n');
    addpath(genpath(projectRoot));
    
    % Save the path for future sessions
    savepath;
    fprintf('  [OK] Paths added successfully\n\n');
    
    %% Verify folder structure
    fprintf('Verifying folder structure...\n');
    requiredFolders = {'preprocessing', 'features', 'classification', ...
                       'segmentation', 'portion', 'calories', 'gui', ...
                       'dataset', 'models', 'results', 'tests'};
    
    for i = 1:length(requiredFolders)
        folderPath = fullfile(projectRoot, requiredFolders{i});
        if ~exist(folderPath, 'dir')
            mkdir(folderPath);
            fprintf('  [CREATED] %s\n', requiredFolders{i});
        else
            fprintf('  [OK] %s\n', requiredFolders{i});
        end
    end
    fprintf('\n');
    
    %% Verify dataset
    fprintf('Verifying dataset...\n');
    datasetPath = fullfile(projectRoot, 'dataset', 'train');
    foodClasses = {'nasi_lemak', 'roti_canai', 'satay', 'laksa', ...
                   'popiah', 'kaya_toast', 'mixed_rice'};
    
    totalImages = 0;
    for i = 1:length(foodClasses)
        classPath = fullfile(datasetPath, foodClasses{i});
        if exist(classPath, 'dir')
            images = dir(fullfile(classPath, '*.jpg'));
            if isempty(images)
                images = dir(fullfile(classPath, '*.jpeg'));
            end
            if isempty(images)
                images = dir(fullfile(classPath, '*.png'));
            end
            numImages = length(images);
            totalImages = totalImages + numImages;
            fprintf('  [OK] %s: %d images\n', foodClasses{i}, numImages);
        else
            fprintf('  [MISSING] %s\n', foodClasses{i});
        end
    end
    fprintf('\n  Total images: %d\n\n', totalImages);
    
    %% Display food database preview
    fprintf('Food Calorie Database (MyFCD):\n');
    fprintf('  %-15s | %s\n', 'Food Class', 'Base Calories (kcal)');
    fprintf('  %s\n', repmat('-', 1, 40));
    
    calorieData = {
        'nasi_lemak',   650;
        'roti_canai',   300;
        'satay',        200;
        'laksa',        500;
        'popiah',       185;
        'kaya_toast',   300;
        'mixed_rice',   620
    };
    
    for i = 1:size(calorieData, 1)
        fprintf('  %-15s | %d kcal\n', calorieData{i,1}, calorieData{i,2});
    end
    
    fprintf('\n========================================\n');
    fprintf('Project setup complete!\n');
    fprintf('Run "analyzeHawkerFood(imagePath)" to analyze an image.\n');
    fprintf('========================================\n');
end
