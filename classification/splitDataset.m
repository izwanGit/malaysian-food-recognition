%% SPLIT DATASET - Create Train/Test Split
% Moves 20% of images from each class to test folder
%
% Usage:
%   splitDataset()  % Uses default 80/20 split
%   splitDataset(testRatio)  % Custom split ratio

function splitDataset(testRatio)
    if nargin < 1
        testRatio = 0.2;  % 20% for testing
    end
    
    % Paths
    baseDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(baseDir);
    trainPath = fullfile(projectRoot, 'dataset', 'train');
    testPath = fullfile(projectRoot, 'dataset', 'test');
    
    % Classes
    classNames = {'nasi_lemak', 'roti_canai', 'satay', 'laksa', ...
                  'popiah', 'kaya_toast', 'mixed_rice'};
    
    fprintf('\n=== DATASET SPLIT ===\n');
    fprintf('Train path: %s\n', trainPath);
    fprintf('Test path: %s\n', testPath);
    fprintf('Test ratio: %.0f%%\n\n', testRatio * 100);
    
    rng(42);  % Reproducible random selection
    
    for c = 1:length(classNames)
        className = classNames{c};
        
        trainClassPath = fullfile(trainPath, className);
        testClassPath = fullfile(testPath, className);
        
        % Create test folder for this class
        if ~exist(testClassPath, 'dir')
            mkdir(testClassPath);
        end
        
        % Get all images in train folder
        imageFiles = [dir(fullfile(trainClassPath, '*.jpg')); ...
                      dir(fullfile(trainClassPath, '*.jpeg')); ...
                      dir(fullfile(trainClassPath, '*.png'))];
        
        if isempty(imageFiles)
            fprintf('[%s] No images found, skipping.\n', className);
            continue;
        end
        
        % Calculate how many to move
        numImages = length(imageFiles);
        numTest = round(numImages * testRatio);
        
        % Random selection
        shuffledIdx = randperm(numImages);
        testIdx = shuffledIdx(1:numTest);
        
        % Move images to test folder
        moved = 0;
        for i = testIdx
            srcFile = fullfile(trainClassPath, imageFiles(i).name);
            dstFile = fullfile(testClassPath, imageFiles(i).name);
            
            try
                movefile(srcFile, dstFile);
                moved = moved + 1;
            catch ME
                warning('Failed to move %s: %s', imageFiles(i).name, ME.message);
            end
        end
        
        fprintf('[%s] Moved %d images to test (%.0f%% of %d)\n', ...
                className, moved, (moved/numImages)*100, numImages);
    end
    
    fprintf('\n=== SPLIT COMPLETE ===\n');
    
    % Verify counts
    fprintf('\nVerification:\n');
    totalTrain = 0;
    totalTest = 0;
    for c = 1:length(classNames)
        className = classNames{c};
        
        trainFiles = dir(fullfile(trainPath, className, '*.jpg'));
        trainFiles = [trainFiles; dir(fullfile(trainPath, className, '*.png'))];
        testFiles = dir(fullfile(testPath, className, '*.jpg'));
        testFiles = [testFiles; dir(fullfile(testPath, className, '*.png'))];
        
        fprintf('  %s: Train=%d, Test=%d\n', className, length(trainFiles), length(testFiles));
        totalTrain = totalTrain + length(trainFiles);
        totalTest = totalTest + length(testFiles);
    end
    fprintf('\nTotal: Train=%d, Test=%d\n', totalTrain, totalTest);
end
