%% CLEAN DATASET - Remove Corrupted Images
% Iterates through the dataset and moves unreadable files to a backup folder.

datasetPath = '/Users/izwan/CSC566_MINI GROUP PROJECT_HAWKER FOOD CALORIE_TEAMONE/dataset/train';
corruptedPath = fullfile(fileparts(datasetPath), 'corrupted');

if ~exist(corruptedPath, 'dir')
    mkdir(corruptedPath);
end

% Get all image files
imds = imageDatastore(datasetPath, 'IncludeSubfolders', true);
files = imds.Files;
numFiles = numel(files);

fprintf('Checking %d files for corruption...\n', numFiles);

corruptedCount = 0;
for i = 1:numFiles
    try
        img = imread(files{i});
        if isempty(img)
            error('Empty image');
        end
    catch
        corruptedCount = corruptedCount + 1;
        [~, name, ext] = fileparts(files{i});
        parentDir = fileparts(files{i});
        [~, parentName] = fileparts(parentDir);
        
        targetFolder = fullfile(corruptedPath, parentName);
        if ~exist(targetFolder, 'dir')
            mkdir(targetFolder);
        end
        
        movefile(files{i}, fullfile(targetFolder, [name ext]));
        fprintf('Moved corrupted file: %s/%s%s\n', parentName, name, ext);
    end
    
    if mod(i, 500) == 0
        fprintf('Progress: %d/%d\n', i, numFiles);
    end
end

fprintf('Done! Moved %d corrupted files to: %s\n', corruptedCount, corruptedPath);
