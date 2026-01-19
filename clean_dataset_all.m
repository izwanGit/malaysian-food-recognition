%% COMPREHENSIVE DATA CLEANING - Train and Test
% Iterates through both train and test datasets and removes unreadable files.

projectRoot = pwd;
subsets = {'train', 'test'};
corruptedPath = fullfile(projectRoot, 'dataset', 'corrupted');

if ~exist(corruptedPath, 'dir')
    mkdir(corruptedPath);
end

for s = 1:length(subsets)
    subsetPath = fullfile(projectRoot, 'dataset', subsets{s});
    if ~exist(subsetPath, 'dir'), continue; end
    
    fprintf('Checking %s dataset...\n', subsets{s});
    
    imds = imageDatastore(subsetPath, 'IncludeSubfolders', true);
    files = imds.Files;
    numFiles = numel(files);
    
    corruptedCount = 0;
    for i = 1:numFiles
        try
            % Test if file exists and is readable
            info = imfinfo(files{i});
            img = imread(files{i});
        catch
            corruptedCount = corruptedCount + 1;
            [~, name, ext] = fileparts(files{i});
            parentDir = fileparts(files{i});
            [~, parentName] = fileparts(parentDir);
            
            % Unique folder for corrupted subset/class
            targetFolder = fullfile(corruptedPath, subsets{s}, parentName);
            if ~exist(targetFolder, 'dir'), mkdir(targetFolder); end
            
            movefile(files{i}, fullfile(targetFolder, [name ext]));
            fprintf('  [*] Removed: %s/%s/%s%s\n', subsets{s}, parentName, name, ext);
        end
        
        if mod(i, 500) == 0
            fprintf('  Progress: %d/%d\n', i, numFiles);
        end
    end
    fprintf('Finished %s: %d corrupted files found.\n\n', subsets{s}, corruptedCount);
end

fprintf('All datasets cleaned.\n');
