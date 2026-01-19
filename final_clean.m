%% FINAL CLEANUP - Direct Deletion of Corrupted Files
projectRoot = pwd;
subsets = {'train', 'test'};
fprintf('Starting final cleanup...\n');

for s = 1:length(subsets)
    subsetPath = fullfile(projectRoot, 'dataset', subsets{s});
    if ~exist(subsetPath, 'dir')
        fprintf('Directory not found: %s\n', subsetPath);
        continue;
    end
    
    fprintf('Cleaning subset: %s\n', subsets{s});
    % Get all files recursively
    files = dir(fullfile(subsetPath, '**', '*.*'));
    % Filter out directories and hidden files
    isImg = ~[files.isdir] & ~startsWith({files.name}, '.');
    files = files(isImg);
    
    deletedCount = 0;
    for i = 1:numel(files)
        filePath = fullfile(files(i).folder, files(i).name);
        try
            img = imread(filePath);
            if isempty(img)
                delete(filePath);
                deletedCount = deletedCount + 1;
            end
        catch
            delete(filePath);
            deletedCount = deletedCount + 1;
            fprintf('  [*] Deleted corrupted: %s\n', files(i).name);
        end
        if mod(i, 1000) == 0
            fprintf('  Progress: %d/%d\n', i, numel(files));
        end
    end
    fprintf('Finished %s: %d files deleted.\n', subsets{s}, deletedCount);
end
fprintf('All datasets are now clean. You can start training.\n');
exit;
