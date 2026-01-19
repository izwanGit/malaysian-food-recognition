%% CLEAN DATASET - Remove Corrupt and Low-Quality Images
% Scans the training dataset for:
% 1. Corrupt files (cannot be read)
% 2. "Ghost" images (Segmentation finds < 1% food content)
% Moves bad files to 'dataset/quarantine'

% Configuration
datasetPath = fullfile(pwd, 'dataset', 'train');
quarantineDir = fullfile(pwd, 'dataset', 'quarantine');
if ~exist(quarantineDir, 'dir'), mkdir(quarantineDir); end

classNames = {'nasi_lemak', 'roti_canai', 'satay', 'laksa', ...
              'popiah', 'kaya_toast', 'mixed_rice'};

fprintf('=== DATASET CLEANING PROTOCOL ===\n');
fprintf('Scanning for corrupt files and non-food images...\n\n');

totalMoved = 0;

for c = 1:length(classNames)
    className = classNames{c};
    classDir = fullfile(datasetPath, className);
    
    files = [dir(fullfile(classDir, '*.jpg')); dir(fullfile(classDir, '*.png'))];
    fprintf('Checking %s (%d images)...\n', className, length(files));
    
    for i = 1:length(files)
        filePath = fullfile(classDir, files(i).name);
        reason = '';
        
        try
            % check 1: Corrupt File
            img = imread(filePath);
            
            % Check 2: Tiny Images (thumbnails)
            if size(img, 1) < 50 || size(img, 2) < 50
                reason = 'Too Small';
            end
            
            % Check 3: "Empty" Content (using our Advanced Segmentation)
            if isempty(reason)
                 % Quick segmentation check
                 processed = preprocessImage(img);
                 mask = segmentFood(processed);
                 
                 foodRatio = sum(mask(:)) / numel(mask);
                 
                 % If food takes up less than 1% of the image, it's likely noise/background
                 if foodRatio < 0.01
                     reason = sprintf('No Food Detected (Ratio: %.4f)', foodRatio);
                 end
            end
            
        catch
            reason = 'Corrupt File (Read Error)';
        end
        
        % Move if flagged
        if ~isempty(reason)
            fprintf('  [QUARANTINE] %s: %s\n', files(i).name, reason);
            movefile(filePath, fullfile(quarantineDir, [className '_' files(i).name]));
            totalMoved = totalMoved + 1;
        end
    end
end

fprintf('\n=== CLEANING COMPLETE ===\n');
fprintf('Moved %d suspicious files to: %s\n', totalMoved, quarantineDir);
fprintf('You may manually review them to be sure.\n');
exit;
