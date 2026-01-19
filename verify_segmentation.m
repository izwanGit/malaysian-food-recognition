%% VERIFY SEGMENTATION
% Runs segmentation on random samples from each class to verify quality
% Saves visualizations to 'results/segmentation_review'

% Configuration
numSamples = 20;  % Check 20 samples per class
resultsDir = fullfile(pwd, 'results', 'segmentation_review');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

% Classes
classNames = {'nasi_lemak', 'roti_canai', 'satay', 'laksa', ...
              'popiah', 'kaya_toast', 'mixed_rice'};

fprintf('=== SEGMENTATION QUALITY CHECK ===\n');
fprintf('Processing %d images per class...\n\n', numSamples);

for c = 1:length(classNames)
    className = classNames{c};
    classDir = fullfile(pwd, 'dataset', 'train', className);
    
    % Get images
    files = dir(fullfile(classDir, '*.jpg'));
    files = [files; dir(fullfile(classDir, '*.png'))];
    
    if isempty(files)
        continue;
    end
    
    % Random selection
    rng(42); % reproducibility
    numImages = min(length(files), numSamples);
    indices = randperm(length(files), numImages);
    selectedFiles = files(indices);
    
    fprintf('Checking %s...\n', className);
    
    % Create a montage for this class
    figure('Visible', 'off', 'Position', [100 100 2000 800]);
    tiledlayout(4, 5, 'TileSpacing', 'tight', 'Padding', 'compact');
    
    for i = 1:numImages
        try
            % Load and segment
            imgPath = fullfile(classDir, selectedFiles(i).name);
            img = imread(imgPath);
            processed = preprocessImage(img);
            
            % Run segmentation (Active Contour)
            [mask, ~, overlay] = segmentFood(processed);
            
            % Add to plot
            nexttile;
            imshow(overlay);
            title(sprintf('Sample %d', i), 'FontSize', 8);
            
            % Save individual check if needed
            % imwrite(overlay, fullfile(resultsDir, sprintf('%s_%03d.jpg', className, i)));
            
        catch
            warning('Failed on %s', selectedFiles(i).name);
        end
    end
    
    % Save class overview
    overviewFile = fullfile(resultsDir, sprintf('REVIEW_%s.jpg', className));
    saveas(gcf, overviewFile);
    close(gcf);
    fprintf('  Saved review montage to: %s\n', overviewFile);
end

fprintf('\n=== REVIEW COMPLETE ===\n');
fprintf('Check the %s folder.\n', resultsDir);
exit;
