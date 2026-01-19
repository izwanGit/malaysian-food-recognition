%% VISUALIZE AUGMENTATION - Create figures for report
% Shows original vs multiple augmented versions to demonstrate methodology

function visualizeAugmentation(sampleImagePath)
    %% Paths
    basePath = '/Applications/MATLAB_R2025a.app/toolbox/images/imdata/CSC566_MINI GROUP PROJECT_HAWKER FOOD CALORIE_TEAMONE';
    if nargin < 1
        % Pick a random nasi lemak image if none provided
        nasiPath = fullfile(basePath, 'dataset', 'train', 'nasi_lemak');
        files = dir(fullfile(nasiPath, '*.jpg'));
        sampleImagePath = fullfile(nasiPath, files(1).name);
    end
    
    %% Load image
    img = imread(sampleImagePath);
    [~, name, ext] = fileparts(sampleImagePath);
    
    %% Create Figure
    fig = figure('Name', 'Data Augmentation Actions', ...
                 'Position', [100, 100, 1200, 600], ...
                 'Color', 'white');
    
    % 1. Original
    subplot(2, 4, 1);
    imshow(img);
    title('Original Image', 'FontSize', 12);
    
    % Generate 7 augmented versions
    for i = 2:8
        [augImg, types] = augmentImage(img);
        
        subplot(2, 4, i);
        imshow(augImg);
        
        % Make title from augmentation types
        typeStr = strjoin(types, '\n');
        title(sprintf('Augmented #%d\n%s', i-1, typeStr), 'FontSize', 9);
    end
    
    sgtitle('Data Augmentation Strategy - Nasi Lemak', ...
            'FontSize', 16, 'FontWeight', 'bold');
            
    %% Save to results
    resultsPath = fullfile(basePath, 'results');
    if ~exist(resultsPath, 'dir'), mkdir(resultsPath); end
    
    savePath = fullfile(resultsPath, 'augmentation_samples.png');
    saveas(fig, savePath);
    fprintf('Augmentation visualization saved to: %s\n', savePath);
end
