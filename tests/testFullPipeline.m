%% TEST FULL PIPELINE - Integration Test for Complete System
% Tests the complete analysis pipeline end-to-end
%
% Run with: run('tests/testFullPipeline.m')

function testFullPipeline()
    fprintf('\n=== Testing Full Analysis Pipeline ===\n\n');
    
    % Get project path
    testDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(testDir);
    addpath(genpath(projectRoot));
    
    % Find test images
    foodClasses = {'nasi_lemak', 'roti_canai', 'satay', 'laksa', ...
                   'popiah', 'kaya_toast', 'mixed_rice'};
    
    %% Test each food class
    successCount = 0;
    totalTests = 0;
    
    for i = 1:length(foodClasses)
        className = foodClasses{i};
        classPath = fullfile(projectRoot, 'dataset', 'train', className);
        
        images = dir(fullfile(classPath, '*.jpg'));
        if isempty(images)
            images = dir(fullfile(classPath, '*.jpeg'));
        end
        if isempty(images)
            images = dir(fullfile(classPath, '*.png'));
        end
        
        if isempty(images)
            fprintf('  [SKIP] %s: No images found\n', className);
            continue;
        end
        
        % Test with first image of each class
        testImagePath = fullfile(classPath, images(1).name);
        totalTests = totalTests + 1;
        
        fprintf('Testing %s... ', className);
        
        try
            % Run the pipeline (without classifier for now, just structure)
            img = imread(testImagePath);
            processedImg = preprocessImage(img);
            [features, ~] = extractFeatures(processedImg);
            [mask, labeledRegions, ~] = segmentFood(processedImg);
            [portionRatio, portionLabel, ~] = estimatePortion(mask, className);
            [calories, nutrition] = calculateCalories(className, portionRatio);
            
            % Validate outputs
            assert(~isempty(processedImg), 'Pre-processing failed');
            assert(length(features) == 124, 'Feature extraction failed');
            assert(~isempty(mask), 'Segmentation failed');
            assert(portionRatio > 0, 'Portion estimation failed');
            assert(calories > 0, 'Calorie calculation failed');
            
            fprintf('PASSED (%.0f kcal, %s)\n', calories, portionLabel);
            successCount = successCount + 1;
            
        catch ME
            fprintf('FAILED: %s\n', ME.message);
        end
    end
    
    %% Summary
    fprintf('\n=== Pipeline Test Summary ===\n');
    fprintf('Passed: %d/%d tests\n', successCount, totalTests);
    
    if successCount == totalTests
        fprintf('All tests PASSED!\n');
    else
        fprintf('Some tests failed. Check logs above.\n');
    end
end
