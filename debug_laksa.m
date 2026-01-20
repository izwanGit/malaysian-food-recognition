%% DEBUG LAKSA SEGMENTATION
addpath(genpath(pwd));
imagePath = '/Users/izwan/.gemini/antigravity/brain/d1b56b10-f125-4cd9-98fa-3a2f82f97969/uploaded_image_1768929859419.png';
results = analyzeHawkerFoodDL(imagePath);
fprintf('\nFINAL RESULTS:\n');
fprintf('Class: %s\n', results.foodClass);
fprintf('Confidence: %.2f%%\n', results.confidence * 100);
fprintf('Mask Size: %d pixels\n', sum(results.foodMask(:)));
exit;
