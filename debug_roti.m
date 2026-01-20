%% DEBUG ROTI CANAI SEGMENTATION
addpath(genpath(pwd));
imagePath = 'dataset/test/roti_canai/189.jpg';
results = analyzeHawkerFoodDL(imagePath);
fprintf('\nFINAL RESULTS:\n');
fprintf('Class: %s\n', results.foodClass);
fprintf('Confidence: %.2f%%\n', results.confidence * 100);
fprintf('Mask Size: %d pixels\n', sum(results.mask(:)));
exit;
