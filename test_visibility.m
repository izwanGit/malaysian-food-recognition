function test_visibility()
    fprintf('SUCCESS: MATLAB sees new files in root!\n');
    fprintf('Current folder: %s\n', pwd);
    fprintf('trainCNN.m exist: %d\n', exist('trainCNN.m', 'file'));
end
