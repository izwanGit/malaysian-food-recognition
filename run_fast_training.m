%% Run Fast CNN Training
% A simple wrapper to invoke training with fast settings (3 epochs)
% Run this file to train the CNN model quickly.

% Ensure we are in project root or can find it
if ~exist('deeplearning', 'dir')
    % Try to find path relative to this file
    root = fileparts(mfilename('fullpath'));
    cd(root);
end

% Add all subfolders to path
addpath(genpath(pwd));

fprintf('===========================================================\n');
fprintf('   STARTING ROBUST FAST TRAINING (3 Epochs)\n');
fprintf('===========================================================\n');

% Set options for fast training
opts = struct();
opts.maxEpochs = 3;
opts.miniBatchSize = 16; % Smaller batch size for safety

try
    % Run training
    trainCNNClassifier([], opts);
    fprintf('\nSUCCESS: Fast training script finished.\n');
catch ME
    fprintf('\nERROR: Training failed.\n%s\n', ME.message);
    fprintf('%s\n', getReport(ME));
end

fprintf('===========================================================\n');
exit;
