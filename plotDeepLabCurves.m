%% Plot DeepLabv3+ Training Curves (REAL DATA)
outputDir = 'report_figures';
load('models/foodSegmentationDL.mat');
ti2 = segModel.trainInfo;

f = figure('Position', [100,100,1200,500], 'Color', 'w');

subplot(1,2,1);
plot(ti2.TrainingLoss, 'b-', 'LineWidth', 1.5);
xlabel('Iteration', 'FontSize', 12);
ylabel('Loss', 'FontSize', 12);
title('DeepLabv3+ Training Loss', 'FontSize', 14, 'FontWeight', 'bold');
legend('Training Loss', 'Location', 'northeast');
grid on;

subplot(1,2,2);
plot(ti2.TrainingAccuracy, 'b-', 'LineWidth', 1.5);
xlabel('Iteration', 'FontSize', 12);
ylabel('Accuracy (%)', 'FontSize', 12);
finalAcc = ti2.TrainingAccuracy(end);
title(sprintf('DeepLabv3+ Training Accuracy (Final: %.2f%%)', finalAcc), 'FontSize', 14, 'FontWeight', 'bold');
legend('Training Accuracy', 'Location', 'southeast');
grid on;

sgtitle(sprintf('DeepLabv3+ Training Progress - %s', segModel.trainDate), 'FontSize', 16, 'FontWeight', 'bold');
saveas(f, fullfile(outputDir, 'Fig36_DeepLabv3_Training_Curves_REAL.png'));
close(f);
disp('Saved: Fig36_DeepLabv3_Training_Curves_REAL.png');
