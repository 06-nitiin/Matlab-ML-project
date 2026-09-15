function plotMulticlassROC(predictions, Y, classNames, resultsDirectory, saveFigures)
%PLOTMULTICLASSROC Plot one-vs-rest ROC curves for every model and class.

modelFields = fieldnames(predictions);
figure('Name', 'Multiclass ROC Curves', 'Color', 'w');
hold on;
legendEntries = {};

for modelIndex = 1:numel(modelFields)
    field = modelFields{modelIndex};
    scores = predictions.(field).Scores;
    for classIndex = 1:numel(classNames)
        positive = (Y == categorical(classNames(classIndex)));
        [falsePositiveRate, truePositiveRate, ~, auc] = perfcurve(...
            positive, scores(:, classIndex), true);
        plot(falsePositiveRate, truePositiveRate, 'LineWidth', 1.4);
        legendEntries{end + 1} = sprintf('%s - %s (AUC %.3f)', ...
            strrep(field, '_', ' '), classNames{classIndex}, auc); %#ok<AGROW>
    end
end

plot([0, 1], [0, 1], 'k--', 'LineWidth', 1);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('One-vs-Rest Multiclass ROC Curves');
grid on;
legend(legendEntries, 'Location', 'eastoutside');
hold off;

if saveFigures
    saveas(gcf, fullfile(resultsDirectory, 'multiclass_roc_curves.png'));
end
end
