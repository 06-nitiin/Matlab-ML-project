function plotConfusionMatrices(predictions, Y, classNames, resultsDirectory, saveFigures)
%PLOTCONFUSIONMATRICES Plot one normalized confusion matrix per model.

modelFields = fieldnames(predictions);
for i = 1:numel(modelFields)
    field = modelFields{i};
    figure('Name', [field, ' Confusion Matrix'], 'Color', 'w');
    chart = confusionchart(Y, predictions.(field).Labels, ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');
    chart.ClassLabels = classNames;
    title(strrep(field, '_', ' '));

    if saveFigures
        saveas(gcf, fullfile(resultsDirectory, [field, '_confusion_matrix.png']));
    end
end
end
