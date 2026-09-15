function plotModelComparison(metricsTable, resultsDirectory, saveFigures)
%PLOTMODELCOMPARISON Plot cross-validation accuracy and macro-F1 scores.

figure('Name', 'Model Comparison', 'Color', 'w');
bar([metricsTable.MeanAccuracy, metricsTable.MeanMacroF1] * 100);
set(gca, 'XTick', 1:height(metricsTable), ...
    'XTickLabel', cellstr(metricsTable.Model));
ylabel('Score (%)');
title('Cross-Validation Model Comparison');
legend({'Accuracy', 'Macro-F1'}, 'Location', 'southoutside');
grid on;
xtickangle(25);

if saveFigures
    saveas(gcf, fullfile(resultsDirectory, 'model_comparison.png'));
    writetable(metricsTable, fullfile(resultsDirectory, 'metrics.csv'));
end
end
