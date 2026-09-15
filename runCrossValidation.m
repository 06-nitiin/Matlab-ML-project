function [metricsTable, predictions, foldInfo] = runCrossValidation(X, Y, config)
%RUNCROSSVALIDATION Compare classifiers using identical stratified folds.

classNames = categories(Y);
cv = cvpartition(Y, 'KFold', config.NumFolds);
modelNames = {'Decision Tree', 'SVM', 'kNN', 'Random Forest'};
includeNN = config.IncludeNeuralNetwork && exist('fitcnet', 'file') == 2;
if includeNN
    modelNames{end + 1} = 'Neural Network';
else
    fprintf('Neural network skipped: fitcnet is unavailable.\n');
end

numModels = numel(modelNames);
numClasses = numel(classNames);
numObservations = size(X, 1);

predictions = struct();
for i = 1:numModels
    key = matlab.lang.makeValidName(modelNames{i});
    predictions.(key).Labels = repmat(categorical(classNames(1), classNames), ...
        numObservations, 1);
    predictions.(key).Scores = nan(numObservations, numClasses);
end

foldAccuracy = nan(config.NumFolds, numModels);
foldMacroF1 = nan(config.NumFolds, numModels);

for fold = 1:config.NumFolds
    idxTrain = training(cv, fold);
    idxTest = test(cv, fold);
    XTrain = X(idxTrain, :);
    YTrain = Y(idxTrain);
    XTest = X(idxTest, :);
    YTest = Y(idxTest);

    trained = trainModels(XTrain, YTrain, config);

    for modelIndex = 1:numModels
        modelName = modelNames{modelIndex};
        key = matlab.lang.makeValidName(modelName);
        [YHat, scores] = predictModel(trained.(key), XTest, modelName, classNames);

        predictions.(key).Labels(idxTest) = YHat;
        predictions.(key).Scores(idxTest, :) = scores;

        foldAccuracy(fold, modelIndex) = mean(YHat == YTest);
        foldMacroF1(fold, modelIndex) = macroF1(YTest, YHat, classNames);
    end
end

metricsTable = table('Size', [numModels, 5], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'Model', 'MeanAccuracy', 'StdAccuracy', ...
    'MeanMacroF1', 'StdMacroF1'});

for modelIndex = 1:numModels
    metricsTable.Model(modelIndex) = string(modelNames{modelIndex});
    metricsTable.MeanAccuracy(modelIndex) = mean(foldAccuracy(:, modelIndex));
    metricsTable.StdAccuracy(modelIndex) = std(foldAccuracy(:, modelIndex));
    metricsTable.MeanMacroF1(modelIndex) = mean(foldMacroF1(:, modelIndex));
    metricsTable.StdMacroF1(modelIndex) = std(foldMacroF1(:, modelIndex));
end

foldInfo = struct('FoldAccuracy', foldAccuracy, 'FoldMacroF1', foldMacroF1, ...
    'ModelNames', {modelNames}, 'NumFolds', config.NumFolds);
end

function models = trainModels(XTrain, YTrain, config)
models = struct();
models.DecisionTree = fitctree(XTrain, YTrain);

svmTemplate = templateSVM('Standardize', true, 'KernelFunction', 'gaussian');
models.SVM = fitcecoc(XTrain, YTrain, 'Learners', svmTemplate);
models.kNN = fitcknn(XTrain, YTrain, 'NumNeighbors', 5, 'Standardize', 1);
models.RandomForest = fitcensemble(XTrain, YTrain, 'Method', 'Bag', ...
    'NumLearningCycles', 100);

if config.IncludeNeuralNetwork && exist('fitcnet', 'file') == 2
    models.NeuralNetwork = fitcnet(XTrain, YTrain, ...
        'LayerSizes', [10, 5], 'Standardize', true);
end
end

function [YHat, scores] = predictModel(model, XTest, modelName, classNames)
[YHat, rawScores] = predict(model, XTest);
YHat = categorical(YHat, classNames);
scores = normalizeScores(rawScores, modelName, classNames);
end

function scores = normalizeScores(rawScores, modelName, classNames)
scores = zeros(size(rawScores, 1), numel(classNames));
if strcmp(modelName, 'SVM') || strcmp(modelName, 'Random Forest')
    modelClassNames = categories(categorical(classNames));
    [~, locations] = ismember(modelClassNames, classNames);
    scores(:, locations(locations > 0)) = rawScores(:, locations > 0);
elseif size(rawScores, 2) == numel(classNames)
    scores = rawScores;
else
    scores(:, 1:size(rawScores, 2)) = rawScores;
end

% Convert arbitrary classifier outputs into comparable class probabilities.
rowTotals = sum(exp(scores), 2);
validRows = rowTotals > 0 & all(isfinite(rowTotals));
scores(validRows, :) = exp(scores(validRows, :)) ./ rowTotals(validRows);
end

function value = macroF1(actual, predicted, classNames)
f1 = zeros(numel(classNames), 1);
for i = 1:numel(classNames)
    current = categorical(classNames(i));
    tp = sum(actual == current & predicted == current);
    fp = sum(actual ~= current & predicted == current);
    fn = sum(actual == current & predicted ~= current);
    precision = safeDivide(tp, tp + fp);
    recall = safeDivide(tp, tp + fn);
    f1(i) = safeDivide(2 * precision * recall, precision + recall);
end
value = mean(f1);
end

function value = safeDivide(numerator, denominator)
if denominator == 0
    value = 0;
else
    value = numerator / denominator;
end
end
