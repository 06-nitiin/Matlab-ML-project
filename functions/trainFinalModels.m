function models = trainFinalModels(X, Y, config)
%TRAINFINALMODELS Train each classifier on the complete dataset.

models = struct();
models.DecisionTree = fitctree(X, Y);

svmTemplate = templateSVM('Standardize', true, 'KernelFunction', 'gaussian');
models.SVM = fitcecoc(X, Y, 'Learners', svmTemplate);
models.kNN = fitcknn(X, Y, 'NumNeighbors', 5, 'Standardize', 1);
models.RandomForest = fitcensemble(X, Y, 'Method', 'Bag', ...
    'NumLearningCycles', 100);

if config.IncludeNeuralNetwork && exist('fitcnet', 'file') == 2
    models.NeuralNetwork = fitcnet(X, Y, ...
        'LayerSizes', [10, 5], 'Standardize', true);
end
end
