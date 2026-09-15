%% MATLAB Machine Learning Project: Iris Classification
% Reproducible comparison of classical machine-learning classifiers.
%
% Requirements:
%   - MATLAB R2018b or newer
%   - Statistics and Machine Learning Toolbox
%   - Deep Learning Toolbox is optional; the neural network is skipped when
%     fitcnet is unavailable.

clear;
clc;
close all;

%% Configuration
rng(42, 'twister');
addpath(fullfile(fileparts(mfilename('fullpath')), 'functions'));

config = struct();
config.RandomSeed = 42;
config.NumFolds = 5;
config.IncludeNeuralNetwork = true;
config.SaveFigures = true;
config.ResultsDirectory = fullfile(fileparts(mfilename('fullpath')), 'results');

if config.SaveFigures && ~exist(config.ResultsDirectory, 'dir')
    mkdir(config.ResultsDirectory);
end

%% Load and prepare the Iris dataset
[X, Y, featureNames, classNames] = loadIrisData();

fprintf('Iris dataset loaded: %d observations, %d features, %d classes.\n', ...
    size(X, 1), size(X, 2), numel(classNames));
fprintf('Cross-validation: %d stratified folds; random seed: %d.\n\n', ...
    config.NumFolds, config.RandomSeed);

%% Evaluate all models using the same folds
[metricsTable, predictions, foldInfo] = runCrossValidation(X, Y, config); %#ok<ASGLU>

disp('Cross-validation results:');
disp(metricsTable);

%% Train final models on all available data
models = trainFinalModels(X, Y, config);

%% Visualizations
plotModelComparison(metricsTable, config.ResultsDirectory, config.SaveFigures);
plotConfusionMatrices(predictions, Y, classNames, config.ResultsDirectory, config.SaveFigures);
plotMulticlassROC(predictions, Y, classNames, config.ResultsDirectory, config.SaveFigures);

%% Interactive prediction using the final SVM model
fprintf('\nInteractive prediction uses the final SVM model.\n');
interactivePrediction(models.SVM, featureNames);

fprintf('\nProject complete. Results are available in: %s\n', config.ResultsDirectory);
