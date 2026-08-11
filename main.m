% MATLAB Machine Learning Project: Iris Flower Classification

% Loading the built-in Iris Dataset
load fisheriris;

% Feature Engineering (Want to give AI more information to work around with)
petalArea = meas(:, 3) .* meas(:, 4); % Petal Area
X = [meas, petalArea];
Y = species;  % Target labels

c = cvpartition(Y, 'Holdout', 0.3);   % Splitting the dataset (Training and Testing)
idxTrain = training(c);
idxTest = test(c);

XTrain = X(idxTrain, :);
YTrain = Y(idxTrain, :);
XTest = X(idxTest, :);
YTest = Y(idxTest, :);

fprintf('Data loaded with FEATURE ENGINEERING (Petal Area added).\n');

%  Decision Tree 
fprintf('Training Decision Tree Model..\n');
Mdl_DT = fitctree(XTrain, YTrain);
YPred_DT = predict(Mdl_DT, XTest); % Fixed: predict
acc_DT = sum(strcmp(YPred_DT, YTest)) / numel(YTest);

% SVM (Support Vector Machine) 
% I think finding the boundaries part would be important cause BOUNDARIES, sorry!
fprintf('Training SVM Model..\n');
t = templateSVM('Standardize', true, 'KernelFunction', 'gaussian');
Mdl_SVM = fitcecoc(XTrain, YTrain, 'Learners', t);
[YPred_SVM, scores_SVM] = predict(Mdl_SVM, XTest); % Fixed: predict
acc_SVM = sum(strcmp(YPred_SVM, YTest)) / numel(YTest); % Fixed: YTest

%  kNN (k-Nearest Neighbors) 
fprintf('Training kNN Model..\n');
Mdl_kNN = fitcknn(XTrain, YTrain, 'NumNeighbors', 5);
YPred_kNN = predict(Mdl_kNN, XTest);
acc_kNN = sum(strcmp(YPred_kNN, YTest)) / numel(YTest); % Fixed: / instead of .

fprintf('Models trained: Decision Tree (%.1f%%), SVM (%.1f%%), kNN (%.1f%%)\n', ...
    acc_DT*100, acc_SVM*100, acc_kNN*100);

% Neural Network (Finding complex patterns)
fprintf('Training Neural Network Model\n');
[YTrain_num, speciesNames] = grp2idx(YTrain);
YTest_num = grp2idx(YTest);

Mdl_NN = fitcnet(XTrain, YTrain_num, 'LayerSizes', [10, 5], 'Standardize', true);
YPred_NN_num = predict(Mdl_NN, XTest);
YPred_NN = speciesNames(YPred_NN_num);
acc_NN = sum(strcmp(YPred_NN, YTest)) / numel(YTest);

fprintf('Neural Network Trained. Accuracy: %.1f%%\n', acc_NN*100);

% Randome Forest (Ensemble Learning)
% Close to 50 trees whichj would hence result in better stability and accuracy.
fprintf('Training Random Forest Model\n');
Mdl_RF = fitcensemble(XTRain, YTrain, 'Method', 'Bag', 'NumLearningCycles', 50);
YPred_RF = predict(Mdl_RF, XTest);
acc_RF = sum(strcmp(YPred_RF, YTest)) / numel(YTest);

fprintf('Random forest Trained. Accuracy: %.1f%%\n', acc_RF*100);

%  Visualizations

%  Accuracy Comparison Bar Chart 
figure;
modelNames = {'Decision Tree', 'SVM', 'kNN', 'Neural Network', 'Random Forest'};
accuracies = [acc_DT, acc_SVM, acc_kNN, acc_NN, acc_RF];
bar(accuracies * 100);
set(gca, 'xticklabel', modelNames);
title('Model Accuracy Comparison');
ylabel('Accuracy (%)');
grid on;

% Confusion Matrix for the best model (e.g., SVM)
figure;
confusionchart(YTest, YPred_SVM);
title('Confusion Matrix: SVM Model');

% ROC Curve for the best model
figure;
rocObj = rocmetrics(YTest, scores_SVM, Mdl_SVM.ClassNames);
plot(rocObj);
title('ROC Curve: SVM Performance');

fprintf('\nAll visualizations generated. Project complete\n');


% Interactive Prediction (testing the model with own measurements)
fprintf('\n Customer Flower Prediction'\n);
myFlower =  [5.1, 3.5, 1.4, 0.2]; 

% Calculating the petal area for the custom flower;
myFlowerArea = myFlower(3) * myFlower(4);
myFlowerFull = [myFlower, myFlowerArea];

% Using the best model to predit
predictedSpecies = predict(Mdl_SVM, myFlowerFull);
fprintf('Input Measurements: SL=%.1f, SW=%.1f, PL=%.1f, PW=%.1f\n', myFlower);
fprintf('The AI predicts this flower is: %s/n', char(predictedSpecies));

