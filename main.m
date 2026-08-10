% MATLAB Machine Learning Project: Iris Flower Classification

% Loading the built-in Iris Dataset
load fisheriris;

X = meas;  % Features (sepal length, width and petal length, width)
Y = species;  % Target labels

c = cvpartition(Y, 'Holdout', 0.3);   % Splitting the dataset (Training and Testing)
idxTrain = training(c);
idxTest = test(c);

XTrain = X(idxTrain, :);
YTrain = Y(idxTrain, :);
XTest = X(idxTest, :);
YTest = Y(idxTest, :);

fprintf('Data loaded and split into training (70%%) and testing (30%%) sets.\n');

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
YPred_SVM = predict(Mdl_SVM, XTest);
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
[~, YTrain_num] = grp2idx(YTrain);
[~, YTest_num] = grp2idx(YTest);

Mdl_NN = fitcnet(XTrain, YTrain_num, 'LayerSizes', [10, 5], 'Standardize', true);
YPred_NN_num = predict(Mdl_NN, XTest);
YPred_NN = Mdl_NN.ClassNames(YPred_NN_num);
acc_NN = sum(strcmp(YPred_NN, YTest)) / numel(YTest);

fprintf('Neural Network Trained. Accuracy: %.1f%%\n', acc_NN*100);

%  Visualizations

%  Accuracy Comparison Bar Chart 
figure;
modelNames = {'Decision Tree', 'SVM', 'kNN', 'Neural Network'};
accuracies = [acc_DT, acc_SVM, acc_kNN, acc_NN];
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
rocObj = rocmetrics(YTest, scores(Mdl_SVM, XTest), Mdl_SVM.ClassNames);
plot(rocObj);
title('ROC Curve: SVM Performance');

fprintf('\nAll visualizations generated. Project complete\n');

