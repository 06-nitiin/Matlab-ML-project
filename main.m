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

% --- 1. Decision Tree ---
fprintf('Training Decision Tree Model..\n');
Mdl_DT = fitctree(XTrain, YTrain);
YPred_DT = predict(Mdl_DT, XTest); % Fixed: predict
acc_DT = sum(strcmp(YPred_DT, YTest)) / numel(YTest);

% --- 2. SVM (Support Vector Machine) ---
% Finding boundaries is exactly what SVM does best!
fprintf('Training SVM Model..\n');
t = templateSVM('Standardize', true, 'KernelFunction', 'gaussian');
Mdl_SVM = fitcecoc(XTrain, YTrain, 'Learners', t);
YPred_SVM = predict(Mdl_SVM, XTest);
acc_SVM = sum(strcmp(YPred_SVM, YTest)) / numel(YTest); % Fixed: YTest

% --- 3. kNN (k-Nearest Neighbors) ---
fprintf('Training kNN Model..\n');
Mdl_kNN = fitcknn(XTrain, YTrain, 'NumNeighbors', 5);
YPred_kNN = predict(Mdl_kNN, XTest);
acc_kNN = sum(strcmp(YPred_kNN, YTest)) / numel(YTest); % Fixed: / instead of .

fprintf('Models trained: Decision Tree (%.1f%%), SVM (%.1f%%), kNN (%.1f%%)\n', ...
    acc_DT*100, acc_SVM*100, acc_kNN*100);
