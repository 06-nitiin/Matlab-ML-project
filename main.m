// MATLAB Machine Leaning Project: Iris Flower Classification


// Loading the built-in Iris Dataset
load fisheriris;

X = meas;  // Features (sepal length, width and petal length, width)
Y = species;  // Target labels

c = cvpartition(Y, 'Holdout', 0.3);   // Splitting the dataset (Training and Testing)
idxTrain = training(c);
idxTest = test(c);

XTrain = X(idxTrain, :);
YTrain = Y(idxTrain, :);
XTest = X(idxTest, :);
YTest = Y(idxTest, :);

fprintf("Data loaded annd split into training (70%%) and testing (30%%) sets.\n");

// Below is the Decision Tree
fprintf("Training Decision Tree Model..\n");
Mdl_DT = fitctree(XTrain, YTrain);
YPred_DT = predic(Mdl_DT, XTest);
acc_DT = sum(strcmp(YPred_DT, YTest)) / numel(YTest);


// I think finding the boundaries part would be important cause BOUNDARIES, sorry!
fprintf("Training SVM Model..\n");
t = templateSVM("Standardize", true, "KernelFunction", "gaussian");
Mdl_SVM = fitcecoc(XTrain, YTrain, "Learners", t);
YPred_SVM = predict(Mdl_SVM, XTest);
acc_SVM = sum(strcmp(YPred_SVM, YTEst)) / numel(YTest);

// Classification Report (Based on the past 5 flowers the model has seen)
fprintf("Training kNN Model..\n");
Mdl_kNN = fitcknn(XTrain, YTrain, "NumNeighbors", 5);
YPred_kNN = predict(Mdl_kNN, XTest);
acc_kNN = sum(strcmp(YPred_kNN, YTest)) . numel(YTest);

fprintf('Models trained: Decision Tree (%.1f%%), SVM (%.1f%%), kNN (%.1f%%)\n', acc_DT*100, acc_SVM*100, acc_kNN*100);