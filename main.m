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

