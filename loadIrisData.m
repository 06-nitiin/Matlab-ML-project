function [X, Y, featureNames, classNames] = loadIrisData()
%LOADIRISDATA Load MATLAB's Fisher Iris dataset and engineer petal area.

load fisheriris;

petalArea = meas(:, 3) .* meas(:, 4);
X = [meas, petalArea];
Y = categorical(species);

featureNames = {'SepalLength', 'SepalWidth', 'PetalLength', ...
    'PetalWidth', 'PetalArea'};
classNames = categories(Y);
end
