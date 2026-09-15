function interactivePrediction(model, featureNames)
%INTERACTIVEPREDICTION Predict an Iris species from five measurements.

fprintf('\nEnter measurements in centimeters. Press Enter to use the example.\n');
defaultValues = [5.1, 3.5, 1.4, 0.2];
values = zeros(1, 4);

for i = 1:4
    prompt = sprintf('%s [%.1f]: ', featureNames{i}, defaultValues(i));
    userValue = input(prompt);
    if isempty(userValue)
        userValue = defaultValues(i);
    end
    if ~isscalar(userValue) || ~isnumeric(userValue) || userValue <= 0
        error('Each measurement must be a positive numeric scalar.');
    end
    values(i) = userValue;
end

petalArea = values(3) * values(4);
inputRow = [values, petalArea];
predictedSpecies = predict(model, inputRow);

fprintf('Petal area: %.3f cm^2\n', petalArea);
fprintf('Predicted species: %s\n', char(predictedSpecies));
end
