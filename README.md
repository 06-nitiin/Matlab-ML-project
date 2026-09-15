# MATLAB Machine Learning Project: Iris Classification

This project compares several supervised machine-learning algorithms on MATLAB's built-in Fisher Iris dataset. It is designed as a reproducible, beginner-friendly portfolio project and demonstrates feature engineering, stratified cross-validation, classification metrics, visual evaluation, and interactive prediction.

## Models

The project evaluates the following classifiers:

- Decision Tree
- Gaussian-kernel Support Vector Machine using ECOC for multiclass classification
- k-Nearest Neighbors
- Random Forest-style bagged ensemble
- Neural Network, when `fitcnet` is available in the installed MATLAB version

The engineered feature set contains sepal length, sepal width, petal length, petal width, and petal area.

## Requirements

- MATLAB R2018b or newer
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox is optional. The neural-network model is skipped automatically when `fitcnet` is unavailable.
- A graphics-capable MATLAB session for plots

## Running the project

1. Clone or download the repository.
2. Open MATLAB and set the project directory as the Current Folder.
3. Run:

```matlab
main
```

The script uses a fixed random seed (`42`) and five stratified folds so that the model comparison can be reproduced. At the end, it asks for four Iris measurements. Press Enter at each prompt to use the included example values.

## Project structure

```text
Matlab-ML-project/
├── main.m
├── functions/
│   ├── interactivePrediction.m
│   ├── loadIrisData.m
│   ├── plotConfusionMatrices.m
│   ├── plotModelComparison.m
│   ├── plotMulticlassROC.m
│   ├── runCrossValidation.m
│   └── trainFinalModels.m
├── results/
│   └── Generated figures and metrics are saved here
├── .gitignore
└── README.md
```

## Evaluation outputs

Running `main.m` creates a `results` directory containing:

- `metrics.csv`: mean and standard deviation of accuracy and macro-F1 across folds.
- `model_comparison.png`: comparison of accuracy and macro-F1.
- One normalized confusion-matrix image for each model.
- `multiclass_roc_curves.png`: one-vs-rest ROC curves with AUC values.

Accuracy measures the proportion of correct predictions. Macro-F1 calculates F1 independently for each class and then gives every class equal weight, which is useful even when class sizes differ.

## Notes on reproducibility

The Iris dataset is loaded using MATLAB's built-in `fisheriris` dataset. The script sets the random-number generator with `rng(42, 'twister')`, uses the same stratified folds for every model, and trains final models on the complete dataset only after cross-validation is complete.

## Limitations and possible extensions

This project uses a small, clean dataset, so its results should not be interpreted as representative of real-world production performance. Useful future contributions include hyperparameter tuning, repeated cross-validation, feature-distribution plots, model persistence, unit tests, and a MATLAB App Designer interface.

## Contributing

Fork the repository, create a feature branch, make focused changes, and open a pull request. Please describe the MATLAB version and toolboxes used when reporting an issue or submitting a contribution.

## License

This project is released under the MIT License. Add a `LICENSE` file containing the standard MIT License text if one is not already present in your repository.
