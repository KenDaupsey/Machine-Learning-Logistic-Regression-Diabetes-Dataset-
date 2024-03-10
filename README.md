# Logistic Regression for Diabetes Prediction

This repository contains a machine learning project that implements logistic regression to predict the likelihood of diabetes based on various features such as pregnancies, glucose level, blood pressure, skin thickness, insulin level, body mass index (BMI), diabetes pedigree function, and age. The project utilizes the popular Diabetes dataset and follows a structured approach to build, train, and evaluate the logistic regression model.

## Table of Contents

- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Getting Started

To get started with this project, you'll need to have Python and the following libraries installed:

- pandas
- numpy
- matplotlib
- scikit-learn

You can install these libraries using pip:

```
pip install pandas numpy matplotlib scikit-learn
```

## Dataset

The project utilizes the widely-used Diabetes dataset, which is included in the repository. The dataset contains several features and a target variable (outcome) indicating whether an individual has diabetes (1) or not (0). The dataset is loaded into a pandas DataFrame for further processing.

## Project Structure

The project follows a logical structure, consisting of the following steps:

1. **Import Libraries**: The necessary libraries for data manipulation, numerical computations, visualization, and machine learning tasks are imported.

2. **Load Dataset**: The Diabetes dataset is loaded into a pandas DataFrame, and the first few rows are displayed to provide an initial overview of the data.

3. **Data Preprocessing**: The features and target variable are selected from the dataset, and the data is split into training and testing sets using the `train_test_split` function from scikit-learn.

4. **Model Creation and Training**: A logistic regression model is created using the `LogisticRegression` class from scikit-learn. The model is trained on the training data using the `fit` method.

5. **Model Evaluation**: The performance of the trained model is evaluated using various metrics:
   - **Accuracy Score**: Calculated using the `accuracy_score` function from scikit-learn, it measures the overall accuracy of the model's predictions.
   - **Confusion Matrix**: Generated using the `confusion_matrix` function, it provides a breakdown of the model's predictions into true positives, true negatives, false positives, and false negatives.
   - **Classification Report**: Created using the `classification_report` function, it provides a detailed report with metrics such as precision, recall, and F1-score for each class.
   - **Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC)**: The ROC curve is plotted using the `roc_curve` function, and the AUC is calculated using the `auc` function. The ROC curve and AUC provide insights into the model's ability to distinguish between the two classes.

6. **Results Display**: The evaluation results, including accuracy, confusion matrix, classification report, and ROC curve, are displayed using print statements and matplotlib plots.

## Usage

To run the project, simply execute the provided code in a Python environment or notebook. The code is well-commented and easy to follow, making it accessible for both beginners and experienced practitioners.

## Results

The project provides a comprehensive evaluation of the logistic regression model's performance in predicting diabetes based on the given features. Key results include:

- **Accuracy Score**: The model achieves an accuracy score of approximately 0.7468, indicating that it correctly classifies instances around 75% of the time.
- **Confusion Matrix**: The confusion matrix provides a detailed breakdown of the model's predictions, allowing for further analysis of false positives and false negatives.
- **Classification Report**: The classification report includes metrics such as precision, recall, and F1-score for each class, providing insights into the model's predictive capabilities for both positive and negative instances.
- **ROC Curve and AUC**: The ROC curve and AUC value of 0.811 suggest that the model has a good ability to discriminate between the two classes, with 1.0 being the optimal value.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The Diabetes dataset used in this project is publicly available and widely used for machine learning research and education.
- The scikit-learn library provided the necessary tools for machine learning tasks, including logistic regression and evaluation metrics.
