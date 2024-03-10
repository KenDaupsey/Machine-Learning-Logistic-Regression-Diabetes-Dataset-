# Machine-Learning-Logistic-Regression-Diabetes-Dataset-

This repository contains a machine learning project that applies logistic regression to analyze the Diabetes dataset. The project aims to build a predictive model that can classify whether an individual has diabetes or not, based on various features such as pregnancies, glucose level, blood pressure, skin thickness, insulin level, body mass index (BMI), diabetes pedigree function, and age.

## Getting Started
To get started with this project, you'll need to have Python and the following libraries installed:

pandas
numpy
matplotlib
scikit-learn
You can install these libraries using pip:


Copy code
pip install pandas numpy matplotlib scikit-learn
Dataset
The Diabetes dataset used in this project is included in the repository. It contains several features and a target variable (outcome) indicating whether an individual has diabetes (1) or not (0).

## Project Structure
The project is structured as follows:

### Import Libraries: 
The necessary libraries (pandas, numpy, matplotlib, and scikit-learn) are imported for data manipulation, visualization, and machine learning tasks.

### Load Dataset: 
The Diabetes dataset is loaded into a pandas DataFrame.

### Data Preprocessing: 
The features and target variables are selected from the dataset, and the data is split into training and testing sets.

### Model Creation and Training: 
A logistic regression model is created and trained on the training data.

### Model Evaluation: 
The performance of the trained model is evaluated using various metrics:
Accuracy score
Confusion matrix
Classification report
Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC)

### Results: 
The evaluation results, including accuracy, confusion matrix, classification report, and ROC curve, are displayed.

### Usage
To run the project, simply execute the provided code in a Python environment or notebook. The code is well-commented and easy to follow.

### Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

### License
This project is licensed under the MIT License.

### Acknowledgments
The Diabetes dataset used in this project is publicly available and widely used for machine learning research and education.
The scikit-learn library provided the necessary tools for machine learning tasks, including logistic regression and evaluation metrics.
