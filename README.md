# Linear-regression-applied-in-multi-trait-polygenic-scores.
## Project Overview
This project applies linear regression twice, each with a different feature as the target variable, to study variations in multi-trait polygenic scores (PGS). The goal is to analyze PGS data through various machine learning techniques and evaluate model performance across multiple regression approaches.

## Code Structure
The main code is divided into 14 sections to streamline the process of data handling, preparation, and model fitting:

- Data Import and Treatment: Import datasets and perform initial preprocessing.
- PCA without Normalization: Principal Component Analysis without normalization for initial dimensionality reduction.
- PCA with Normalization: Principal Component Analysis with normalization to improve feature scaling.
- Data Treatment: Additional data cleaning and transformations.
- Data Preparation for Machine Learning: Prepare the data for model training by feature engineering and selection.
- Data Split for Performance Evaluation: Split the data into training and testing sets.
- First Regression Model: Fit the first linear regression model with a specific target feature.
- Fitting Ridge Regression Model: Apply Ridge regression for regularization.
- Fitting Lasso Regression Model: Fit a Lasso regression model to enhance feature selection.
- Gaussian Process Regression: Test the data with Gaussian Process Regression for comparison.
- Second Regression Model: Fit the second linear regression model with a different target feature.
- Fitting Ridge Regression Model: Apply Ridge regression on the second model.
- Fitting Lasso Regression Model: Fit a Lasso model on the second regression target.
- Gaussian Process Regression: Test the second model with Gaussian Process Regression.
  
This structure allows for a comprehensive exploration of the data and model performance, providing insights into how different regression techniques handle variations in polygenic scores.
