# Price Range Classification using Machine Learning

## Overview

This project focuses on predicting the price range of products using
machine learning classification techniques.

The project was completed as part of the Codebasics Data Science
Internship and includes data preprocessing, feature engineering,
model training, evaluation, and experiment tracking using MLflow.

## Objectives

- Clean and preprocess the survey dataset
- Perform feature engineering
- Encode categorical variables
- Train multiple classification models
- Evaluate model performance
- Compare model results
- Track experiments using MLflow
- Manage experiments and models using DagsHub

## Machine Learning Models

The following models are evaluated:

- Gaussian Naive Bayes
- Logistic Regression
- Support Vector Machine
- Random Forest
- XGBoost
- LightGBM

## Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- MLflow
- DagsHub
- Jupyter Notebook

## Dataset

The original dataset is not included in this repository
because it is a private/provided dataset.

## Experiment Tracking

MLflow is used to track model parameters, metrics, classification
reports, and trained models.

DagsHub is used as the remote MLflow tracking platform.

## Repository Structure

```text
price-range-classification/
├── week 3 and 4 task.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── .env.example