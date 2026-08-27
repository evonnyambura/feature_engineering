# Data Preprocessing and Categorical Encoding Pipeline

This repository contains a Jupyter Notebook demonstrating data cleaning, missing value imputation, and categorical encoding techniques. It establishes an end-to-end machine learning pipeline using the Titanic dataset to train a baseline predictive model.

## Project Goal

The objective of this notebook is to transform raw data with missing values and text fields into a clean, numerical format for machine learning models. 

The pipeline performs four main steps:
1. Cross-Reference Imputation: Fills missing data by looking up matching records in a secondary file.
2. Statistical Cleaning: Fills remaining blank spots with medians and modes.
3. Categorical Encoding: Converts text-based words into numbers.
4. Predictive Modeling: Trains a baseline model to evaluate how well the prepared data predicts an outcome.


## Workflow and Breakdown

### 1. Ingesting and Cross-Referencing Data
The notebook uses glob to locate files and loads them using pandas. It builds an index map from a secondary data file to fill missing values in the primary dataset by looking up matching names.

### 2. Statistical Imputation
Remaining blank values are filled using statistical metrics to keep the data distribution balanced. Continuous numbers like age and fare are filled using the median value. Text categories like the embarking port are filled using the mode, which is the most common entry.

### 3. Categorical Encoding
Machine learning algorithms require numbers rather than text to perform mathematical calculations. The notebook loops through the dataset, automatically identifies text columns, and applies LabelEncoder to convert those words into unique integer IDs.

### 4. Baseline Modeling and Results
The encoded dataset is split into an 80% training set for learning and a 20% testing set for evaluation. A Logistic Regression model is trained on this data to predict the final target outcome.

* Baseline Prediction Accuracy: 84.7%


## Technologies used

* Pandas: For loading, filtering, and structuring data tables.
* NumPy: For handling missing values and mathematical arrays.
* Matplotlib: For visualizing target distributions with bar charts.
* Scikit-Learn: For data encoding, train-test splitting, and running the Logistic Regression algorithm.
