# Iris Flower Classification

## Overview

This Python script is designed to classify iris flowers into different species using various machine learning algorithms. It loads the Iris dataset from "IRIS.csv", performs data exploration, visualization, preprocessing, model training, and evaluation.

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/shreyuu/CODSOFT.git
   cd DataScience/task_03(IRIS_FLOWER_CLASSIFICATION)
   ```
2. **Run the script:**

   ```bash
   python python_main.py
   ```

## Description

### File Structure

- `python_main.py`: The main Python script containing data loading, exploration, visualization, preprocessing, model training, and evaluation.
- `IRIS.csv`: Dataset containing information about iris flowers.

### Functionality

1. **Data Loading and Exploration:**

   - The script loads the Iris dataset from the provided CSV file.
   - Basic information, shape, data types, summary statistics, and missing values are displayed for exploration.
2. **Data Visualization:**

   - Histograms and scatter plots are created to visualize the distribution of features (sepal length, sepal width, petal length, petal width) and their relationships with species.
3. **Data Preprocessing:**

   - The 'species' column is encoded using LabelEncoder for model training.
   - The dataset is split into training and testing sets.
4. **Model Training and Evaluation:**

   - Three classification algorithms (Logistic Regression, K Nearest Neighbors, Decision Tree Classifier) are trained and evaluated.
   - Model accuracy scores are calculated and displayed.

## Acknowledgments

Feel free to modify the script to experiment with different classification algorithms, hyperparameters, or feature engineering techniques to improve the accuracy of iris flower classification.
