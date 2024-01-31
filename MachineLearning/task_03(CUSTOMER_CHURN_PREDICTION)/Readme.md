# Churn Prediction with Random Forest Classifier

## Overview

This Python script aims to predict customer churn using a Random Forest Classifier. It loads and preprocesses customer data from the "Churn_Modelling.csv" file, trains a Random Forest model, and evaluates its performance using classification metrics such as precision, recall, F1-score, and confusion matrix.

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/shreyuu/CODSOFT.git
   cd MachineLearning/task_03(CUSTOMER_CHURN_PREDICTION)
   ```
2. **Run the script:**

   ```bash
   python project_main.py
   ```

## Description

### File Structure

- `project_main.py`: The main Python script containing data loading, preprocessing, model training, and evaluation.
- `Churn_Modelling.csv`: Dataset containing customer information for churn prediction.

### Functionality

1. **Data Loading and Preprocessing:**

   - The script loads customer data from the provided CSV file.
   - Selected features (predictors) and the target variable ("Exited") are defined.
   - Data is split into training and testing sets using a 80:20 ratio.
   - Features are standardized using StandardScaler.
2. **Model Training and Evaluation:**

   - A Random Forest Classifier is trained on the scaled training data.
   - The trained model is used to predict churn on the test data.
   - Classification report and confusion matrix are generated to evaluate model performance.
3. **How to Load Data:**

   - Ensure the "Churn_Modelling.csv" file is in the same directory as the script.
   - Modify the file path in the script if the dataset is located elsewhere.

## Acknowledgments

Feel free to modify the script to incorporate additional features, experiment with different preprocessing techniques, or try out other machine learning algorithms for churn prediction.
