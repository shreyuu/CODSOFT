# Titanic Survival Prediction

## Overview

This Python script is designed to predict the survival of passengers aboard the Titanic using machine learning models. It utilizes various preprocessing techniques, oversampling with SMOTE, and three classification algorithms: Logistic Regression, Random Forest Classifier, and Gradient Boosting Classifier.

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/shreyuu/CODSOFT.git
   cd DataScience/task_01(TITANIC_SURVIVAL_PREDICTION)
   ```
2. **Run the script:**

   ```bash
   python project_main.py
   ```

## Description

### File Structure

- `project_main.py`: The main Python script containing data loading, preprocessing, model training, evaluation, and visualization.
- `tested.csv`: Dataset containing Titanic passenger information.

### Functionality

1. **Data Loading and Preprocessing:**

   - The script loads the Titanic dataset from the provided CSV file.
   - Missing values in the 'Age' and 'Fare' columns are filled with the median.
   - Missing values in the 'Cabin' column are replaced with 'Unknown'.
   - Categorical features are encoded using LabelEncoder.
   - Additional features like 'Title', 'Age_Group', and 'Family_Size' are derived from existing columns.
2. **Model Training and Evaluation:**

   - Three classifiers (Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier) are trained and evaluated.
   - SMOTE is used for oversampling to handle class imbalance.
   - Classification reports and mean cross-validation scores are generated for each model.
3. **Data Visualization:**

   - Various plots are created to visualize the distribution of features and survival outcomes among passengers.

## Acknowledgments

Feel free to modify the script to experiment with different preprocessing techniques, feature engineering, or machine learning algorithms to improve predictive performance.
