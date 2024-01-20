# Machine Learning Model Evaluation

This Python script is designed to load, preprocess, and evaluate multiple machine learning models for a credit card fraud detection task. The script uses popular libraries such as pandas, scikit-learn, TensorFlow, and matplotlib.

## Prerequisites

Make sure you have the following libraries installed:

- pandas
- scikit-learn
- tensorflow
- matplotlib

You can install them using the following command:

```bash
pip install pandas scikit-learn tensorflow matplotlib
```

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/shreyuu/CODSOFT.git
   cd MachineLearning/task_02(CREDITCARD_FRAUD_DETECTION)

   ```
2. **Run the script:**

   ```bash
   python python_main.py
   ```

## Description

### File Structure

- `python_main.py`: The main Python script containing data loading, preprocessing, model training, and evaluation.
- `archive/fraudTrain.csv`: Training dataset for credit card fraud detection.
- `archive/fraudTest.csv`: Test dataset for credit card fraud detection.

### Functionality

1. **Data Loading and Preprocessing:**

   - The script loads the training and test datasets from CSV files.
   - Unnecessary columns ("Unnamed: 0", "trans_num", "street") are dropped.
   - One-hot encoding is applied to categorical variables.
2. **Model Training and Evaluation:**

   - Multiple models are trained and evaluated, including Logistic Regression, Decision Tree, K-Nearest Neighbors, and Support Vector Classifier.
   - Evaluation metrics such as accuracy, Jaccard Index, and F1 Score are printed for each model.
   - Additional metrics are computed for specific models, such as Log Loss for Logistic Regression and extra metrics for Decision Tree, KNN, SVC, and TensorFlow models.
3. **Data Alignment:**

   - Columns in the training and test datasets are aligned to ensure consistency.
4. **How to Load Data:**

   - Modify the file paths in the script to point to your datasets.
5. **How to Add Models:**

   - Additional models can be added to the `models` list in the `evaluate_models` function.

## Acknowledgments

Feel free to modify the script to suit your specific requirements or add more features as needed.
