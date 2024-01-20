import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, jaccard_score, f1_score
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import log_loss
import tensorflow as tf
import matplotlib.pyplot as plt

# Load and prepare data
def load_and_preprocess(file_path, n_rows):
    data = pd.read_csv(file_path).drop(columns=["Unnamed: 0", "trans_num", "street"])
    return pd.get_dummies(data=data.head(n=n_rows))

# Train and evaluate models
def evaluate_models(X_train, y_train, X_test, y_test):
    models = [
        LR(solver='liblinear'),
        Tree(),
        KNN(n_neighbors=4),
        SVC(),
        # Add more models as needed
    ]

    for model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        jaccard = jaccard_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        model_name = model.__class__.__name__
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print(f"{model_name} Jaccard Index: {jaccard:.4f}")
        print(f"{model_name} F1 Score: {f1:.4f}")

        # Additional metrics for specific models
        if isinstance(model, LR):
            predict_proba = model.predict_proba(X_test)
            log_loss_value = log_loss(y_test, predict_proba)
            print(f"{model_name} Log Loss: {log_loss_value:.4f}")
        elif isinstance(model, Tree):
            # Additional metrics for decision tree
            Tree_JaccardIndex = jaccard_score(y_test, predictions)
            Tree_F1_Score = f1_score(y_test, predictions)
            print(f"{model_name} Jaccard Index: {Tree_JaccardIndex:.4f}")
            print(f"{model_name} F1 Score: {Tree_F1_Score:.4f}")
        elif isinstance(model, KNN):
            # Additional metrics for KNN
            KNN_Accuracy_Score = accuracy_score(y_test, predictions)
            KNN_JaccardIndex = jaccard_score(y_test, predictions)
            KNN_F1_Score = f1_score(y_test, predictions)
            print(f"{model_name} Accuracy: {KNN_Accuracy_Score:.4f}")
            print(f"{model_name} Jaccard Index: {KNN_JaccardIndex:.4f}")
            print(f"{model_name} F1 Score: {KNN_F1_Score:.4f}")
        elif isinstance(model, SVC):
            # Additional metrics for SVC
            SVC_Accuracy_Score = accuracy_score(y_test, predictions)
            # Add other metrics as needed
            print(f"{model_name} Accuracy: {SVC_Accuracy_Score:.4f}")
        elif isinstance(model, tf.keras.models.Sequential):
            # Additional metrics for TensorFlow model
            # You can add metrics from the `history` object obtained during model training
            tf_model_metrics = history.history  # Assuming you have 'history' object from model.fit()
            # Print or use metrics as needed

# Your subsequent code ...


# Load data
training_data = load_and_preprocess("/Users/shreyuu/VS_Code_projects/CODSOFT/MachineLearning/task_02(CREDITCARD_FRAUD_DETECTION)/archive/fraudTrain.csv", 20000)
X_train = training_data.drop(columns='is_fraud', axis=1)
y_train = training_data['is_fraud']

test_data = load_and_preprocess("/Users/shreyuu/VS_Code_projects/CODSOFT/MachineLearning/task_02(CREDITCARD_FRAUD_DETECTION)/archive/fraudTest.csv", 5000)
X_test = test_data.drop(columns='is_fraud', axis=1)
y_test = test_data['is_fraud']

# Align columns in X_train and X_test
X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

# Evaluate models
evaluate_models(X_train, y_train, X_test, y_test)