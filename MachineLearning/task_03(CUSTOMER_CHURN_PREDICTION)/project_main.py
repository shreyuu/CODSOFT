import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,)

# Load the dataset
customer_data = pd.read_csv("Churn_Modelling.csv")
predictors = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
target_variable = "Exited"


X = customer_data[predictors]
y = customer_data[target_variable]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)

predicted_values = rf_model.predict(X_test_scaled)

print("Classification Report:\n", classification_report(y_test, predicted_values))
print("Confusion Matrix:\n", confusion_matrix(y_test, predicted_values))