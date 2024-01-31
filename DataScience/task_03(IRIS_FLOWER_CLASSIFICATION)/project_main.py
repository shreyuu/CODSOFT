import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
iris_df = pd.read_csv("IRIS.csv")

# Display basic information about the dataset
print(iris_df.head(16))
print(iris_df.shape)
print(iris_df.info())
print(iris_df.describe())
print(iris_df.isnull().sum())

# Visualize data distribution
iris_df['sepal_length'].hist()
iris_df['sepal_width'].hist()
iris_df['petal_length'].hist()
iris_df['petal_width'].hist()
colors = ['red', 'black', 'teal']
species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Scatter plots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
unique_species = iris_df['species'].unique()  # Get unique species in the dataset
for i, ax in enumerate(axes.flat):
    if i < len(unique_species):  # Check if the index is within the range of unique species
        species_data = iris_df[iris_df['species'] == unique_species[i]]
        ax.scatter(species_data['sepal_length'], species_data['sepal_width'], c=colors[i], label=unique_species[i])
        ax.set_xlabel("Sepal Length")
        ax.set_ylabel("Sepal Width")
        ax.legend()
    else:
        break  # Exit the loop if the index exceeds the range of unique species


# Scatter plots for other features
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for i, ax in enumerate(axes.flat):
    if i < len(unique_species):  # Check if the index is within the range of unique species
        x = iris_df[iris_df['species'] == unique_species[i]]
        ax.scatter(x['petal_length'], x['petal_width'], c=colors[i], label=unique_species[i])
        ax.set_xlabel("Petal Length")
        ax.set_ylabel("Petal Width")
        ax.legend()
    else:
        break  # Exit the loop if the index exceeds the range of unique species


# Correlation heatmap
numeric_columns = iris_df.drop(columns='species')
corr = numeric_columns.corr()
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')

# Data preprocessing
le = LabelEncoder()
iris_df['species'] = le.fit_transform(iris_df['species'])
print(iris_df.head(16))

# Splitting the dataset
X = iris_df.drop(columns='species')
y = iris_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model training and evaluation
LR = LogisticRegression()
LR.fit(X_train, y_train)
KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)

# Model accuracy
LR_accuracy = LR.score(X_test, y_test) * 100
KNN_accuracy = KNN.score(X_test, y_test) * 100
DT_accuracy = DT.score(X_test, y_test) * 100
print(f"Accuracy by using Logistic Regression: {LR_accuracy}%")
print(f"Accuracy by using K Nearest Neighbors Algorithm: {KNN_accuracy}%")
print(f"Accuracy by using Decision Tree Classifier: {DT_accuracy}%")