import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import warnings


warnings.filterwarnings('ignore')


my_data = pd.read_csv('task_01(TITANIC_SURVIVAL_PREDICTION)/tested.csv')


columns_with_missing = ['Age', 'Fare']
for col in columns_with_missing:
    my_data[col].fillna(my_data[col].median(), inplace=True)
my_data['Cabin'].fillna('Unknown', inplace=True)

duplicate_count = my_data.duplicated().sum()
print("Number of duplicated rows:", duplicate_count)


for col in my_data.select_dtypes(include="object"):
    print(f"Unique values in {col}:", my_data[col].unique())


my_data['Title'] = my_data['Name'].str.extract(r',\s(.*?)\.')
my_data['Title'] = my_data['Title'].replace('Ms', 'Miss')
my_data['Title'] = my_data['Title'].replace('Dona', 'Mrs')
my_data['Title'] = my_data['Title'].replace(['Col', 'Rev', 'Dr'], 'Rare')


age_bins = [-np.inf, 17, 32, 45, 50, np.inf]
age_labels = ["Children", "Young", "Mid-Aged", "Senior-Adult", 'Elderly']
my_data['Age_Group'] = pd.cut(my_data['Age'], bins=age_bins, labels=age_labels)
my_data['Family_Size'] = my_data['SibSp'] + my_data['Parch']
my_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
my_data.insert(4, 'Age_Group', my_data.pop('Age_Group'))
my_data.insert(7, 'Family_Size', my_data.pop('Family_Size'))

encoder = LabelEncoder()
categorical_cols = ['Sex', 'Age_Group', 'Cabin', 'Embarked', 'Title']
for col in categorical_cols:
    my_data[col] = encoder.fit_transform(my_data[col])
X = my_data.drop('Survived', axis=1)
y = my_data['Survived']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_regression = LogisticRegression()
random_forest = RandomForestClassifier()
gradient_boosting = GradientBoostingClassifier()
logistic_regression.fit(X_train_scaled, y_train)
random_forest.fit(X_train_scaled, y_train)
gradient_boosting.fit(X_train_scaled, y_train)
lr_pred = logistic_regression.predict(X_test_scaled)
rf_pred = random_forest.predict(X_test_scaled)
gbc_pred = gradient_boosting.predict(X_test_scaled)
lr_report = classification_report(y_test, lr_pred)
lr_scores = cross_val_score(logistic_regression, X_train_scaled, y_train, cv=5, scoring='accuracy')

rf_report = classification_report(y_test, rf_pred)
rf_scores = cross_val_score(random_forest, X_train_scaled, y_train, cv=5, scoring='accuracy')

gbc_report = classification_report(y_test, gbc_pred)
gbc_scores = cross_val_score(gradient_boosting, X_train_scaled, y_train, cv=5, scoring='accuracy')


print('Logistic Regression Report:', lr_report)
print('Logistic Regression Mean Cross-Validation Score:', lr_scores.mean())
print('Random Forest Report:', rf_report)
print('Random Forest Mean Cross-Validation Score:', rf_scores.mean())
print('Gradient Boosting Report:', gbc_report)
print('Gradient Boosting Mean Cross-Validation Score:', gbc_scores.mean())

fig, ax = plt.subplots()
survived_counts = my_data['Survived'].value_counts()
ax.pie(survived_counts, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
ax.set_title('Distribution of Survived Passengers')
plt.show()

fig = px.pie(my_data, names='Pclass', title='Distribution of Passenger Class', color_discrete_sequence=px.colors.qualitative.Set3)
fig.show()

fig = px.histogram(my_data, x='Sex', title='Count of Different Genders', color='Sex', color_discrete_sequence=px.colors.qualitative.Set2)
fig.show()

fig = px.histogram(my_data, x='Age', nbins=30, title='Distribution of Age', histnorm='probability density')
fig.show()

fig = px.histogram(my_data, x='Fare', nbins=30, title='Distribution of Fare', histnorm='probability density')
fig.show()

fig = px.histogram(my_data, x='Embarked', title='Distribution of Embarked', color='Embarked', color_discrete_sequence=px.colors.qualitative.Set1)
fig.show()

fig = px.histogram(my_data, x='Title', title='Distribution of Titles', color='Title', color_discrete_sequence=px.colors.qualitative.Pastel)
fig.show()

fig = px.histogram(my_data, x='Pclass', color='Survived', barmode='group', title='Survival by Passenger Class', labels={'Pclass': 'Passenger Class'})
fig.show()

fig = px.histogram(my_data, x='Age_Group', color='Survived', barmode='group', title='Survival by Age Groups', labels={'Age_Group': 'Age Group'})
fig.show()

fig = px.histogram(my_data, x='Family_Size', color='Survived', barmode='group', title='Survival by Family Size', labels={'Family_Size': 'Family Size'})
fig.show()

fig = px.histogram(my_data, x='Embarked', color='Survived', barmode='group', title='Survival by Embarked Port', labels={'Embarked': 'Embarked Port'})
fig.show()