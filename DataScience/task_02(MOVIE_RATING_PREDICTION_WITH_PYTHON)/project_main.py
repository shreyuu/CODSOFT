import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score as score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

# Load data
movie_data = pd.read_csv('IMDb Movies India.csv', encoding='latin1')
print(movie_data.Name)
df['Year'] = df['Year'].str.extract(r'\((\d{4})\)$', expand=False).astype(int)

# Data exploration
print(movie_data.head(11))
print(movie_data.describe())
print(movie_data.dtypes)
print(movie_data.isnull().sum())

# Data preprocessing
movie_data.dropna(inplace=True)
movie_data = movie_data[~movie_data['Duration'].str.contains('\(')]  # Filter out rows with values like '(2019)'
movie_data['Duration'] = movie_data['Duration'].str.extract('(\d+)')  # Extract only numeric part
movie_data['Duration'] = pd.to_numeric(movie_data['Duration'], errors='coerce')  # Convert to numeric

# Genre analysis
genre_counts = movie_data['Genre'].str.split(', ', expand=True).stack().value_counts()

# Visualization
plt.figure(figsize=(16, 6))
wordcloud = WordCloud(width=950, height=550, background_color='white').generate_from_frequencies(genre_counts)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('on')
plt.title('Genre Word Cloud')
plt.show()

# Modeling
def evaluate_model(y_test, preds, name):
    mse = mean_squared_error(y_test, preds)
    r2 = score(y_test, preds)
    return {'MSE': mse, 'R2': r2}

X = movie_data.drop(['Name', 'Genre', 'Rating', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], axis=1)
y = movie_data['Rating']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=1),
    "Decision Tree": DecisionTreeRegressor(random_state=1),
    "Extended Gradient Boosting": XGBRegressor(n_estimators=100, random_state=1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=60),
    "Light Gradient Boosting": LGBMRegressor(n_estimators=100, random_state=60),
    "Cat Boost": CatBoostRegressor(n_estimators=100, random_state=1, verbose=False),
    "K Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
}

results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    results[name] = evaluate_model(y_test, preds, name)

# Display results
models_df = pd.DataFrame(results.items(), columns=['MODELS', 'SCORES'])
models_df.sort_values