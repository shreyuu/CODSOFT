import pandas as pd
import numpy as np

# Load data
movie_data = pd.read_csv('IMDb Movies India.csv', encoding='latin1')

# Data exploration
print(movie_data.head(11))
print(movie_data.describe())
print(movie_data.dtypes)
print(movie_data.isnull().sum())

# Data preprocessing
movie_data.dropna(inplace=True)

# Handling non-numeric values in 'Year', 'Duration', and 'Votes' columns
columns_to_convert = ['Year', 'Duration', 'Votes']
for column in columns_to_convert:
    movie_data[column] = pd.to_numeric(movie_data[column], errors='coerce')

# Filter out rows with non-numeric values in 'Duration'
movie_data = movie_data[~movie_data['Duration'].isnull()]

# Extract numeric part of 'Duration' column
movie_data['Duration'] = movie_data['Duration'].astype(str).str.extract('(\d+)').astype(float)

# Genre analysis
genre_counts = movie_data['Genre'].str.split(', ', expand=True).stack().value_counts()
print("Genre Counts:", genre_counts)

# Visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud
plt.figure(figsize=(16, 6))
if genre_counts.empty:
    print("No genre data available.")
else:
    wordcloud = WordCloud(width=950, height=550, background_color='white').generate_from_frequencies(genre_counts)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('on')
    plt.title('Genre Word Cloud')
    plt.show()


# Modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score as score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

# Define features and target
X = movie_data.drop(['Name', 'Genre', 'Rating', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], axis=1)
y = movie_data['Rating']

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Model initialization
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

# Model training and evaluation
results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    results[name] = evaluate_model(y_test, preds, name)

# Display results
models_df = pd.DataFrame(results.items(), columns=['MODELS', 'SCORES'])
models_df.sort_values(by='SCORES', ascending=False)
