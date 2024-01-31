# Movie Rating Prediction

## Overview

This Python script aims to predict movie ratings using various regression models. It loads movie data from the "IMDb Movies India.csv" file, performs data exploration, preprocessing, visualization, and model training using different regression algorithms.

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/shreyuu/CODSOFT.git
   cd DataScience/task_02(MOVIE_RATING_PREDICTION_WITH_PYTHON)
   ```
2. **Run the script:**

   ```bash
   python python_main.py
   ```

## Description

### File Structure

- `python_main.py`: The main Python script containing data loading, preprocessing, exploration, visualization, model training, and evaluation.
- `IMDb Movies India.csv`: Dataset containing information about Indian movies and their ratings.

### Functionality

1. **Data Loading and Preprocessing:**

   - The script loads the movie data from the provided CSV file.
   - Missing values are handled, and unnecessary columns like 'Name', 'Genre', 'Rating', 'Director', and actors are dropped.
   - The 'Duration' column is processed to extract only numeric values.
2. **Data Exploration and Visualization:**

   - Basic statistics, data types, and missing value summaries are printed for exploration.
   - A word cloud visualization is created to analyze movie genres.
3. **Modeling:**

   - Several regression models are trained and evaluated, including Linear Regression, Random Forest Regression, Decision Tree Regression, XGBoost, Gradient Boosting, Light Gradient Boosting, Cat Boost, and K Nearest Neighbors.
4. **Evaluation:**

   - Mean Squared Error (MSE) and R-squared (R2) scores are calculated to evaluate the performance of each model.

## Acknowledgments

Feel free to modify the script to experiment with different regression algorithms, hyperparameters, or feature engineering techniques to improve the accuracy of movie rating predictions.
