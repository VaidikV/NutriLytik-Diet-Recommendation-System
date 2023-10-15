import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

import random
data = pd.read_csv('food_dataset.csv.csv')

# ----------------- DATA PREPROCESSING -----------------

# UNDERSTAND THE DATASET

# print(data)
# Display the first few rows of the dataset
# print(data.head())

# Get information about the dataset, including data types and missing values
# print(data.info())

# Get summary statistics for numerical features
# print(data.describe())

# Check for missing values
# print(data.isnull().sum())
# data.fillna(0, inplace=True)

# Now solving Missing value error
# Initialize the imputer

imputer = SimpleImputer(strategy='mean')

# ----------------- FEATURE ENGINEERING -----------------

data['Health_Score'] = data['Calories'] - data['Fats'] + data['Proteins']

data['Is_Vegetarian'] = data['Veg']

data['Protein_to_Calories_Ratio'] = data['Proteins'] / data['Calories']
data['Fat_to_Protein_Ratio'] = data['Fats'] / data['Proteins']
# data['Sugars_to_Calories_Ratio'] = data['Sugars'] / data['Calories']
# data['Iron_to_Calories_Ratio'] = data['Iron'] / data['Calories']
data['Sodium_to_Calories_Ratio'] = data['Sodium'] / data['Calories']
data['Calcium_to_Calories_Ratio'] = data['Calcium'] / data['Calories']
data['Potassium_to_Calories_Ratio'] = data['Potassium'] / data['Calories']
data['Carbs_to_Calories_Ratio'] = data['Carbohydrates'] / data['Calories']
data['Proteins_to_Iron_Ratio'] = data['Proteins'] / data['Iron']

data['Breakfast_Score'] = data['Calories'] - data['Fats']
data['Lunch_Score'] = data['Calories'] + data['Proteins']
data['Dinner_Score'] = data['Proteins'] - data['Fats']

# ----------------- USER INPUT -----------------

# Get user input for dietary restrictions
vegetarian_pref = "yes"

# Get user input for health goals
calorie_limit = 1500.0

# Get user input for meal preferences
meal_pref = "breakfast"

# Get user input for health goals
health_goal = "healthy"

# Create a user profile
user_profile = {
    'Meal_Preference': meal_pref,
    'Vegetarian': vegetarian_pref,
    'Max_Calories': calorie_limit,
    'Health_Goal': health_goal,
}

# ----------------- RECOMMENDATION GENERATION -----------------

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=random.randint(1, 10000))

# all_data = pd.concat([train_data, test_data], axis=0)

# Define your features and target variable
X = train_data[['Is_Vegetarian', 'Protein_to_Calories_Ratio',
                'Fat_to_Protein_Ratio', 'Calcium_to_Calories_Ratio',
                'Sodium_to_Calories_Ratio', 'Potassium_to_Calories_Ratio', 'Carbs_to_Calories_Ratio',
                'Proteins_to_Iron_Ratio']]
X = imputer.fit_transform(X)  # removing null values
y = train_data['Health_Score']

# Initialize and train the model
model = GradientBoostingRegressor(n_estimators=300, random_state=42, learning_rate=0.2, max_depth=4, subsample=0.8)

model.fit(X, y)

# Extract the test data for prediction
X_test = test_data[['Is_Vegetarian', 'Protein_to_Calories_Ratio',
                    'Fat_to_Protein_Ratio', 'Calcium_to_Calories_Ratio',
                    'Sodium_to_Calories_Ratio', 'Potassium_to_Calories_Ratio', 'Carbs_to_Calories_Ratio',
                    'Proteins_to_Iron_Ratio']]
X_test = imputer.transform(X_test)  # removing null values

# Predict recommendation scores
recommendation_scores = model.predict(X_test)

# Combine recommendation scores with the test data
test_data['Recommendation_Score'] = recommendation_scores

# Now, filter and sort food items based on user preferences and health goals
filtered_test_data = test_data  # Start with all food items

# Apply user preferences for meals
if meal_pref == "lunch":
    filtered_test_data = filtered_test_data[filtered_test_data['Lunch'] == 1]
elif meal_pref == "breakfast":
    filtered_test_data = filtered_test_data[filtered_test_data['Breakfast'] == 1]
else:
    filtered_test_data = filtered_test_data[filtered_test_data['Dinner'] == 1]

# Apply vegetarian preference
if vegetarian_pref.lower() == 'yes':
    filtered_test_data = filtered_test_data[filtered_test_data['Is_Vegetarian'] == 0]


# Filter food items based on the user's health goal and maximum calorie limit
if health_goal.lower() == 'weight loss':
    # Reduce the calorie limit by a certain factor for weight loss (adjustable)
    max_calorie_limit = calorie_limit * 0.2
elif health_goal.lower() == 'weight gain':
    # Increase the calorie limit by a certain factor
    max_calorie_limit = calorie_limit * 1.7
else:
    max_calorie_limit = calorie_limit  # For staying healthy

# Filter food items based on the maximum calorie limit
filtered_test_data = filtered_test_data[filtered_test_data['Calories'] <= max_calorie_limit]

# Sort the filtered test data by recommendation scores
recommended_items = filtered_test_data.sort_values(by='Recommendation_Score', ascending=False)

# Get the top N recommended food items (N can be determined based on user preferences)
top_n_recommendations = recommended_items.head(20)

print(f"\nRecommended meals for {meal_pref}\n"
      f"Vegetarian: {vegetarian_pref}\n"
      f"Max Calories: {calorie_limit}\n"
      f"Health Goal: {health_goal}\n\n"
      f"Here are the recommendations:\n\n"
      f"{top_n_recommendations}")

# Calculate Mean Absolute Error (MAE)
y_true = test_data['Health_Score']
mae = mean_absolute_error(y_true, recommendation_scores)
print(f"Mean Absolute Error (MAE): {mae}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_true, recommendation_scores)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate R-squared (R2)
r2 = r2_score(y_true, recommendation_scores)
print(f"R-squared (R2): {r2}")

data.to_csv('updated_food_dataset.csv', index=False)  # for testing purposes
