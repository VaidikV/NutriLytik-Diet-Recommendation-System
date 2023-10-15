from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load pre-trained model
data = pd.read_csv('input.csv')
imputer = SimpleImputer(strategy='mean')  # for removing null values
scaler = MinMaxScaler()

# ----------------- FEATURE ENGINEERING -----------------

data['Health_Score'] = data['Calories'] - data['Fats'] + data['Proteins']

data['Is_Vegetarian'] = data['Veg']

data['Protein_to_Calories_Ratio'] = data['Proteins'] / data['Calories']
data['Fat_to_Protein_Ratio'] = data['Fats'] / data['Proteins']
# Using only certain features which help in increasing accuracy
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

# Normaliziing the values
columns_to_scale = ['Breakfast_Score', 'Lunch_Score', 'Dinner_Score']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.2, max_depth=4, subsample=0.8)
X = data[['Is_Vegetarian', 'Protein_to_Calories_Ratio', 'Fat_to_Protein_Ratio',
          'Calcium_to_Calories_Ratio', 'Sodium_to_Calories_Ratio', 'Potassium_to_Calories_Ratio',
          'Carbs_to_Calories_Ratio', 'Proteins_to_Iron_Ratio']]
X = imputer.fit_transform(X)
y = data['Health_Score']
model.fit(X, y)


@app.route('/')
def hello_world():
    return "Welcome To Nutri-Lytik: A Diet Recommendation System"


@app.route('/recommend', methods=['POST'])
def recommend_meal():
    try:
        # Get user input from the request
        user_input = request.get_json()
        meal_pref = user_input.get('meal_preference')
        vegetarian_pref = user_input.get('vegetarian_pref')
        calorie_limit = float(user_input.get('max_calories'))
        health_goal = user_input.get('health_goal')

        # Apply user preferences to filter recommendations
        filtered_data = data.copy()

        if meal_pref == "lunch":
            filtered_data = filtered_data[filtered_data['Lunch'] == 1]
        elif meal_pref == "breakfast":
            filtered_data = filtered_data[filtered_data['Breakfast'] == 1]
        else:
            filtered_data = filtered_data[filtered_data['Dinner'] == 1]

        if vegetarian_pref.lower() == 'yes':
            filtered_data = filtered_data[filtered_data['Is_Vegetarian'] == 0]

        if health_goal.lower() == 'weight loss':
            max_calorie_limit = calorie_limit * 0.2
        elif health_goal.lower() == 'weight gain':
            max_calorie_limit = calorie_limit * 1.7
        else:
            max_calorie_limit = calorie_limit

        filtered_data = filtered_data[filtered_data['Calories'] <= max_calorie_limit]

        # Prepare data for model prediction
        x_input = filtered_data[['Is_Vegetarian', 'Protein_to_Calories_Ratio', 'Fat_to_Protein_Ratio',
                                 'Calcium_to_Calories_Ratio', 'Sodium_to_Calories_Ratio', 'Potassium_to_Calories_Ratio',
                                 'Carbs_to_Calories_Ratio', 'Proteins_to_Iron_Ratio']]
        x_input = imputer.transform(x_input)

        # Predict recommendation scores
        recommendation_scores = model.predict(x_input)

        # Combine recommendation scores with the filtered data
        filtered_data['Recommendation_Score'] = recommendation_scores

        # Sort the filtered data by recommendation scores
        recommended_items = filtered_data.sort_values(by='Recommendation_Score', ascending=False)

        # Get the top N recommended food items (N can be determined based on user preferences)
        top_n_recommendations = recommended_items.head(10)["Food_items"]

        return jsonify({"recommendations": top_n_recommendations.to_dict()})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
