from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import re

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")
CORS(app)

# Load dataset
data = pd.read_csv('food_dataset.csv')

# Load pre-trained model
model_filename = 'diet_recommendation_model.pkl'
loaded_model = joblib.load(model_filename)

# Define an imputer for handling missing values
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
        diabetic_pref = user_input.get('diabetic_pref')

        if diabetic_pref.lower() == 'no':

            # Apply user preferences to filter recommendations
            filtered_data = data.copy()

            # Recommending food based on users choice of meal time.
            if meal_pref == "lunch":
                filtered_data = filtered_data[filtered_data['Lunch'] == 1]
            elif meal_pref == "breakfast":
                filtered_data = filtered_data[filtered_data['Breakfast'] == 1]
            else:
                filtered_data = filtered_data[filtered_data['Dinner'] == 1]

            if vegetarian_pref.lower() == 'yes':
                filtered_data = filtered_data[filtered_data['Is_Vegetarian'] == 0]

            # Recommending food based on users health goal
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
            x_input = imputer.fit_transform(x_input)

            # Predict recommendation scores
            recommendation_scores = loaded_model.predict(x_input)

            # Combine recommendation scores with the filtered data
            filtered_data['Recommendation_Score'] = recommendation_scores

            # Sort the filtered data by recommendation scores
            recommended_items = filtered_data.sort_values(by='Recommendation_Score', ascending=False)

            # Get the top N recommended food items (N can be determined based on user preferences)
            top_n_recommendations = recommended_items.head(10)["Food_items"]

            return jsonify({"recommendations": top_n_recommendations.to_dict()})

        else:
            open_ai_key = os.environ.get("openai_key")

            llm_resto = OpenAI(temperature=0.6, openai_api_key=open_ai_key)
            prompt_template_resto = PromptTemplate(
                input_variables=['meal_pref', 'vegetarian_pref', 'calorie_limit', 'health_goal'],
                template="Diet Recommendation System:\n"
                         "I want you to recommend 10 {meal_pref} names specifically for diabetic patients"
                         "based on the following criteria:\n"
                         "Person maximum calorie limit: {calorie_limit}\n"
                         "Person's health goal: {health_goal}\n"
                         "Person is veg or non veg: {vegetarian_pref}\n"
            )

            chain_resto = LLMChain(llm=llm_resto, prompt=prompt_template_resto)
            input_data = {'meal_pref': meal_pref,
                          'vegetarian_pref': vegetarian_pref,
                          'calorie_limit': calorie_limit,
                          'health_goal': health_goal,
                          }
            results = chain_resto.run(input_data)

            pattern = re.compile(r'\d+\.\s(.+?)\n')

            # Find all matches using the pattern
            matches = re.findall(pattern, results)

            # Create a dictionary with indices as keys and food names as values
            food_dict = {index + 1: food_name for index, food_name in enumerate(matches)}

            return jsonify(food_dict)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
