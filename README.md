# Nutri-Lytik : Diet Recommendation System

## Table of Contents
- [Overview](#overview)
- [Screenshots](#screenshots)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the API](#running-the-api)
  - [Making Recommendations](#making-recommendations)

## Overview

Nutri-Lytik is a diet recommendation system designed to help users make healthier food choices based on their dietary preferences and health goals. The system uses machine learning to provide personalized meal recommendations.

## Screenshots
<img width="864" alt="image" src="https://github.com/VaidikV/NutriLytik-Diet-Recommendation-System/assets/63895478/160df18f-7afb-4788-9e3c-75a7137b6e2c">

## Features

- Personalized meal recommendations for breakfast, lunch, and dinner.
- Support for vegetarian and non-vegetarian diets.
- Consideration of calorie limits and health goals (e.g., weight loss or weight gain).
- API for integrating the recommendation system into other applications.

## Getting Started

### Prerequisites

- Python 3.x
- Flask (for the API)
- Scikit-learn (for machine learning)
- Pandas (for data handling)
- Joblib (for model serialization)

### Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/nutri-lytik.git

2. Install the required Python packages:
   ```sh
   pip insall -r requirements.txt

## Usage

### Running the API
Start the Nutri-Lytik API by running the following command in the project directory:
```sh
python app.py
```
### Making Recommendations
You can make dietary recommendations by sending a POST request to the /recommend endpoint of the API. The request should include user preferences in JSON format.

Example request;
```sh
{
  "meal_preference": "breakfast",
  "vegetarian_pref": "no",
  "max_calories": 1500,
  "health_goal": "healthy"
}
```
