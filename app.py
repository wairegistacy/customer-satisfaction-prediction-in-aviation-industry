from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from config import Config
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

model_path = 'random_forest_reg.pkl'
food_model_path = 'food_prediction_model.pkl'
inflight_services_model_path = 'inflight_services_model.pkl'
cleanliness_model_path = 'cleanliness_model.pkl'

model_name = joblib.load(model_path)
food_model_name = joblib.load(food_model_path)
inflight_services_model_name = joblib.load(inflight_services_model_path)
cleanliness_model_name = joblib.load(cleanliness_model_path)


df_columns = ['Gender','Customer Type','Age','Type of Travel','Class','Flight Distance','Departure Delay in Minutes']


variable_dict = {
    "Male": 1,
    "Female": 0,
    "Loyal Customer": 0,
    "disloyal Customer": 1,
    "Disloyal Customer": 1,
    "Business Travel": 0,
    "Personal Travel": 1,
    "Eco": 0,
    "Economy": 0,
    "Eco Plus": 1,
    "Economy Plus": 1,
    "Business": 2,
}

desired_column_order = ['Age', 'Class','Flight Distance','Departure Delay in Minutes', 'Gender','Customer Type', 'Type of Travel']

one_hot_columns = ['Age', 'Class', 'Flight Distance', 'Departure Delay in Minutes', 'Gender_Male', 'Customer Type_disloyal Customer', 'Type of Travel_Personal Travel']



@app.route("/", methods=["GET", 'POST'])
def home():
    """
    Home page route
    """
    if request.method == 'POST':
        message = request.form['message']
        return jsonify(your_message=message)
    return render_template("index.html")

@app.route("/analysis", methods=["GET", 'POST'])
def analysis():
    """
    Analysis route
    """
    try:
        if request.method == 'POST' and request.files['myfile']:
            departuredelay = int(request.form["departuredelay"])
            flightDistance = int(request.form['flightDistance'])
            df = pd.read_csv(request.files.get('myfile'))

            original_df = df.copy()

            df = df.drop(columns=['Name'])
            
            df['Class'] = df['Class'].map(variable_dict)
            df['Gender'] = df['Gender'].map(variable_dict)
            df['Customer Type'] = df['Customer Type'].map(variable_dict)
            df['Type of Travel'] = df['Type of Travel'].map(variable_dict)
            df['Departure Delay in Minutes'] = departuredelay
            df['Flight Distance'] = flightDistance

            print(df['Departure Delay in Minutes'])

            df = df[desired_column_order]

            # Replace the existing columns with the new list of columns
            df.columns = one_hot_columns

            prediction = model_name.predict(df)
            food_prediction = food_model_name.predict(df)
            inflight_services_prediction = inflight_services_model_name.predict(df)
            cleanliness_prediction = cleanliness_model_name.predict(df)

            print(df)


            prediction_list = list(prediction)
            food_prediction_list = list(food_prediction)
            inflight_services_prediction_list = list(inflight_services_prediction)
            cleanliness_prediction_list = list(cleanliness_prediction)

            # Count the occurrences of 'satisfied'
            count_satisfied = prediction_list.count('satisfied')
            count_dissatisfied = prediction_list.count('neutral or dissatisfied')

            # food prediction rows
            food_df = original_df.copy()
            food_df['Food rating'] = food_prediction_list
            food_df = food_df[food_df['Food rating'] < 4]
            food_df = food_df[['Name', 'Gender','Age', 'Class', 'Food rating']]

            is_food = food_df.empty


            # inflight services prediction rows
            inflight_services_df = original_df.copy()
            inflight_services_df['Inflight services'] = inflight_services_prediction_list
            inflight_services_df = inflight_services_df[inflight_services_df['Inflight services'] < 4]
            inflight_services_df = inflight_services_df[['Name', 'Gender','Age', 'Class', 'Inflight services']]

            is_inflight = inflight_services_df.empty


            # cleanliness prediction rows
            cleanliness_df = original_df.copy()
            cleanliness_df['Cleanliness'] = cleanliness_prediction_list
            cleanliness_df = cleanliness_df[cleanliness_df['Cleanliness'] < 4]
            cleanliness_df = cleanliness_df[['Name', 'Gender','Age', 'Class', 'Cleanliness']]

            is_cleanliness = cleanliness_df.empty

            
        return render_template("analysis.html", count_satisfied=count_satisfied, count_dissatisfied=count_dissatisfied, 
                            total_passengers = len(prediction_list), 
                            is_food=is_food,
                            food_df=food_df.to_html(classes='table'),
                            is_inflight = is_inflight,
                                inflight_services_df=inflight_services_df.to_html(classes='table'),
                                is_cleanliness=is_cleanliness,
                                cleanliness_df=cleanliness_df.to_html(classes='table'), )
    except Exception as e:
        # Handle exceptions and display an error message
        error_message = f"An error occurred: {str(e)}"
        return render_template("error.html", error_message=error_message)


@app.route('/message', methods=['POST'])
def message():
    """
    Message route
    """
    message = request.json.get("message")
    return jsonify(your_message=message)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")