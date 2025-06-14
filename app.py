from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipe_line.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/clsJobData', methods=['GET', 'POST'])
def cls_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            title = request.form.get('title'),
            location = request.form.get('location'),
            description = request.form.get('description'),
            function = request.form.get('function'),
            industry = request.form.get('industry')


        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        # print("Before Prediction")
        predict_pipeline = PredictPipeline()
        # print("Mid Prediction")
        dict =  {0: 'bereichsleiter',
                 1: 'director_business_unit_leader',
                 2: 'manager_team_leader',
                 3: 'managing_director_small_medium_company',
                 4: 'senior_specialist_or_project_manager',
                 5: 'specialist'}
        results = predict_pipeline.predict(pred_df)
        predicted = dict[results[0]]
        return render_template('home.html', results=predicted )


if __name__ == "__main__":
    app.run(host="0.0.0.0")