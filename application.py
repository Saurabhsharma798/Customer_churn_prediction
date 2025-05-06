from flask import Flask,request,render_template
import pandas as pd
import numpy as np
from src.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

#route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_data():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Age=request.form.get('Age'),
            Gender=request.form.get('Gender'),
            Tenure=request.form.get('Tenure'),
            Usage_Frequency=request.form.get('Usage_Frequency'),
            Support_Calls=request.form.get('Support_Calls'),
            Payment_Delay=request.form.get('Payment_Delay'),
            Subscription_Type=request.form.get('Subscription_Type'),
            Contract_Length=request.form.get('Contract_Length'),
            Total_Spend=request.form.get('Total_Spend'),
            Last_Interaction=request.form.get('Last_Interaction'),
        )

        pred_data=data.get_data_as_data_frame()

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_data)

        return render_template('home.html',results=results[0])
    

if __name__=='__main__':
    app.run(host='0.0.0.0')