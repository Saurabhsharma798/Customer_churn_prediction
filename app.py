from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Annotated,Literal
from predict_pipeline import CustomData,PredictPipeline
import pandas as pd


app=FastAPI()


class UserInput(BaseModel):
    Age:Annotated[int,Field(...,gt=0,lt=120)]
    Gender:Annotated[Literal['Male','Female'],Field(...)]
    Tenure:Annotated[int,Field(...)]
    Usage_Frequency:Annotated[int,Field(...)]
    Support_Calls:Annotated[int,Field(...)]
    Payment_Delay:Annotated[int,Field(...)]
    Subscription_Type:Annotated[Literal['Premium','Standard','Basic'],Field(...)]
    Contract_Length:Annotated[Literal['Monthly','Quarterly','Annual'],Field(...)]
    Total_Spend:Annotated[int,Field(...)]
    Last_Interaction:Annotated[int,Field(...)]






app.get('/')
def home():
    return render_template('index.html')


app.get('/predict')
def predict_form():
    return render_template('home.html')


app.post('/predict')
def predict_data(data:UserInput):
    
    data_df=pd.DataFrame(data)
    predict_pipeline=PredictPipeline()
    result=predict_pipeline.predict(data_df)