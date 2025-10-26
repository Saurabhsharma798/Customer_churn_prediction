from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Annotated,Literal
from src.predict_pipeline import CustomData,PredictPipeline
from recommender import get_recommendation
from fastapi.middleware.cors import CORSMiddleware


app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://<your-ec2-ip>:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



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




@app.post('/predict')
def predict_data(data:UserInput):
    
    custom_data = CustomData(
            Age=data.Age,
            Gender=data.Gender,
            Tenure=data.Tenure,
            Usage_Frequency=data.Usage_Frequency,
            Support_Calls=data.Support_Calls,
            Payment_Delay=data.Payment_Delay,
            Subscription_Type=data.Subscription_Type,
            Contract_Length=data.Contract_Length,
            Total_Spend=data.Total_Spend,
            Last_Interaction=data.Last_Interaction
        )
    
    final_data=custom_data.get_data_as_data_frame()

    pipeline=PredictPipeline()

    prediction=pipeline.predict(final_data)
    recommender=get_recommendation(custom_data,str(prediction))

    return {'prediction':prediction[0],
            "recommendation":recommender.strip()}
