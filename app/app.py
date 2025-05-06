from src.utils import load_model
import pandas as pd
import os

def make_predictions(user_data:dict):
    #input to dataframe
    data=pd.DataFrame([user_data])

    #load model
    model_path=os.path.join('artifacts','model.pkl')
    model=load_model(model_path)

    # make predictions
    prediction=model.predict(data)
    return prediction[0]


