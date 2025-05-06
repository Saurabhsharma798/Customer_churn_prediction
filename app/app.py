from utils.utils import load_model,load_data
import pandas as pd

def make_predictions(user_data):
    data=pd.DataFrame([user_data])
    model=load_model()
    prediction=model.predict(data)
    return prediction


