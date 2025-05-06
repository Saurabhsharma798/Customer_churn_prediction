import pandas as pd
import os
from src.utils import load_model

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        model_path=os.path.join('artifacts','model.pkl')
        model=load_model(model_path)
        pred=model.predict(features)
        return pred


class CustomData:
    def __init__(self,Age:float,Gender:str,Tenure:float,Usage_Frequency:float, Support_Calls:float,Payment_Delay:float,Subscription_Type:str,Contract_Length:str,Total_Spend:float,Last_Interaction:float):
        self.Age=Age
        self.Gender=Gender
        self.Tenure=Tenure
        self.Usage_Frequency=Usage_Frequency
        self.Support_Calls=Support_Calls
        self.Payment_Delay=Payment_Delay
        self.Subscription_Type=Subscription_Type
        self.Contract_Length=Contract_Length
        self.Total_Spend=Total_Spend
        self.Last_Interaction=Last_Interaction

    def get_data_as_data_frame(self):
        custom_data_input_dict={
            "Age":[self.Age],
            "Gender":[self.Gender],
            "Tenure":[self.Tenure],
            "Usage Frequency":[self.Usage_Frequency],
            "Support Calls":[self.Support_Calls],
            "Payment Delay":[self.Payment_Delay],
            "Subscription Type":[self.Subscription_Type],
            "Contract Length":[self.Contract_Length],
            "Total Spend":[self.Total_Spend],
            "Last Interaction":[self.Last_Interaction],
        }
        return pd.DataFrame(custom_data_input_dict)
    

