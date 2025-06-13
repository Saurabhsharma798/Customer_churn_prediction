import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key=os.getenv('grok_api_key')

def get_recommendation(customer_data:dict,prediction_result:str)->str:
    prompt=f"""Customer Details:{customer_data}
    churn prediction:{prediction_result}
    
    As a customer retention assistant,give a short recommendation in **3 bullet points** (max 50 words each) on how to reduce churn for this customer. Be clear and specific."""


    headers={
        "Authorization":f"Bearer {api_key}",
        "Content-Type":"application/json"
    }


    body={
        "model":"llama3-8b-8192",
        "messages":[
            {"role":"system","content":"you are an expert in customer retention."},
            {"role":"user","content":prompt}
        ]
    }

    response=requests.post("https://api.groq.com/openai/v1/chat/completions",headers=headers,json=body)

    response_data=response.json()
    try:
        
        return response_data['choices'][0]['message']['content']
    
    except Exception as e:
        print('groq api error:',response.status_code,response.text)
        return "sorry, no recommendation available right now."