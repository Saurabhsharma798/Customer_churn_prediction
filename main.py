from src.pipeline import get_pipeline
from src.model_trainer import train_model
from src.utils import load_data,load_model,save_model
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def main():
    #load raw data
    data_path=os.path.join('data','data.csv')
    df=load_data(data_path)


    #split into features and labels
    X=df.drop(columns=['CustomerID','Churn']) 
    y=df['Churn']
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=11)

    train_data = pd.concat([X_train, y_train], axis=1)
    train_data = train_data.dropna(subset=[y_train.name])
    X_train = train_data.drop(columns=[y_train.name])
    y_train = train_data[y_train.name]

    #get pipeline
    pipeline=get_pipeline()


    #train the model
    model=train_model(pipeline,X_train,y_train)

    #save model to artifacts 
    os.makedirs('artifacts',exist_ok=True)
    model_path=os.path.join('artifacts','model.pkl')
    save_model(model,model_path)

    print(f'model training completed saved at {model_path}')


if __name__=='__main__':
    main()