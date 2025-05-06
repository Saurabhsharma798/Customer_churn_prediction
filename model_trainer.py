from pipeline.pipeline  import get_pipeline
from utils.utils import load_data,save_model    
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def model_training():
    
    #Load data
    df=load_data('data/data.csv')
    X=df.drop(columns=['CustomerId','Churn'])
    y=df['Churn']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=11)


    #Build and train pipeline
    pipeline=get_pipeline()
    pipeline.fit(X_train,y_train)


    #save trained pipeline
    save_model(pipeline,'artifacts/model.pkl')


if __name__=='__main__':
    model_training()