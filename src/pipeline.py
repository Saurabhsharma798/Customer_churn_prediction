import pandas as pd 
import numpy as np 
import sklearn
from sklearn.compose import ColumnTransformer
from  sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def get_pipeline():
        
    num_cols=['Age','Tenure','Usage Frequency','Support Calls','Payment Delay','Total Spend','Last Interaction']
    cat_cols_onehot_encode=['Subscription Type','Contract Length']
    cat_cols_label_encode=['Gender']


    label_pipeline=Pipeline(
        steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('encoding',OrdinalEncoder())
        ]
    )

    onehot_pipeline=Pipeline(
        steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('encoding',OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    num_pipeline=Pipeline(
        steps=[
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())
        ]
    )

    preprocessor=ColumnTransformer(
        transformers=[
            ('label',label_pipeline,cat_cols_label_encode),
            ('onehot',onehot_pipeline,cat_cols_onehot_encode),
            ('num',num_pipeline,num_cols)
        ]
    )


    pipeline=Pipeline(
        steps=[
            ('preprocessor',preprocessor),
            ('model',RandomForestClassifier())

        ]
    )

    return pipeline
