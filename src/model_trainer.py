
def train_model(pipeline,X,y):
    
    #Build and train model
    model=pipeline.fit(X,y)
    return model