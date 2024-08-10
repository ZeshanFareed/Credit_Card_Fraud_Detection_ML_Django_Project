from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def user(request):
    return render(request, 'userinput.html')

def viewdata(request):
    # Load the dataset
    df = pd.read_csv("C:/Users/PMLS/Documents/ML/ML Algorithms/creditcard.csv")
    
    legit = df[df.Class == 0]
    fraud = df[df.Class == 1]
    
    legit_sample = legit.sample(n=492)
    
    new_datasets = pd.concat([legit_sample, fraud], axis=0)
    
    X = new_datasets.drop(columns='Class', axis=1)
    y = new_datasets['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    LR = LogisticRegression()
    LR.fit(X_train, y_train)
    
    # Retrieve input data from the GET request
    new_data = [
        float(request.GET['Time']),
        float(request.GET['V1']),
        float(request.GET['V2']),
        float(request.GET['V3']),
        float(request.GET['V4']),
        float(request.GET['V5']),
        float(request.GET['V6']),
        float(request.GET['V7']),
        float(request.GET['V8']),
        float(request.GET['V9']),
        float(request.GET['V10']),
        float(request.GET['V11']),
        float(request.GET['V12']),
        float(request.GET['V13']),
        float(request.GET['V14']),
        float(request.GET['V15']),
        float(request.GET['V16']),
        float(request.GET['V17']),
        float(request.GET['V18']),
        float(request.GET['V19']),
        float(request.GET['V20']),
        float(request.GET['V21']),
        float(request.GET['V22']),
        float(request.GET['V23']),
        float(request.GET['V24']),
        float(request.GET['V25']),
        float(request.GET['V26']),
        float(request.GET['V27']),
        float(request.GET['V28']),
        float(request.GET['Amount'])
    ]
    
    new_data_df = pd.DataFrame([new_data])
    
    # Make prediction (reshape input data if needed)
    y_pred = LR.predict(new_data_df)
    
    data = {
        'prediction': y_pred[0],
        'message': '',
    }
    
    if y_pred[0] == 0:
        data['message'] = 'The Predicted Credit Card is Legit'
    else:
        data['message'] = 'The Predicted Credit Card is Fraud'
        
    return render(request, 'viewdata.html', data)
