from flask import Flask, jsonify, request
import numpy as np
#from sklearn.externals import joblib
import joblib

#Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# to display all column of datapoints
pd.set_option('display.max_columns', None)

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


X=pd.read_csv('final_test.csv')

X = X[[ 'duration' ,'src_bytes','dst_bytes','wrong_fragment',
'urgent','hot','num_failed_logins','num_compromised','num_root','num_file_creations','num_shells',
'num_access_files','num_outbound_cmds','count','srv_count','serror_rate','srv_serror_rate',
'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','protocol_type', 'service' , 'flag']]

def data_preprocessing(X):
    
    """
    Normal=0 ,Probing=1,DOS=2,U2R=3,R2L=4
    
    """
    
    continuous_features=["duration","src_bytes",
    "dst_bytes","wrong_fragment","urgent","hot","num_failed_logins","num_compromised","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"]

    
    #LabelEncoder() function Encode target labels with value between 0 and n_classes-1
    le = LabelEncoder()

    # protocol
    X_protocol_type = le.fit_transform(X['protocol_type'].values)
    X_service = le.fit_transform(X['service'].values)
    X_flag = le.fit_transform(X['flag'].values)

    X_main = X[continuous_features]
    
    X_main['protocol_type'] = X_protocol_type
    X_main['service'] = X_service
    X_main['flag'] = X_flag
    
    
    return X_main


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/index')
def index():
    return flask.render_template('index.html')

binary_model = joblib.load('RandomForestClassifier_over_sampling_SMOTETomek.joblib')
multiclass_model = joblib.load('XGBClassifier_multiclass_without_sampling.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    
    
    binary_model_pred = binary_model.predict(X)
    
    multiclass_model_pred = multiclass_model.predict(X)
    
    i = [0 if j == 0 else 1 for j in binary_model_pred]
    j=int(i[0])
    
    k = [l for l in multiclass_model_pred]   
    l=int(k[0]) 
    
    if j == 1:
        
        return jsonify({'The Result in Instrusion Singal': l })
    
    else:
        
        return jsonify({'The Result in Non Instrusion Singal': j })
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
