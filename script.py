import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from numpy import loadtxt
from tensorflow.keras.models import load_model

app = Flask(__name__) #Initialize the flask App
#model = pickle.load(open('latest_model.pkl', 'rb'))
model = load_model('car_predictor.h5')
encoder = pickle.load(open('one_hot_encoder_neural_net.pkl','rb'))
scaler = pickle.load(open('scaler_neural_net.pkl','rb'))
normalizer = pickle.load(open('normalizer_neural_net.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/overview')
def overview():
    return render_template('overview.html')
    
@app.route('/dashboard-1')
def dashboard_1():
    return render_template('dashboard_1.html')
    
@app.route('/dashboard-2')
def dashboard_2():
    return render_template('dashboard_2.html')
    
@app.route('/dashboard-3')
def dashboard_3():
    return render_template('dashboard_3.html')
    
@app.route('/dashboard-4')
def dashboard_4():
    return render_template('dashboard_4.html')
    
@app.route('/dashboard-5')
def dashboard_5():
    return render_template('dashboard_5.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    brand = final[0]
    model_name = final[1]
    gear = final[2]
    year = final[3]
    milerate = final[5]
    final_value = np.append(encoder.transform([final[:3]]).toarray(), scaler.transform([final[3:]]))

    
    
    y_pred = model.predict(np.array([final_value]))
    
    prediction = int(normalizer.inverse_transform(y_pred))

    return render_template('index.html', prediction=prediction, brand=brand, model_name=model_name, gear=gear, year=year, milerate=milerate)

if __name__ == "__main__":
    app.run(debug=True)
