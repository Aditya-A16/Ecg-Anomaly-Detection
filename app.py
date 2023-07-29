from flask import Flask,request, render_template
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mae
import numpy as np


app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    uploaded_file = request.files["csvFile"]
    
    uploaded_file.save("sample.csv")  

    df = pd.read_csv('sample.csv')
    data = df.iloc[:1, :].to_numpy()
    data = np.divide(data, 13.682)
    data = data.reshape(1, -1)
   
    model = load_model('autoencoder_Model')
    reconstructions = model(data)
    loss = mae(reconstructions, data)
    
    
    result = loss.numpy() < 0.03293371
    

    
    if result:
      data = "Your heart ECG signal looks fine"
    else:
      data = 'Danger!, Please go to the doctor'
    return render_template("final.html", data = data)


if __name__ == '__main__':
  app.run()
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=8080)	
    