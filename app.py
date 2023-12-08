from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from keras.models import load_model
import librosa
import os
app = Flask(__name__)
model = load_model('D:\code\model.h5')
@ app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods = ['POST'])
def predict():   
    if request.method == 'POST':
        f = request.files["file"]
        counter=0
        X_test = pd.DataFrame(columns=['feature'])
        #for index,path in enumerate(f):
        X, sample_rate = librosa.load(f
                                      , res_type='kaiser_fast'
                                      ,duration=2.5
                                      ,sr=44100
                                      ,offset=0.5
                                     )
        sample_rate = np.array(sample_rate)

        # mean as the feature. Could do min and max etc as well. 
        mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                            sr=sample_rate, 
                                            n_mfcc=13),axis=0)
        X_test.loc[counter]=[mfccs]
            #counter=counter+1 
        X_test = pd.DataFrame(X_test['feature'].values.tolist())
        #print(X_test)
        '''X_test= X_test.fillna(0)
        mean= np.mean(X_test, axis =0)
        std = np.std(X_test,axis=0)
        X_test= (X_test - mean)/std
        X_test = np.array(X_test)
        X_test= np.expand_dims(X_test, axis=2)'''
        prediction = model.predict([X_test])
        prediction=np.argmax(prediction,axis=1)
        output = prediction[0]
        if output == 0:
            return render_template('index.html', prediction = 'ANGER')
        elif output == 1:
            return render_template('index.html', prediction = 'DISGUST')
        elif output == 2:
            return render_template('index.html', prediction='FEAR')        
        elif output == 3:
            return render_template('index.html', prediction='HAPPINESS')   
        elif output == 4:
            return render_template('index.html', prediction='NEUTRAL')   
        elif output == 5:
            return render_template('index.html', prediction='SAD')
        #elif output == 6:
            #return render_template('index.html', prediction='SURPRISED')

if __name__ == '__main__':
  app.run()
