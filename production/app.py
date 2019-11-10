###import dependecies
import os
import requests
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify


###Load the pretrained network

##Load the Structure
with open('model_config.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)

##Load the weights
model.load_weights('weights.h5')


###Create the Flask API

##Define the Flask application
app = Flask(__name__)

##Defining the data reashape function

#Load the data from upload
@app.route("/api/v1/<string:data>", methods=["POST"])

#Variables
n_steps = 10 # How many days to predict

#Format the Data
df = read_it(data)
df2 = scale_it(inputs2, cols)
dataset = reshape(df2)
X = split(dataset, n_steps)

#Variables cont. 
n_features = X.shape[2]

def read_it(data):
    inputs2 = pd.read_csv("uploads/" + data)
    cols = inputs2.columns
    return inputs2, cols

#MinMaxScaler
def scale_it(inputs2, cols):
    scaler = MinMaxScaler()
    inputs3 = pd.DataFrame(scaler.fit_transform(inputs2))
    inputs3.columns = cols
    return inputs3

#Reshape for 
def reshape (dataset): #,y_seq): This variable was only required during development phase of the modeling, not required for production 
    #Get the column headers
    cols2 = np.array(dataset.columns)
    #cols = np.delete(cols, np.where(cols == y_seq))
    seq_list = []
    
    #split the dataset into vectors and convert into a 4D tensor
    for i in range(len(cols2)):
        seq = np.array(dataset[cols2[i]])
        #Convert to [row, columns] structure
        seq = seq.reshape((len(seq), 1))
        seq_list.append(seq)
        
    #This part was only required to develop the model, under production there is no 'y' variable
    #out = np.array(dataset[y_seq])
    #out = out.reshape((len(out), 1))
    #seq_list.append(out)
    
    dataset_shaped = np.hstack((seq_list))
    #horizontally stack columns
    return dataset_shaped

    # split a multivariate sequence into samples
def split(sequences, n_steps):
    X = list()  # updated to production ## y = list(),
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x = sequences[i:end_ix, :-1] # seq_y= , sequences[end_ix-1, -1] (Updated to production)
        X.append(seq_x)
        #y.append(seq_y)
    return np.array(X) #, array(y)

#Perform predictions with pre-trained model
prediction = model.predict(X)
scaler = MinMaxScaler()
scaled_pred = scaler.inverse_transform(prediction)

#Return the prediction to the user
jsonify(scaled_pred)

#Start the Flask application
app.run(port=5000, debug=False)