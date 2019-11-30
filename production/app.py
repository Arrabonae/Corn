###import dependecies
import os
import requests
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify, render_template, redirect

#Create the Flask API
app = Flask(__name__)

# Create a directory in a known location to save files to.
#uploads_dir = os.path.join(app.instance_path, 'uploads')
app.config['UPLOAD'] = '/Users/snowwhite/Documents/Github/Project Osaka/production/uploads/'

#Load the data from upload
@app.route('/', methods=['GET', 'POST'])

def upload_form():
    if request.method == 'POST':
        if request.files:
            upload = request.files["file"]
            upload.save(os.path.join(app.config['UPLOAD'], 'data.csv'))
            return render_template('upload.html', message= 'Your file has been saved!', forward_message='Click here to Predict')
    return render_template('upload.html', forward_message='Please upload your data First!')

#Define the prediction function
@app.route('/forward', methods=['GET', 'POST'])
def predict():  
    #Read the data - in the proper format
    def read_it():
        inputs2 = pd.read_csv(os.path.join(app.config['UPLOAD'], 'data.csv'))
        cols = inputs2.columns
        return inputs2, cols

    #MinMaxScaler
    def scale_it(scaler,inputs2, cols):
        #scaler = MinMaxScaler()
        inputs3 = pd.DataFrame(scaler.fit_transform(inputs2))
        inputs3.columns = cols
        return inputs3

    #Reshape the date for LSTM use
    def reshape (dataset, y_seq): #This variable was only required during development phase of the modeling, not required for production 
        #Get the column headers
        cols2 = np.array(dataset.columns)
        cols2 = np.delete(cols2, np.where(cols2 == y_seq))
        seq_list = []
        
        #split the dataset into vectors and convert into a 4D tensor
        for i in range(len(cols2)):
            seq = np.array(dataset[cols2[i]])
            #Convert to [row, columns] structure
            seq = seq.reshape((len(seq), 1))
            seq_list.append(seq)
            
        #This part was only required to develop the model, under production there is no 'y' variable
        out = np.array(dataset[y_seq])
        out = out.reshape((len(out), 1))
        seq_list.append(out)
        
        dataset_shaped = np.hstack((seq_list))
        #horizontally stack columns
        return dataset_shaped

    # split a multivariate sequence into samples/batches
    def split(sequences, n_steps):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X) , np.array(y)

    #Define the model structure. We could have import the architecture from a seperate file, 
    #but that file does not support dynamic varaibles like, n_steps and n_features
#All the magic will happen from here
    #scaler = MinMaxScaler()
    n_steps = 5 # How many days to predict, required for the model
    inputs2, cols = read_it() # First of all we read the data in in the right format
    #input3 = scale_it(scaler, inputs2, cols) # the data what we read, needs to be scaled with MinMaxScaler
    dataset_shaped = reshape(inputs2, 'Price') #format the scaled data for LSTm use (part 1, where are formatting the columns into arrays)
    X, y = split(dataset_shaped, n_steps) #Also need to format the Scaled data into 4D tensors
    n_features = X.shape[2] # This wariable is required for the model, gives the feature dimension

    #Perform predictions with pre-trained model

    LSTM_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(344, return_sequences=True, input_shape=(n_steps, n_features)),
        tf.keras.layers.Dropout(0.1677),
        tf.keras.layers.LSTM(214,
                  return_sequences=True),
        tf.keras.layers.Dropout(0.2741),
        tf.keras.layers.LSTM(287,
                  return_sequences=True),
        tf.keras.layers.Dropout(0.08178),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1),
        ])
    #Load the pre-trained weights into the model
    LSTM_model.load_weights('/Users/snowwhite/Documents/Github/Project Osaka/production/weights.h5')
    adam=tf.keras.optimizers.Adam(lr=0.003711, epsilon=None, amsgrad=True, decay=0)
    LSTM_model.compile(loss ='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    #prediction = LSTM_model.predict(X)[0][0]


    ##Scale the results BACK to original format

    #scaled_pred = scaler.inverse_transform(prediction)
    def predi(num_prediction, model):
        prediction_list = dataset_shaped[-n_steps:][::-1]
        result = []
    
        for _ in range(num_prediction):
            z = prediction_list[-n_steps:]
            x, y = split(z, n_steps)
            out = model.predict(x)[0][0]
            features_mean= z.mean(axis=0)
            prediction_list = np.vstack((prediction_list,features_mean))
        
            result.append(out)
        return result

    num_prediction = 30
    forecast = predi(num_prediction, LSTM_model)

    #Return the prediction to the user
    return render_template('result.html', prediction = forecast)

#Start the Flask application
if __name__ == '__main__':
    app.run(port=5000, debug=False)
