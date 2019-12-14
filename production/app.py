###import dependecies
import os
import requests
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify, render_template, redirect
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import datetime

#Create the Flask API
app = Flask(__name__)

# Create a directory in a known location to save files to.
#app.config['UPLOAD'] = '/Users/snowwhite/Documents/Github/Project Osaka/production/uploads/'
app.config['UPLOAD'] = os.path.join(os.getcwd(),'production/uploads')
#app.config['UPLOAD'] = os.path.join(os.path.realpath('app.py'),'/uploads')

#Load the data from upload
@app.route('/', methods=['GET', 'POST'])

#This is basically renders the upload html page, and provides a User Interface for triggering the Prediciton function
def upload_form():
    if request.method == 'POST':
        if request.files:
            #os.unlink(app.config['UPLOAD'])
            upload = request.files["file"]
            upload.save(os.path.join(app.config['UPLOAD'], 'data.csv'))
            return render_template('upload.html', message= 'Your file has been saved!', forward_message='Click here to Predict')
    return render_template('upload.html', forward_message='Please upload your data First!')

#Define the Prediction function, 
@app.route('/forward', methods=['GET', 'POST'])
def predict():  
    #Read the data - in the proper format
    def load_input_file():
        inputs2 = pd.read_csv(os.path.join(app.config['UPLOAD'], 'data.csv'), parse_dates=['Date'])
        inputs2.set_index('Date', inplace=True) #Date added
        inputs2 = inputs2.dropna()
        cols = inputs2.columns
        return inputs2, cols

    def back_scale(y_train, y_hat):
        scaler = MinMaxScaler()
        y_train_scaled = pd.DataFrame(scaler.fit_transform(y_train.reshape(-1,1)))
        y_hat_scaled = pd.DataFrame(scaler.inverse_transform(y_hat.reshape(-1,1)))
        return y_hat_scaled

    #Reshape the date for LSTM use, As we use more than one Features (LSTM Network - usually takes 1 feature to work with - the Tensors need to reshaped
    def reshape (dataset, y_seq): 
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
            
        out = np.array(dataset[y_seq])
        out = out.reshape((len(out), 1))
        seq_list.append(out)
        
        dataset_shaped = np.hstack((seq_list))
        #horizontally stack columns
        return dataset_shaped

    # split a multivariate sequence into samples/batches, This is an LSTM related slip to make sure that the Tensors have enough dimensions for the Network to process
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
    
    # this function is basically do the following: first get the input how many desy wee need to predict (default is 30 days) then, it gonna make the predictions one by one. 
    #The reason for this that the LSTM network is unique ina  way that uses multiple features (trained with 4 features) in order to make predicitions on a datapoint basis, 
    #We have to make some shortcuts, as the LSTM network cannot handle the fact that there are no features in the future. 
    #Hence, the features are engineered as a way to project them (using mean) to the future, and therefore providing enough datapoints for the Network to work with.  
    def forecast(n_prediction, model,dataset_shaped,n_steps):
        look_back = 120 #so using 60 days wort of data to calculate the next mean
        prediction_list = dataset_shaped[-look_back:][::-1]
        result = []
    
        for _ in range(n_prediction):
            z = prediction_list[-look_back:]
            x, y = split(z, n_steps)
            out = model.predict(x)[0][0]
            features_mean= z.mean(axis=0)  #Projecting the features to the future
            prediction_list = np.vstack((prediction_list,features_mean))
            result.append(out)
        return result
    
#All the magic will happen from here

#Variables
    if not os.path.exists(os.path.join(app.config['UPLOAD'], 'data.csv')):
        return print("Please upload your file first")
 
    n_steps = 5 #How many days to use to predict, required for the model
    n_prediction = 30 #How many day to predict ahead in the future
    
    inputs2, cols = load_input_file() # First of all we read the data in in the right format
    #input3 = scale_it(scaler, inputs2, cols) # the data what we read, needs to be scaled with MinMaxScaler
    dataset_shaped = reshape(inputs2, 'Last') #'Last' #format the scaled data for LSTM use (part 1, where are formatting the columns into arrays)
    #np.save('shaped', dataset_shaped)
    X, y = split(dataset_shaped, n_steps) #Also need to format the Scaled data into 4D tensors
    #np.save('X_train', X)
    #np.save('y_train', y)
    n_features = X.shape[2] # This wariable is required for the model, gives the feature dimension
    
 #Define the LSTM architecture, the weights are loaded seperatelly, but I've decided to keep the architecture part of the main code for more transparency
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
    #LSTM_model.load_weights('/Users/snowwhite/Documents/Github/Project Osaka/production/weights_3.h5')
    LSTM_model.load_weights(os.path.join(os.getcwd(),'production/weights_3.h5'))
    adam=tf.keras.optimizers.Adam(lr=0.003711, epsilon=None, amsgrad=True, decay=0)
    LSTM_model.compile(loss ='mean_squared_error', optimizer=adam, metrics=['accuracy'])   

    final_forecast = forecast(n_prediction, LSTM_model,dataset_shaped,n_steps)
    final_scaled = back_scale(y, np.array(final_forecast))
    #final_scaled = pd.DataFrame(final_scaled)
    def build_graph(final_scaled, inputs2, n_prediction):
        #dates = np.array([datetime.datetime(inputs2.datetimeIndex(inputs2.index[0]).year,inputs2.datetimeIndex(inputs2.index[0]).month, inputs2.datetimeIndex(inputs2.index[0]).day, i, 0) for i in range(24)])
        dataset = pd.DataFrame(final_scaled)
        dataset.index = [x for x in pd.date_range(inputs2.index.values[0] + pd.offsets.Day(1),periods=n_prediction)]
        img = BytesIO()
        plt.figure(figsize=(15,8))
        plt.xlabel('Dates')
        plt.ylabel('Closing Prices')
        plt.grid(True)
        plt.plot(inputs2['Last'], 'blue', label='Historical Data')
        plt.plot(dataset, 'red', label='Forecast')
        plt.legend()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return 'data:image/png;base64,{}'.format(graph_url)
    
    plot_url = build_graph(final_scaled, inputs2, n_prediction)

    #Return the prediction to the user
    return render_template('result.html', prediction = final_scaled, plot_url = plot_url)

#Start the Flask application
if __name__ == '__main__':
    app.run(port=5000, debug=False)
