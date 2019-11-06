
##import dependecies
import os
import requests
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify


##Load the pretrained network

#Load the Structure
model = tf.keras.models.model_from_file() #Still need to be defined

#Load the weights
model.load_weights() #Still need to be defined

##Create the Flask API

#Define the Flask application
app = Flask(__name__)