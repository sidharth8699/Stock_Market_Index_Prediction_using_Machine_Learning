# Stock Index Prediction

This project aims to predict the closing prices of a stock index for the next 2 days based on historical data. The prediction is made using a pre-trained model and involves data preprocessing and scaling.

## Dependencies

The following libraries are required to run the code:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import numpy as np
import pandas as pd
import datetime
import sklearn
from numpy import array
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM


## Instructions

1. Data Preparation:
   - Ensure that you have the 'sample_input.csv' file, which contains the historical data of the stock index.
   - Make sure the 'sample_close.txt' file is available, containing the actual closing prices for evaluation.

2. Pre-Trained Model:
   - The code assumes that the pre-trained model 'model_LSTM.k1' is available in the current directory.
   - Ensure that the model file is correctly named and saved in the same directory as the code.

3. Running the Code:
   - Execute the 'evaluate()' function in the code to perform the stock index prediction.
   - The code will load the input data, preprocess it, and make predictions for the next 2 days' closing prices.
   - The mean square error and directional accuracy will be calculated and displayed.

Note: It is important to keep the file names, file locations, and dependencies intact for the code to run without errors.

