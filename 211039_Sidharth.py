

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
import os

# Get the directory path by removing the script filename from the path
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the file names relative to the script location
file1_name = 'sample_input.csv'
file2_name = 'sample_close.txt'
file3_name = 'model_LSTM.k1'

# Construct the paths to the files using the script directory
file1_path = os.path.join(script_dir, file1_name)
file2_path = os.path.join(script_dir, file2_name)
file3_path = os.path.join(script_dir, file3_name)



def predict_func(data):
    """
    Modify this function to predict closing prices for next 2 samples.
    Take care of null values in the sample_input.csv file which are listed as NAN in the dataframe passed to you 
    Args:
        data (pandas Dataframe): contains the 50 continuous time series values for a stock index

    Returns:
        list (2 values): your prediction for closing price of next 2 samples
    """
    model = tf.keras.models.load_model(file3_path)

    # model.summary()

    sample_input = pd.read_csv(file1_path,index_col = 0, header =0)

    data = pd.DataFrame(sample_input)
    # data

    data = data[['Close']]

    data.index = pd.to_datetime(data.index, format = '%Y-%m-%d')

    data = data.interpolate(method = 'spline',order = 3)
    df1=data
    
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    
    test_data=df1
    
    l_test_data=len(df1)-50
    
    x_input=test_data[l_test_data:].reshape(1,-1)
    
    
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    
    # demonstrate prediction for next 2 days



    lst_output=[]
    n_steps=50
    i=0
    while(i<2):
       
        if(len(temp_input)>50):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    

    lst_output=scaler.inverse_transform(lst_output)

    return lst_output


# In[3]:


def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv(file1_path)

    actual_close = np.loadtxt(file2_path)

    pred_close = predict_func(df)

    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = np.mean(np.square(actual_close-pred_close))
    pred_close=np.ravel(pred_close)
    
    pred_prev = [df['Close'].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close

    actual_prev = [df['Close'].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr)-np.array(pred_prev)
    actual_dir = np.array(actual_curr)-np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir*actual_dir)>0)*100

    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')



# In[4]:


if __name__ == "__main__":
    evaluate()


# In[ ]:




