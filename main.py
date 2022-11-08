import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import pandas_datareader as web # to read data from web
import numpy as np
import os
#import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, LSTM


# Proof of Concept
# Import TX Instruments stock data using pandas_datareader from Yahoo Finance
txn_data= web.DataReader('TXN',data_source="yahoo",start='2014-01-01',end='2022-02-02')
print(txn_data.head())
print(txn_data.shape)

print('Test')

# Plot the history of the companies closing price in $
plt.figure(figsize=(20,10))
sns.lineplot(x = txn_data.index, y = txn_data['Close'])
plt.xlabel('Year', fontsize=22)
plt.ylabel('Closing Price (USD)',fontsize=22)
plt.show()

# Split the dataset into a train and test set
data_to_train = txn_data[:1750]
data_to_test = txn_data[1750:]

# Save the resulting data to csv files
data_to_train.to_csv('train_data.csv')
data_to_test.to_csv('test_data.csv')

txn_data= txn_data.iloc[: , 3:4]
print(txn_data.head())

# Create NumPy array
training_dataset= txn_data.iloc[:1750,:].values
test_dataset= txn_data.iloc[1750:,:].values

sc= MinMaxScaler(feature_range=(0,1))
scaled_training_dataset= sc.fit_transform(training_dataset)

# Create a data structure with 90 timesteps and 1 output
x_train=[] # Independent variables
y_train= [] # Dependent variables 
for i in range(90,1750):
    x_train.append(scaled_training_dataset[i-90:i,0]) # Append 90 days prev data, not including 90 
    y_train.append(scaled_training_dataset[i,0])
    
x_train, y_train= np.array(x_train), np.array(y_train)
print(x_train.shape, y_train.shape)  # Check the current shape 

# Using AI LSMT Model (Long Short-Term Memory)
x_train= np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1)) # Re shape to 3D as required by LSMT model
x_train.shape

## Data is shaped and ready to be used within our LSTM Model
## I am currently in dependency hell with Python regarding tensorflow and keras
## So I will work on this 11-8 to fix my issues to show proof of concept with prediction model using LSMT
## Model related code is not included for now until I can get it to run
