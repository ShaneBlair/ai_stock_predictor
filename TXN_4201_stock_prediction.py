# -*- coding: utf-8 -*-
"""4201_stock_prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vY5ey0v8wfesFaRB4E5nR5b5hp1oP1mg
"""

!pip install numpy
!pip install --upgrade pandas
!pip install --upgrade pandas-datareader
# Use pandas web datareader to pull financial information
import pandas_datareader as web
import pandas as pd
import numpy as np

# Import stock data from Yahoo Finance API
user_input = input('Please enter a stock ticker: ')
stock_data= web.DataReader(user_input,data_source="yahoo",start='2015-01-01',end='2021-09-30')
stock_data.head()

stock_data.shape

#Pull the closing price in stock's history
!pip install matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
plt.figure(figsize=(16,8))
sns.lineplot(x= stock_data.index,y=stock_data['Close'])
plt.xlabel('Date', fontsize=20)
plt.ylabel('Closing Price USD',fontsize=20)
plt.title('Closing Price History',fontsize=25)

# Split into train and test dataset
data_to_train = stock_data[:1530]
data_to_test = stock_data[1530:]

# Save a training dataset and testing dataset
data_to_train.to_csv('train_data.csv')
data_to_test.to_csv('test_data.csv')

user_data= stock_data.iloc[: , 3:4]
user_data.head()

# Create NumPy array
training_dataset= user_data.iloc[:1530,:].values

testing_dataset= user_data.iloc[1530:,:].values

# Normalizae the training dataset
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1))
scaled_training_dataset = sc.fit_transform(training_dataset)

# Create data structure with 60 timesteps and 1 output
X_train= [] # Independent variable
y_train= [] # Dependent ''  
# Append past 60 days data 
for i in range(60,1530):
    X_train.append(scaled_training_dataset[i-60:i,0]) # Append previous 60 days, not including 60 (i - 60)
    y_train.append(scaled_training_dataset[i,0])
    
X_train, y_train= np.array(X_train), np.array(y_train)

X_train.shape, y_train.shape

# Reshape the LSMT model to be 3-D
X_train= np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_train.shape

# Importing the Keras libraries and packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#  Initialize the RNN
model= Sequential()

# Add the first LSTM layer and some dropout regularisation
model.add(LSTM(units=100,return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(rate=0.4))

# Second ''
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(rate=0.4))

# Third ''
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(rate=0.4))

# Fourth '' 
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(rate=0.4))

# Fifth ''
model.add(LSTM(units=100))
model.add(Dropout(rate=0.4))

# Add the Output Layer
model.add(Dense(units=1))

# Compiling the Model (regression)
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

# Fitting our model to the Training dataset
history=model.fit(X_train,y_train,epochs=35,batch_size=32)

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

real_stock_price = test_data.iloc[:, 3:4].values

real_stock_price.shape

testing_dataset.shape

# Concatenate the dataset and scale
data_total= pd.concat([train_data['Close'], test_data['Close']],  axis=0)
inputs = data_total[len(data_total)-len(test_data)-60:].values 
inputs = inputs.reshape(-1,1) 
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 230):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
# 3D format
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

inputs.shape

data_total.shape

X_test.shape

# Make prediction from model
predicted_stock_price = model.predict(X_test)

# Inverse scaling
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plot result
plt.figure(figsize=(15,8))
plt.plot(real_stock_price, color='Red', label='Real Stock Price')
plt.plot(predicted_stock_price, color='Blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction',fontsize=20)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Stock Price',fontsize=15)
plt.legend()
plt.show()

# Visualising the results
plt.figure(figsize=(15,8))
plt.plot(testing_dataset, color='Red', label='Real Stock Price')
plt.plot(predicted_stock_price, color='Blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

print("Real Stock Price closing values (by day): ")
print(pd.DataFrame(real_stock_price))
print("Predicted Stock Price closing values (by day): ")
print(pd.DataFrame(predicted_stock_price))