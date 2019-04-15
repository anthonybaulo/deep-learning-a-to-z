# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
"""Keras can only take numpy arrays as input. Pandas creats that"""

training_set = dataset_train.iloc[:, 1:2].values
"""iloc[:, 1:2].values  means all of the row data (:) for column 1 (1:2) the
upper bound (2) is not included in the data
.values creates a numpy array
"""

# Feature Scaling
"""Normalization is used since there is a sigmoid 
function in the output layer"""
from sklearn.preprocessing import MinMaxScaler

"""All of the scaled data will be bt 0 and 1"""
sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(training_set)



# Creating a data structure with 60 timesteps and 1 output
"""At every step, the RNN will consider the previous 60 steps to predict the 
single next step.
This number came from the instructor experimenting
"""
X_train = []
y_train = []

"""We have to start at the 60th date, so we can look back 60 days
The upper bound will be the number of data values in the training set"""

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    # The range of rows (i-60) to i, from column 0
    # The upper bound (i) is not included
    
    y_train.append(training_set_scaled[i, 0])
    # Row i from column 0
    # The stock price for day i
    
"""Convert the lists to np arrays"""
X_train, y_train = np.array(X_train), np.array(y_train)


# Reshaping
"""Adding another dimension to the training set to make room for 
adding other indicators if desired

A Keras RNN Layer expects 3D tensor with shape: 
(batch_size, timesteps, input_dim)"""

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
"""
X_train.shape[0] = # of rows
X_train.shape[1] = # of columns
1 = # of indicators we're using"""



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()
"""regression predicts a continous value"""


# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
"""
units (neurons) - somewhat arbitrary, though you don't want too low
return_sequences=True if there will be another LSTM layer after this
input_shape corresponds to the timesteps and indicators, the first dimension
is already taken into account
"""
regressor.add(Dropout(0.2))


# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
"""You do not need to return sequences since the next layer is Dense"""
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))
"""single dimension - the next stock price"""

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
"""normally RMSProp is recommended for RNNs, but instructor has already 
experimented and found adam to out perform
mean_squared_error is good for RNN"""

#################################################
# Don't run this on the CPU, only on Google Colab

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
"""Instructor experimented to find 100 epochs, and 32 batch_size is suitable
for this particular training set"""

#################################################
"""I trained this on Google Colab and saved the model to GoogleStock.h5"""
# Load model from file

from keras.models import load_model

regressor = load_model('GoogleStock.h5')
print("Loaded model from file")
#################################################



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
"""We have to concatenate the pre-scaled training set with the test set
because we don't want to change the test set at all.
We want the data in the columns labeled 'Open'
axis = 0 means we're concatenating vertically (adding more rows)"""
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

"""total dataset minus the test set would be the first day of the test set
minus 60 would be the lower bound of the range of days we'll need to
use at input. The : means to take all the remaning values to the end of the set.
.values makes it a numpy array"""
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

"""Make it the correct shape"""
inputs = inputs.reshape(-1,1)

"""Scale the inputs"""
inputs = sc.transform(inputs)



"""We are testing the 20 trading days in Jan 2017
The 'inputs' variable has 80 values, the 20 days of Jan, with 60 previous days"""
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
    # The range of rows (i-60) to i, from column 0
    # The upper bound (i) is not included

"""Convert to numpy array"""
X_test = np.array(X_test)

"""The RNN expects a 3D format"""
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

"""Make the prediction"""
predicted_stock_price = regressor.predict(X_test)

"""transform back to original scale"""
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

"""
Parameter Tuning for Regression is the same as Parameter Tuning for 
Classification which you learned in Part 1 - Artificial Neural Networks, 
the only difference is that you have to replace:

scoring = 'accuracy'  

by:

scoring = 'neg_mean_squared_error' 

in the GridSearchCV class parameters.
"""