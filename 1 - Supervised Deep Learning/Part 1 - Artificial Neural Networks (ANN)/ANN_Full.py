# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) #for index 1
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]  # avoid dummy variable trap, remove first column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential  # for initialization
from keras.layers import Dense        # for building the densely connected layers

# Initializing the ANN
classifier = Sequential()  # defining the ANN as a sequence of layers, as opposed to a graph

# Adding the input layer and the first hidden layer
# 6 nodes in hidden layer, receiving 11 inputs (independent variables), with relu activation
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
# input_dim is no longer necessary, Dense knows the # of units in previous layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
# Since our output is a binary outcome, we use a sigmoid activation and 1 node
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# Updating weights after every 10 observations (batch size)
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test) # gives us probabilities that the person will leave
y_pred = (y_pred > 0.5)  # gives us binary True or False

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Homework
# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
# input the info into a horizontal vector with np.array 
"""the first [] will be for vertical (column) data
so the second [] will be for the horizontal (row) data
np.array([[put data here]])
"""
# Feature Scale with sc (defined earlier in code)
# One of the numbers should be a float (decimal) so the scaler function doesn't throw an error
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)


# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
'''
k-fold Cross validation:
    when k=10, the training set is split into 10 folds
    trains on 9 folds, tests on 1
    there will be 10 unique iterations of this
    use standard deviation to calculate the variance
    
This does not require that we already fit the model, since it includes that
step in the k-fold cross validation    
But make sure all the data preprocessing is done
'''

# Wrap the k-fold CV from sklearn into the keras model:
from keras.wrappers.scikit_learn import KerasClassifier
# Import the k-fold CV from sklearn:
from sklearn.model_selection import cross_val_score

# Needed to build classifier function
from keras.models import Sequential
from keras.layers import Dense

# Build/compile our classifier like we did above, but in a function
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# create a golbal classifier variable, trained with k-fold CV
#   build_fn = build function
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

# get accuracy of k-fold CV when k=10 (cv=10)
#   n_jobs=-1 means use all CPUS for computation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

"""
from keras.layers import Dropout
Apply to any/all layers (recommended to all hidden layers)
It would look like the following:
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(p=0.1))  # with 10% dropout
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p=0.1))  # with 10% dropout
"""


# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

#create function as before, but now with the argument 'optimizer'
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Do not input batch size or epochs in KerasClassifier, those will be tuned in the next line
classifier = KerasClassifier(build_fn = build_classifier)

#create a dict of the hyperparameters you want to tune
#use the same names of the parameters in KerasClassifier
parameters = {'batch_size': [25, 32],
              'epochs': [2, 5],             #I'm using very small #s so it doesn't take forever
              'optimizer': ['adam', 'rmsprop']} #optimizer matches argument name

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10) # for k-fold cv, # of folds

# fit to training set (it will take forever)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

'''
outputs:
    best_parameters: {'batch_size': 32, 'epochs': 5, 'optimizer': 'adam'}
    best_accuracy: 0.806125
'''

