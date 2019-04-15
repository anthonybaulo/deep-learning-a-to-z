# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values   #Extract all rows ":," from all columns except last ":-1"
y = dataset.iloc[:, -1].values    #Extract all rows ":;" from last column only "-1"


###########


# Taking care of missing data
from sklearn.preprocessing import Imputer

#Put cursor at Imputer and press Cmd+i for object info

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

#Fit imputer to the columns with missing data.  all rows ":," 
#from column indexes 1 and 2  "1:3" (upper limit is excluded)
imputer = imputer.fit(X[:, 1:3])

#Replace the missing data with the mean of the columns
X[:, 1:3] = imputer.transform(X[:, 1:3])


###########


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

#Apply to all rows of first column
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

#Create dummy variables
#Specify which features are categorical, first column
onehotencoder = OneHotEncoder(categorical_features = [0])

#Fit to matrix X
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#############


#Splitting the dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#############


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#We don't need to scale y bc the values are already b/t 1-0
#If the ranges were larger (like prices) we would scale


