# Self Organizing Map

"""
Fraud detection example.
Use Unsupervised deep learning to identify patterns in a high dimensional
dataset full of non-linear relationships

Dataset available at: https://archive.ics.uci.edu/ml/datasets/Statlog+%28Australian+Credit+Approval%29
"""

# Importing the libraries
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
# X will not have the final column (Approved vs Declined)
X = dataset.iloc[:, :-1].values

# Y will have only final column
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
"""
For this tut, we are importing from a file named minisom.py in the same directory.
However, it seems there is a recent version that can be pip installed.

https://pypi.org/project/MiniSom/
"""
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
"""
Initializes a Self Organizing Maps.
x,y - dimensions of the SOM (10x10 grid)
input_len - number of features in the dataset
sigma - radius of the different neighborhoods in the grid
learning_rate - initial learning rate
"""
# Initialize the weights
som.random_weights_init(X)

# Train the SOM
som.train_random(data = X, num_iteration = 100)


# Visualizing the results
"""
MID - mean inter-neuron distance
The higher the MID, the more the "winngin" node is an outlier,
which will signal the fraud.
"""


from pylab import bone, pcolor, colorbar, plot, show
# Initialize the window that will contain the map
bone()

"""Put the winning nodes on the map via their MIDs using distonce_map method,
which will return a matrix of all MIDs. We need to take the transpose (T) in
order for the data to be in the right shape for pcolor.
Different colors will correspond to MID ranges
"""
pcolor(som.distance_map().T)

# Make a legend for the colors
# The larger the MID, the more white the node will be in the visualization.
colorbar()

"""Add markers to make the distinction between approval vs decline of bank loan
red circles - did not get approval
green squares - approval"""
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x) # returns the winning node
    plot(w[0] + 0.5, # Adding 0.5 to the x, y coords of winning node
         w[1] + 0.5, # Bc theses are for the lower left corner of the square
         markers[y[i]], # circle or square
         markeredgecolor = colors[y[i]], # red or green border
         markerfacecolor = 'None', # no fill
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
"""Get explicit list of likely fraudulent customers"""

# returns a dict of winning nodes/customers coords
mappings = som.win_map(X)

# get the coords from visually inspecting the map
frauds = np.concatenate((mappings[(3,4)], mappings[(4,2)]), axis = 0)
# Inversed the feature scaling to ge the original numbers back
frauds = sc.inverse_transform(frauds)