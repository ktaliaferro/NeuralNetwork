"""Demonstrate use of the neural_network.Network class on the handwritten digits
data set from the sklearn Python package."""

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

import neural_network

# Import the handwritten digits dataset from sklearn.
digits = datasets.load_digits()
# X is a 1797x64 array.  Each row of X corresponds to an 8x8 pixel image of a
# handwritten digit.
X = digits['data']
# y is an array with 1797 rows.  Each row contains an integer in range(10)
# corresponding to a class label.
y = digits['target']

# Normalize the columns of X.
X = StandardScaler().fit_transform(X)

# Randomly permute the examples.
rand = np.random.permutation(len(y))
X = X[rand,:]
y = y[rand]

# Since the output will be a probability for each class, we will
# put the class labels y into this format as well.  y_mat will be a
# 1797 x 10 array where each row contains a one in the appropriate class
# column and zeros in all other columns.
y_mat = neural_network.y_to_mat(y,digits['target_names'].shape[0])

# Put the cutoff between the training set and the test set at 70%.
cutoff = int(.7 * len(y))

# Initialize a neural network with 25 hidden units.
n_network = neural_network.Network(X.shape[1], 25, y_mat.shape[1])

for i in range(10):
    # Train the neural network on the training set using
    # n_iterations more iterations of conjugate gradient.
    n_iterations = 5
    n_network.learn(X[:cutoff,:], y_mat[:cutoff,:], n_iterations = n_iterations,
            lamb = .1, disp=False)
    # Measure the neural network accuracy on the test set.
    y_prob = n_network.predict(X[cutoff:,:])
    y_pred = neural_network.mat_to_y(y_prob)
    prediction_accuracy = neural_network.accuracy(y_pred, y[cutoff:])
    # Print out the predication accuracy.  Ideally, the accuracy will increase
    # on each iteration of this for loop.
    print ('{0:.2f}% accuracy on the test set after {1:2d} iterations of '
            + 'conjugate gradient.').format(
                    prediction_accuracy * 100, n_iterations * (1+i))
