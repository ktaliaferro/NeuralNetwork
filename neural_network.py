"""A three layer neural network for multi-class classification."""

import numpy as np
from scipy.optimize import fmin_cg
from scipy.special import expit

class Network:
    def __init__(self, n_input, n_hidden, n_output):
        """Store the neural network structure and initializes theta.
        All arguments are integers.
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        # randomize theta and
        # account for the bias units
        theta_1_size = self.n_hidden * (self.n_input + 1)
        ep_1 = 10. / np.sqrt(n_input)
        theta_1 = 2 * ep_1 * np.random.random(theta_1_size) - ep_1
        
        theta_2_size = self.n_output * (self.n_hidden + 1)
        ep_2 = 10. / np.sqrt(n_hidden)
        theta_2 = 2 * ep_2 * np.random.random(theta_2_size) - ep_2
        
        self.theta = np.concatenate((theta_1,theta_2))
    
    def predict(self, X):
        """Apply the neural network to the data.
        The data, X, is a 2 dimensional numpy array with one row for
        each example and one column for each feature.
        """
        if X.shape[1] != self.n_input:
            raise ValueError('The number of columns in X does not ' +
                             'match the number of input units in the ' +
                             'neural network.  ' +
                             '{0:d} != {1:d}.'.format(X.shape[1],
                                                     self.n_input))
        feat = X_to_feat(X)
        # shape theta into matrices
        # (self.n_input + 1)
        Theta1 = self.theta[:self.n_hidden * (self.n_input + 1)].reshape(
            self.n_hidden, (self.n_input + 1))
        Theta2 = self.theta[self.n_hidden * (self.n_input + 1):].reshape(
            self.n_output, self.n_hidden + 1)
            
        # layer 1, the input layer
        a = feat.T
        # layer 2, the hidden layer
        z = np.dot(Theta1,a)
        a = g(z)
        a = np.vstack((np.ones(a.shape[1]),a))
        # layer 3, the output layer
        z = np.dot(Theta2,a)
        a = g(z) # shape n_output x n_examples
        return a.T
        
    def learn(self, X, y_mat, n_iterations, lamb = 0.1, disp = False):
        """Learn the theta parameter.
        X is a 2 dimensional numpy array with one row for
        each example and one column for each feature.  y_mat is a a 2
        dimensional numpy array with one row for each example and
        one column for each output unit.  n_iterations is an integer.
        """
        if X.shape[1] != self.n_input:
            raise ValueError('The number of columns in X does not ' +
                             'match the number of input units in the ' +
                             'neural network.  ' +
                             '{0:d} != {1:d}.'.format(X.shape[1],
                                                     self.n_input))
        # If there is only one output unit, we will shape y_mat here
        # instead of using the function y_to_mat below.
        if self.n_output == 1:
            y_mat = y_mat.reshape(-1,1)
        feat = X_to_feat(X)
        # Apply conjugate gradient.
        self.theta = fmin_cg(J, self.theta, dJ, \
                    args=(self.n_hidden, self.n_output, feat, \
                        y_mat, lamb), maxiter=n_iterations, disp=disp)

    def cost(self, X, y_mat, lamb):
        """Return the cost using the current self.theta value
        X is a 2 dimensional numpy array with one row for
        each example and one column for each feature.  y_mat is a a 2
        dimensional numpy array with one row for each example and
        one column for each output unit."""
        feat = X_to_feat(X)
        # If there is only one output unit, we will shape y_mat here
        # instead of using the function y_to_mat below.
        if self.n_output == 1:
            y_mat = y_mat.reshape(-1,1)
        return J(self.theta, self.n_hidden, self.n_output,
            feat, y_mat, lamb)
            
    def theta_mat(self):
        """Return self.theta shaped as two matrices."""
        Theta1 = self.theta[:self.n_hidden * (self.n_input + 1)].reshape(
            self.n_hidden, (self.n_input + 1))
        Theta2 = self.theta[self.n_hidden * (self.n_input + 1):].reshape(
            self.n_output, self.n_hidden + 1)
        return Theta1, Theta2
        

def X_to_feat(X):
    """Append a column for the bias unit."""
    return np.hstack((np.ones(X.shape[0]).reshape(-1,1),X))

def mat_to_y(y_mat):
    """Given the output of the neural network as a prediction
    percentage for each class, return the class with the highest
    prediction percentage.
    """
    if y_mat.shape[1] == 1:
        return np.where(y_mat > .5, 1, 0).ravel()
    else:
        return np.argmax(y_mat, axis = 1)
    
def y_to_mat(y, n_classes):
    """Given an integer vector y of length m where elements are in [0,n],
    return an m by n output matrix for a neural network classifier.
    This works, but is not needed when n_classes == 1.
    """
    if n_classes == 1:
        return y.reshape(-1,1)
    else:
        y_mat = np.zeros((y.size, n_classes))
        for i in range(y.size):
            y_mat[i, y.ravel()[i]] = 1
        return y_mat
           
def g(z):
    """sigmoid function"""
    return expit(z)
    
def dg(z):
    """derivative of the sigmoid function"""
    gz = g(z)
    return gz * (1 - gz)
    
def accuracy(y_predict, y):
    """compute the accuracy of the prediction"""
    # here, y and y_predict have already been converted to a single
    # column with one row for each example in the classification
    # problem.
    return ((y_predict.ravel() == y.ravel()) + 0).sum() \
        * 1. / np.size(y_predict)

def J(theta, hidden_layer_size, num_labels, feat, y, lamb):
    """cost function"""
    # Reshape theta into matrices.
    # The i stands for 'internal' to indicate that these
    # variables are internal to this function.
    Theta1i = theta[:hidden_layer_size * feat.shape[1]].reshape( \
        hidden_layer_size, feat.shape[1])
    Theta2i = theta[hidden_layer_size * feat.shape[1]:].reshape( \
        num_labels, hidden_layer_size+1)
    
    # compute the probabilities h by going through the three layers
    # layer 1, the input layer
    a = feat.T
    # layer 2, the hidden layer
    z = np.dot(Theta1i,a)
    a = g(z)
    a = np.vstack((np.ones(a.shape[1]),a))
    # layer 3, the output layer
    z = np.dot(Theta2i,a)
    a = g(z)
    h = a.T
    
    # compute the cost
    m = feat.shape[0]
    cost = -1. / m * (np.tensordot(y, np.log(h)) \
        + np.tensordot(1-y, np.log(1-h)))
        
    # add regularization
    cost += lamb * .5 / m * (np.sum(Theta1i[:,1:]**2) \
        + np.sum(Theta2i[:,1:]**2))
    return cost
    
def dJ(theta, hidden_layer_size, num_labels, feat, y, lamb):
    """gradient of the cost function"""
    # Reshape theta into matrices.
    # The i stands for 'internal' to indicate that these
    # variables are internal to this function.
    Theta1i = theta[:hidden_layer_size * feat.shape[1]].reshape( \
        hidden_layer_size, feat.shape[1])
    Theta2i = theta[hidden_layer_size * feat.shape[1]:].reshape( \
        num_labels, hidden_layer_size+1)
    
    # Compute the probabilities h by going through the three layers.
    # Unlike in the cost function, keep the intermediate data.
    m = feat.shape[0]
    Delta1 = np.zeros_like(Theta1i)
    Delta2 = np.zeros_like(Theta2i)
    
    # layer 1, the input layer
    a1 = feat.T # shape (n_input + 1) x n_examples
    # layer 2, the hidden layer
    z2 = np.dot(Theta1i,a1) # shape n_hidden x n_examples
    a2 = g(z2) # shape n_hidden x n_examples
    a2 = np.vstack((np.ones((1,m)),a2)) # shape (n_hidden + 1) x n_examples
    # layer 3, the output layer
    z3 = np.dot(Theta2i,a2) # shape n_output x n_examples
    a3 = g(z3) # shape n_output x n_examples
    
    delta3 = a3-y.T # shape n_output x n_examples
    delta2 = (np.dot(Theta2i.T, delta3)[1:,:] * dg(z2))
        # shape n_hidden x n_examples
    Delta1 += np.dot(delta2,a1.T) # shape n_hidden x (n_input + 1)
    Delta2 += np.dot(delta3,a2.T) # shape n_output x n_hidden
               
    Delta1 /= 1. * m
    Delta2 /= 1. * m
    
    # Regularization
    Delta1[:,1:] += 1. * lamb / m * Theta1i[:,1:]
    Delta2[:,1:] += 1. * lamb / m * Theta2i[:,1:]
    return np.concatenate((Delta1.ravel(), Delta2.ravel()))
