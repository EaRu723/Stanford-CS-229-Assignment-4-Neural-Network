import numpy as np
from sigmoid import *


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd):
    # Reshape nn_params back into the parameters theta1 and theta2, the weight 2-D arrays
    # for our two layer neural network
    theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1) # weights for layer 1 
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1) # weights for layer 2

    # number of training examples
    m = y.size

    # Initializing output for later
    cost = 0
    theta1_grad = np.zeros(theta1.shape)  # 25 x 401
    theta2_grad = np.zeros(theta2.shape)  # 10 x 26

    Y = np.zeros((m, num_labels)) # 5000 x 10

    for i in range(m): 
        Y[i, y[i] - 1] = 1
    
    a1 = np.c_[np.ones(m), X] # Input layer (5000 x 401)
    a2 = np.c_[np.ones(m), sigmoid(np.dot(a1, theta1.T))] # Hidden layer (5000 x 26)
    hypothesis = sigmoid(np.dot(a2, theta2.T)) # Output layer = prediction (5000 x 10)

    # remove bias terms
    reg_theta1 = theta1[:, 1:] # 25 x 400
    reg_theta2 = theta2[:, 1:] # 10 x 25

    # compute cost using formula for NN Cost Function
    cost = (np.sum(-Y * np.log(hypothesis) - np.subtract(1,Y) * np.log(np.subtract(1, hypothesis))) / m) + (lmd / (2 * m)) * (np.sum(reg_theta1 * reg_theta1) + np.sum(reg_theta2 ** 2))

    # Backpropagation
    e3 = hypothesis - Y # Error term of output layet (5000 x 10)
    e2 = np.dot(e3, theta2) * (a2 * np.subtract(1, a2)) # Error of hidden layer (5000 x 26)
    e2 = e2[:, 1:] # remove intercept column (5000x25)

    # Gradients
    delta1 = np.dot(e2.T, a1) # Gradients for theta1 (25 x 401)
    delta2 = np.dot(e3.T, a2) # Gradients for theta2 (10 x 26)

    # Regularization terms for gradients
    p1 = (lmd / m) * np.c_[np.zeros(hidden_layer_size), reg_theta1] 
    p2 = (lmd / m) * np.c_[np.zeros(num_labels), reg_theta2]

    # Sum of gradient and regulatizaton term
    theta1_grad = p1 + (delta1 / m)
    theta2_grad = p2 + (delta2 /m)

    # Combines the gradients into a single vector
    grad = np.concatenate([theta1_grad.flatten(), theta2_grad.flatten()])

    return cost, grad
