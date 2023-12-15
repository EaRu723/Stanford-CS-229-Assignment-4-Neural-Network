import numpy as np

# Initialize a weight matrix for a layer in NN
# l_in: number of units in the previous layer (input layer for the current weights)
# l_out: number of units in the current layer (output layer for the current weights)
def rand_initialization(l_in, l_out):
    # Initialize weight matrix with zeros
    w = np.zeros((l_out, 1 + l_in))

    # Set the range for the initial weights
    ep_init = 0.12

    # Randomly initialize the weights within the range [-ep_init, ep_init]
    w = np.random.rand(l_out, 1 + l_in) * (w * ep_init) - ep_init

    return w
