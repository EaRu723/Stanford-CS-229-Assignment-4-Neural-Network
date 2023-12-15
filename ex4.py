import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import scipy.optimize as opt
import displayData as dd
import nncostfunction as ncf
import sigmoidgradient as sg
import randInitializeWeights as rinit
import checkNNGradients as cng
import predict as pd

plt.ion()

# Define the architecture of the neural network
input_layer_size = 400  # Input images are 20x20 pixels unrolled into a 400 node vector
hidden_layer_size = 25  # 25 hidden layer nodes
num_labels = 10         # 10 output labels, from 0 to 9
                        # We have mapped "0" to label 10


# ===================== Part 1: Loading and Visualizing Data =====================
# Load the dataset and display a random subset
print('Loading and Visualizing Data ...')

data = scio.loadmat('ex4data1.mat')
X = data['X'] # feature matrix
y = data['y'].flatten() # labels vector
m = y.size # number of training examples

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
selected = X[rand_indices[0:100], :]

dd.display_data(selected) # Visualize the selected data points

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Loading Parameters =====================
# Load pre-initialized NN parameters

print('Loading Saved Neural Network Parameters ...')

data = scio.loadmat('ex4weights.mat')
theta1 = data['Theta1']
theta2 = data['Theta2']

nn_params = np.concatenate([theta1.flatten(), theta2.flatten()])

# ===================== Part 3: Compute Cost (Feedforward) =====================
# Compute the cost of the neural network (without regularization)

print('Feedforward Using Neural Network ...')

# Weight regularization parameter (0 = no regularization).
lmd = 0

# Compute the cost and gradient with pre-loaded parameters
cost, grad = ncf.nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)

print('Cost at parameters (loaded from ex4weights): {:0.6f}\n(This value should be about 0.287629)'.format(cost))

input('Program paused. Press ENTER to continue')

# ===================== Part 4: Implement Regularization =====================
# Compute the cost with regularization

print('Checking Cost Function (w/ Regularization) ...')

# Weight regularization parameter (1 = regularization here).
lmd = 1

cost, grad = ncf.nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)

print('Cost at parameters (loaded from ex4weights): {:0.6f}\n(This value should be about 0.383770)'.format(cost))

input('Program paused. Press ENTER to continue')

# ===================== Part 5: Sigmoid Gradient =====================
# Evaluate the gradient of the sigmoid function

print('Evaluating sigmoid gradient ...')

g = sg.sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))

print('Sigmoid gradient evaluated at [-1  -0.5  0  0.5  1]:\n{}'.format(g))

input('Program paused. Press ENTER to continue')

# ===================== Part 6: Initializing Parameters =====================
# Randomly initialize the weights for the neural network

print('Initializing Neural Network Parameters ...')

initial_theta1 = rinit.rand_initialization(input_layer_size, hidden_layer_size)
initial_theta2 = rinit.rand_initialization(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_theta1.flatten(), initial_theta2.flatten()])

# ===================== Part 7: Implement Backpropagation =====================
# Check the backpropagation algorithm

print('Checking Backpropagation ... ')


# no regularization
lmd = 0
# check gradients
cng.check_nn_gradients(lmd)

input('Program paused. Press ENTER to continue')

# ===================== Part 8: Implement Regularization =====================
# Check backpropagation with regularization

print('Checking Backpropagation (w/ Regularization) ...')

# set regularizatoin parameter
lmd = 3
cng.check_nn_gradients(lmd)

# Also output the cost_function debugging values
debug_cost, _ = ncf.nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)

print('Cost at (fixed) debugging parameters (w/ lambda = {}): {:0.6f}\n(for lambda = 3, this value should be about 0.576051)'.format(lmd, debug_cost))

input('Program paused. Press ENTER to continue')

# ===================== Part 9: Training NN =====================
# Train the neural network using advanced optimization

print('Training Neural Network ... ')

# set the regularization network
lmd = 1

# define cost and gradient funcitons for the optimization algorithm
def cost_func(p):
    return ncf.nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)[0]


def grad_func(p):
    return ncf.nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)[1]

# use conjugate gradient optimization to find the optimal parameters
nn_params, *unused = opt.fmin_cg(cost_func, fprime=grad_func, x0=nn_params, maxiter=400, disp=True, full_output=True)

# Obtain theta1 and theta2 back from nn_params
theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

input('Program paused. Press ENTER to continue')

# ===================== Part 10: Visualize Weights =====================
# Visuale the weighs learned by the neural network

print('Visualizing Neural Network...')

dd.display_data(theta1[:, 1:])

input('Program paused. Press ENTER to continue')

# ===================== Part 11: Implement Predict =====================
# Use trained neural network to predict labels and compute accuracy

pred = pd.predict(theta1, theta2, X)

print('Training set accuracy: {}'.format(np.mean(pred == y)*100))

input('ex4 Finished. Press ENTER to exit')
