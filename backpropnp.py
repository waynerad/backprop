"""The backpropagation algorithm, coded from scratch in plain Numpy"""

import numpy as np

def forward_propagation(X, W, B):
    """Calculate one forward propagation pass with yhat = WX + B followed by ReLU"""
    # We are doing a linear transformation with WX + B,
    # then we are doing a ReLU with the "maximum"
    Z = np.matmul(W, X) + B
    yhat = np.maximum(0, Z)
    return Z, yhat

def loss(A, Y):
    """The loss, not to be confused with the cost, is just the difference between yhat (the
    values we calculated) and Y (the correct result)"""
    return A - Y

def cost(lossvalues):
    """The cost function we are using here is the square of the error. Many different cost
    functions can be used in machine learning."""
    return np.sum(lossvalues ** 2) / (2 * lossvalues.shape[1])

def relu_gradient_and_chain_rule(dA, Z):
    """Here we take the difference in our activation layer (dA) and, using the chain rule in
    calculus, calculate the delta for dZ, the input into our ReLU layer. Because ReLu is
    so simple, this calculation is very simple."""
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear_transform_gradient_and_chain_rule(X, dZ, W):
    """Here we take the deltas from our activation layer, contained in dZ, and, using the
    chain rule in calculus, work backwards to get the deltas for dX, dW, and dW going
    backward through our linear transform."""
    m = X.shape[1]
    dW = (1/m) * np.dot(dZ, X.T)
    dB = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dX = np.dot(W.T, dZ)
    return dX, dW, dB

def both_relu_and_linear_transform_and_chain_rule(dA, Z, X, W):
    """Here we start with the delta of our activation layer, and do our backpropagation
    through both the ReLU non-linear activation layer and the linear (WX + B)
    layer."""
    dZ = relu_gradient_and_chain_rule(dA, Z)
    dX, dW, dB = linear_transform_gradient_and_chain_rule(X, dZ, W)
    return dX, dW, dB

def gradient_descent_for_n_layers(num_layers, W, B, X, Y, alpha, num_iterations):
    """Here we have a multi-layer neural network. For this, W and B actually contain an
    extra dimension so they can contain all the parameters for all the layers."""
    Z = [None] * (num_layers + 1)
    A = [None] * (num_layers + 1)
    dA = [None] * (num_layers + 1)
    dW = [None] * (num_layers + 1)
    dB = [None] * (num_layers + 1)
    for _ in range(0, num_iterations):

        A[0] = np.array(X, copy=True)
        for j in range(1, num_layers + 1):
            Z[j], A[j] = forward_propagation(A[j - 1], W[j], B[j])

        yhat = A[num_layers]
        dA[num_layers] = loss(yhat, Y)
        cost_value = cost(dA[num_layers])
        print("cost", cost_value)

        for j in reversed(range(1, num_layers + 1)):
            dA[j-1], dW[j], dB[j] = both_relu_and_linear_transform_and_chain_rule(
                dA[j], Z[j], A[j-1], W[j])
            W[j] = W[j] - (alpha * dW[j])
            B[j] = B[j] - (alpha * dB[j])

    return W, B, yhat

def initialize_parameters_for_layers(layer_list, training_examples):
    """Uses the Gaussian (normal) distribution to initialize the parameters
    for all the layers. It randomizes the training examples as layer 0."""
    W = [None] * len(layer_list)
    B = [None] * len(layer_list)
    X = np.random.randn(layer_list[0], training_examples)
    for j in range(1, len(layer_list)):
        W[j] = np.random.randn(layer_list[j], layer_list[j-1])
        B[j] = np.random.randn(layer_list[j], 1)
    return X, W, B

def test_fwd_propagation():
    """Test of forward propagation and prints output to display."""

    X = np.random.randn(3, 5)
    print("test", "X", X)
    print("test", "X.shape", X.shape)
    W = np.random.randn(2, 3)
    print("test", "W", W)
    print("test", "W.shape", W.shape)
    B = np.random.randn(2, 1)
    print("test", "B", B)
    print("test", "B.shape", B.shape)
    Z, yhat = forward_propagation(X, W, B)
    print("test", "Z", Z)
    print("test", "yhat", yhat)

    ycorrect = np.arange(10).reshape(2, 5)
    print("test", "ycorrect", ycorrect)
    test_loss = loss(yhat, ycorrect)
    print("test", "test_loss", test_loss)
    test_cost = cost(test_loss)
    print("test", "test_cost", test_cost)

    # dA = A - Y
    dA = yhat - ycorrect

    dX, dW, dB = both_relu_and_linear_transform_and_chain_rule(dA, Z, X, W)
    print("test", "dW", dW)
    print("test", "dB", dB)

def test_multiple_gradient_descent_with_backprop():
    """Test of full gradient descent. Prints output to display."""
    layer_list = [7, 7, 7, 2]
    print("test", "layer_list", layer_list)

    X, W, B = initialize_parameters_for_layers(layer_list, 5)

    alpha = 0.01
    num_iterations = 2000
    num_layers = len(layer_list) - 1

    Y = np.arange(10).reshape(2, 5)
    W, B, yhat = gradient_descent_for_n_layers(num_layers, W, B, X, Y, alpha, num_iterations)

    print("test done.") # final W and B are", W, "and", B)
    print("final yhat", yhat)

np.random.seed(17)
test_multiple_gradient_descent_with_backprop()
