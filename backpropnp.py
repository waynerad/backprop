import numpy as np

def forward_propagation(X, W, B):
    # We are doing a linear transformation with WX + B,
    # then we are doing a ReLU with the "maximum"
    Z = np.matmul(W, X) + B
    yhat = np.maximum(0, Z)
    return Z, yhat

def loss(A, Y):
    return A - Y

def cost(loss):
    # squared error
    return np.sum(loss ** 2) /(2*loss.shape[1])

def relu_gradient_and_chain_rule(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def linear_transform_gradient_and_chain_rule(X, dZ, W, B):
    m = X.shape[1]
    dW = (1/m) * np.dot(dZ, X.T)
    dB = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    dX = np.dot(W.T, dZ)
    return dX, dW, dB

def both_relu_and_linear_transform_and_chain_rule(dA, Z, X, W, B):
    dZ = relu_gradient_and_chain_rule(dA, Z)
    dX, dW, dB = linear_transform_gradient_and_chain_rule(X, dZ, W, B)
    return dX, dW, dB

def gradient_descent_for_one_layer(W, B, X, Y, alpha, num_iterations):
    for i in range(0, num_iterations):
        Z, yhat = forward_propagation(X, W, B)
        dA = loss(yhat, Y)
        cost_value = cost(dA)
        # print("cost", cost_value)
        dX, dW, dB = both_relu_and_linear_transform_and_chain_rule(dA, Z, X, W, B)
        W = W - (alpha * dW)
        B = B - (alpha * dB)
    return W, B

def gradient_descent_for_n_layers(num_layers, W, B, X, Y, alpha, num_iterations):
    Z = [None] * (num_layers + 1)
    A = [None] * (num_layers + 1)
    dA = [None] * (num_layers + 1)
    dW = [None] * (num_layers + 1)
    dB = [None] * (num_layers + 1)
    for i in range(0, num_iterations):

        A[0] = np.array(X, copy = True)
        for j in range(1, num_layers + 1):
            Z[j], A[j] = forward_propagation(A[j - 1], W[j], B[j])

        yhat = A[num_layers]
        dA[num_layers] = loss(yhat, Y)
        cost_value = cost(dA[num_layers])
        print("cost", cost_value)

        for j in reversed(range(1, num_layers + 1)):
            dA[j-1], dW[j], dB[j] = both_relu_and_linear_transform_and_chain_rule(dA[j], Z[j], A[j-1], W[j], B[j])
            W[j] = W[j] - (alpha * dW[j])
            B[j] = B[j] - (alpha * dB[j])

    return W, B, yhat

def initialize_parameters_for_layers(layer_list, training_examples):
    W = [None] * len(layer_list)
    B = [None] * len(layer_list)
    X = np.random.randn(layer_list[0], training_examples)
    for j in range(1, len(layer_list)):
        W[j] = np.random.randn(layer_list[j],layer_list[j-1])
        B[j] = np.random.randn(layer_list[j],1)
    return X, W, B

def test_fwd_propagation():

    X = np.random.randn(3,5)
    print("test", "X", X)
    print("test", "X.shape", X.shape)
    W = np.random.randn(2,3)
    print("test", "W", W)
    print("test", "W.shape", W.shape)
    B = np.random.randn(2,1)
    print("test", "B", B)
    print("test", "B.shape", B.shape)
    Z, yhat = forward_propagation(X, W, B)
    print("test", "Z", Z)
    print("test", "yhat", yhat)

    ycorrect = np.arange(10).reshape(2,5)
    print("test", "ycorrect", ycorrect)
    test_loss = loss(yhat, ycorrect)
    print("test", "test_loss", test_loss)
    test_cost = cost(test_loss)
    print("test", "test_cost", test_cost)

    # dA = A - Y
    dA = yhat - ycorrect

    dX, dW, dB = both_relu_and_linear_transform_and_chain_rule(dA, Z, X, W, B)  
    print("test","dW",dW)
    print("test","dB",dB)

def test_gradient_descent():

    X = np.random.randn(3,5)
    print("test", "X", X)
    print("test", "X.shape", X.shape)
    W = np.random.randn(2,3)
    B = np.random.randn(2,1)

    Y = np.arange(10).reshape(2,5)
    print("test", "Y", Y)

    alpha = 1.3
    num_iterations = 32
    W, B = gradient_descent(W, B, X, Y, alpha, num_iterations)

def test_multiple_gradient_descent_with_backprop():
    # layer_list = [3, 2, 4, 2]
    layer_list = [7, 7, 7, 2]
    print("test", "layer_list", layer_list)

    X, W, B = initialize_parameters_for_layers(layer_list, 5)

    alpha = 0.01
    num_iterations = 2000
    num_layers = len(layer_list) - 1

    Y = np.arange(10).reshape(2,5)
    W, B, yhat = gradient_descent_for_n_layers(num_layers, W, B, X, Y, alpha, num_iterations)

    print("test done.") # final W and B are", W, "and", B)
    print("final yhat", yhat)

np.random.seed(17)
test_multiple_gradient_descent_with_backprop()


