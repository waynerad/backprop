# backprop
The backpropagation algorithm, coded from scratch in plain Numpy, so I'm sure I understand how it works, then reimplemented again using TensorFlow.

The Numpy implementation is in backpropnp.py. The forward propagation is handled by the forward_propagation() function (with ReLU activation function), loss, and cost functions, with relu_gradient_and_chain_rule() doing backpropagation on the ReLU, and linear_transform_gradient_and_chain_rule() doing backpropagation on the linear transform.

The function both_relu_and_linear_transform_and_chain_rule() packages both into one handy function, used by gradient_descent_for_one_layer() and gradient_descent_for_n_layers(). A variable called layer_list lets you make as many layers as you want, with whatever sizes you want. (They all have to be fully connected and use ReLU, though.)

The various functions can be run by running the test functions. A print in gradient_descent_for_n_layers() that you can uncomment will show you the cost function, which should be monotonically declining.

The TensorFlow version is in backproptf.py. In TensorFlow, only the forward propagation is coded. TensorFlow's autodifferentiation facilities figure out all the backward gradients for you. As before, test code is provided for running the backprop code.

