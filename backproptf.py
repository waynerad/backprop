"""This program does the same backpropagation as backpropnp.py, except instead of using
hand-coded numpy, it uses TensorFlow. As such, it only has to implement the forward pass;
the backward pass is figured out automatically by TensorFlow's differentiation system."""

import numpy as np
import tensorflow as tf

def create_tensorflow_graph(layer_list, num_training_examples):
    """Here we create the TensorFlow graph for the whole neural network -- all the layers.
    Each layer has a linear component (WX + B) and a non-linear component, which is ReLU.
    The TensorFlow graph also includes the loss, loss squared, and cost which is the loss
    squared divided by twice the number of training examples. Important to note that we
    use Gaussian (normal distribution) initialization (which is called
    glorot_normal_initializer)."""
    X = tf.placeholder(tf.float32, shape=[layer_list[0], num_training_examples], name="X")
    A = X
    for j in range(1, len(layer_list)):
        W = tf.get_variable("W" + str(j), shape=[layer_list[j], layer_list[j-1]],
                            dtype=tf.float32, initializer=tf.glorot_normal_initializer())
        B = tf.get_variable("B" + str(j), shape=[layer_list[j], 1], dtype=tf.float32,
                            initializer=tf.glorot_normal_initializer())
        # Z = WA + B (remember A = X for 1st later, so Z = WX + B)
        Z = tf.add(tf.matmul(W, A), B, name="Z" + str(j))
        A = tf.nn.relu(Z, name="A" + str(j))

    yhat = A
    Y = tf.placeholder(tf.float32, shape=yhat.shape, name="Y")
    loss = tf.subtract(yhat, Y, name="loss")
    loss_squared = tf.square(loss, name="loss_squared")
    thesum = tf.reduce_sum(loss_squared, name="thesum")
    number_of_training_examples = int(yhat.shape[1])
    cost = tf.divide(thesum, 2.0 * number_of_training_examples, name="cost")
    return cost, X, yhat, Y

def test_backprop_with_tensorflow():
    """This is our entry point to test the whole system. We construct the layer list,
    initialize by randomizing the training examples, and tell TensorFlow's
    GradientDescentOptimizer to minimize the cost function with learning rate alpha."""

    np.random.seed(17)
    tf.set_random_seed(17)
    layer_list = [7, 7, 7, 2]
    print("test", "layer_list", layer_list)
    num_training_examples = 5

    cost, X_node, yhat, Y_node = create_tensorflow_graph(layer_list, num_training_examples)

    X_val = np.random.randn(layer_list[0], num_training_examples)
    print("test", "X_val", X_val)

    Y_val = np.arange(10).reshape(2, 5)
    print("test", "Y_val", Y_val)

    alpha = 0.01
    train_op = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

    num_iterations = 10000

    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        num_prints = 5
        for i in range(num_iterations):

            train_op_value, cost_value, yhat_value = sess.run([train_op, cost, yhat],
                                                              feed_dict={X_node: X_val,
                                                                         Y_node: Y_val})
            print(i, "cost", cost_value)
            if i % int(num_iterations/num_prints) == 0:

                print(i, "yhat", yhat_value)

        print("yhat", yhat_value)

test_backprop_with_tensorflow()
