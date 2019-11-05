import logging
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logging.basicConfig(level=logging.ERROR)

# Network parameters
n_hidden1 = 10
n_hidden2 = 10
n_input = 2
n_output = 2

# Learning parameters
learning_constant = 0.2
number_epochs = 1000
batch_size = 1000

# DEFINING WEIGHTS AND BIASES

# Biases first hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1]))
# Biases second hidden layer
b2 = tf.Variable(tf.random_normal([n_hidden2]))
# Biases output layer
b3 = tf.Variable(tf.random_normal([n_output]))

# Weights connecting input layer with first hidden layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
# Weights connecting first hidden layer with second hidden layer
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
# Weights connecting second hidden layer with output layer
w3 = tf.Variable(tf.random_normal([n_hidden2, n_output]))

# Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])


def multilayer_perceptron(input_d):
    # Task of neurons of first hidden layer
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1))
    # Task of neurons of second hidden layer
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
    # Task of neurons of output layer
    out_layer = tf.add(tf.matmul(layer_2, w3), b3)

    return out_layer


def trainNetwork():
    # Create model
    neural_network = multilayer_perceptron(X)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network, Y))

    #loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(
        learning_constant).minimize(loss_op)

    # Initializing the variables
    init = tf.global_variables_initializer()

    batch_x1 = np.loadtxt('data/x1.txt')
    batch_x2 = np.loadtxt('data/x2.txt')
    batch_y1 = np.loadtxt('data/y1.txt')
    batch_y2 = np.loadtxt('data/y2.txt')

    label = batch_y2  # +1e-50-1e-50

    batch_x = np.column_stack((batch_x1, batch_x2))
    batch_y = np.column_stack((batch_y1, batch_y2))

    batch_x_train = batch_x[:, 0:599]

    batch_y_train = batch_y[:, 0:599]

    batch_x_test = batch_x[:, 600:1000]
    batch_y_test = batch_y[:, 600:1000]

    label_train = label[0:599]

    label_test = label[600:1000]

    with tf.Session() as sess:
        sess.run(init)
        # Training epoch
        for epoch in range(number_epochs):

            sess.run(optimizer, feed_dict={X: batch_x_train, Y: batch_y_train})
            # Display the epoch
            if epoch % 100 == 0:
                print("Epoch:", '%d' % (epoch))

        # Test model
        pred = (neural_network)  # Apply softmax to logits
        accuracy = tf.keras.losses.MSE(pred, Y)
        print("Training Accuracy:", accuracy.eval(
            {X: batch_x_train, Y: batch_y_train}))
        # tf.keras.evaluate(pred,batch_x)

        print("Prediction:", pred.eval({X: batch_x_train}))
        output = neural_network.eval({X: batch_x_train})
        plt.plot(batch_y_train[0:10], 'ro', output[0:10], 'bo')
        plt.ylabel('some numbers')

        estimated_class = tf.argmax(pred, 1)  # +1e-50-1e-50
        correct_prediction1 = tf.equal(tf.argmax(pred, 1), label)
        accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

        print("Test Accuracy:", accuracy1.eval({X: batch_x}))

        plt.show()
