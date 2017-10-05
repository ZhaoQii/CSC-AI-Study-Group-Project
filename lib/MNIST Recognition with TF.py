
# coding: utf-8

# # Qi's AI Group Assignment

# Import the packages

import numpy as np
import tensorflow as tf
import pandas as pd
import scipy as sp
from tensorflow.examples.tutorials.mnist import input_data


# Getting MNIST dataset directly using tensorflow's input_data

mnist = input_data.read_data_sets("MNIST", one_hot=True)

# By setting one_hot=True here, the labels are stored into 10-dimensional vector

# Setting hyperparameters, learning rate, epoch and batch size
learning_rate = 0.5
epochs = 20
batch_size = 100

# Setting Input and Output according to our dataset's dimensions
x = tf.placeholder(tf.float32, [None, 784])		
y = tf.placeholder(tf.float32, [None, 10])

# Here I set up a 4 layers network which are input, hidden and hidden2 and output 
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')
W3 = tf.Variable(tf.random_normal([50, 10], stddev=0.03), name='W3')
b3 = tf.Variable(tf.random_normal([10]), name='b3')
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)
hidden_out2 = tf.add(tf.matmul(hidden_out, W2), b2)
hidden_out2 = tf.nn.relu(hidden_out2)
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out2, W3), b3))
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)

# Set cross entropy function as loss function
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))

# Set a simple gradient descent optimizer
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
init_op = tf.global_variables_initializer()

# Get the prediction accuracy return
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# Train and test
with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
