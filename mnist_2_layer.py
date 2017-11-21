import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

mnist = input_data.read_data_sets('E:/Tensorflow/MNIST', one_hot=True)

# Hyperparameters
learning_rate = 0.001
batch_size = 128
epochs = 35

# Parameters
X = tf.placeholder(tf.float32, [batch_size,784], name="X_placeholder")
Y = tf.placeholder(tf.float32, [batch_size,10], name="Y_placeholder")

w1 = tf.Variable(tf.random_normal([784,60], stddev=0.01), name="W1_layer1")
b1 = tf.Variable(tf.zeros([batch_size,60], name="B1_layer1"))

w2 = tf.Variable(tf.random_normal([60,10], stddev=0.01), name="W2_layer2")
b2 = tf.Variable(tf.zeros([batch_size,10]), name="B2_layer2")

'''w3 = tf.Variable(tf.random_normal([100,60], stddev=0.0), name="W3_layer3")
b3 = tf.Variable(tf.zeros([batch_size,60]), name="B3_layer3")

w4 = tf.Variable(tf.random_normal([60,30], stddev=0.0), name="W4_layer4")
b4 = tf.Variable(tf.zeros([batch_size,30]), name="B4_layer4")

w5 = tf.Variable(tf.random_normal([30,10], stddev=0.0), name="W5_outputLayer")
b5 = tf.Variable(tf.zeros([batch_size,10]), name="B5_outputLayer")
'''
# Activation Function
Y1 = tf.nn.sigmoid(tf.matmul(X,w1) + b1)
#Y2 = tf.nn.sigmoid(tf.matmul(Y1,w2) + b2)
#Y3 = tf.nn.sigmoid(tf.matmul(Y2,w3) + b3)
#Y4 = tf.nn.sigmoid(tf.matmul(Y3,w4) + b4)

logits = tf.matmul(Y1,w2) + b2
Y_ = tf.nn.softmax(logits)

# Loss Function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="Entropy")

loss = tf.reduce_mean(entropy)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initializer
init = tf.global_variables_initializer()

# The Loop

with tf.Session() as sess:

    total_loss = 1
    n_batch = int(mnist.train.num_examples/batch_size)
    sess.run(init)
    for i in range(epochs):

        start_time = time.time()

        for _ in range(n_batch):

            X_batch, Y_batch = mnist.train.next_batch(batch_size)

            _, loss_batch = sess.run([optimizer,loss], feed_dict={X:X_batch, Y:Y_batch})

        total_loss += loss_batch


        print("Average loss: {0}: {1} {2}".format(i, loss_batch, total_loss/n_batch))


    print("Total time: {0} sec".format((time.time() - start_time)))

    print("Optimization Done!")

    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(


    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0
	
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y:Y_batch})
        #print(accuracy_batch)

        total_correct_preds += accuracy_batch[-1]
	
    print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))
