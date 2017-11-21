import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

# Step 2: Hyperparameters
learning_rate = 0.001
batch_size = 128
epochs = 35

# Step 3: Importing data
mnist = input_data.read_data_sets('E:/Tensorflow/MNIST', one_hot=True)

# Step 4: Features and labels
X = tf.placeholder(tf.float32, [None,784], name="X_placeholder")
Y = tf.placeholder(tf.float32, [None,10], name="Y_placeholder")

# Step 5: Parameters

w = tf.Variable(tf.random_normal(shape=[784,10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([batch_size,10], name="bias"))

# Step 6: Build the model

logits = tf.matmul(X,w) + b

# Step 7: Loss function

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="entropy")
loss = tf.reduce_mean(entropy)

# Step 8: Cost Function

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Step 9: Handle for initialization

init = tf.global_variables_initializer()

# Step 10: The Loop
with tf.Session() as sess:

    writer =  tf.summary.FileWriter('E:/Tensorflow/MNIST', sess.graph)

    start_time = time.time()

    sess.run(init)

    n_batches = int(mnist.train.num_examples/batch_size)

    for i in range(epochs):
        total_loss = 0

        for _ in range(n_batches):

            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X:X_batch, Y:Y_batch})
            total_loss += loss_batch

        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

    print('Total time: {0} seconds'.format(time.time() - start_time))

    print("Optimization done!")



#Step 11: Verification

    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds,1), tf.argmax(Y,1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0

    for i in range(n_batches):

        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy], feed_dict={X:X_batch, Y:Y_batch})

        total_correct_preds += accuracy_batch[-1]

    print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))

writer.close()
