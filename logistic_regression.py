#!/usr/bin/env python
import tensorflow as tf
import numpy as np

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#Parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 100
display_step = 1

#tf Graph Input
x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None, 10])

#Set model weights
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Construct the model
pred = tf.nn.softmax(tf.matmul(x,W) + b)
#Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices = 1))
#Gradient Descent
optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Initializing all variables
init = tf.initialize_all_variables()

#Launch the graph
with tf.Session() as sess:
	sess.run(init)

	#Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)

		#Loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)

			_, c = sess.run([optim, cost], feed_dict={x: batch_xs, y: batch_ys})

			avg_cost += c/total_batch
		#Display logs per epoch
		if (epoch+1) % display_step == 0:
			print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

	#Test mode
	correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	#Calculate accuracy for 3000 examples
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print "Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}) 
