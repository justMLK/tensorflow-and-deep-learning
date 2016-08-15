#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
print ("Packs loaded")

#Download and extract MNIST data

mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print ("MNIST loaded")

# Parameters of logistic regression
learning_rate = 0.01
training_epochs = 50
batch_size = 100
display_step = 1

#Create Graph for Logistic Regression
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Activation, Cost,Optimizing functions

actv = tf.nn.softmax(tf.matmul(x,W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), reduction_indices=1))

optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#Gradient Descent

pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y,1))
accr = tf.reduce_mean(tf.cast(pred, "float"))
#Optimize with tensorflow
#Initializing the variables
init = tf.initialize_all_variables()
print ("Network constructed")

#Launch the graph
with tf.Session() as sess:
	sess.run(init)

	#Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		num_batch = int(mnist.train.num_examples/batch_size)
		#Loop over all batches
		for i in range(num_batch):
			if 0: #Using tensorflow API
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			else: # Random batch sampling
				randidx = np.random.randint(trainimg.shape[0], size=batch_size)
				batch_xs = trainimg[randidx, :]
				batch_ys = trainlabel[randidx, :]

			#Fit training using batch data
			sess.run(optim, feed_dict={x: batch_xs, y: batch_ys})
			#Compute average loss
			avg_cost += sess.run(cost, feed_dict={x:batch_xs, y:batch_ys})/num_batch

		#Display logs per epoch step
		if epoch%display_step ==0:
			train_acc = accr.eval({x:batch_xs, y:batch_ys})
			print ("Epoch: %03d/%03d cost: %.9f train_acc: %.3f" % (epoch, training_epochs, avg_cost, train_acc))

	print ("Optimization Complete!")

	#Test model
	#Calculate accuracy
	print ("Accuracy: ", accr.eval({x: mnist.test.images, y: mnist.test.labels}))

print("Done.")
