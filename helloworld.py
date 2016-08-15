#!/usr/bin/env python

import tensorflow as tf

#Create a constant operator
#The value represented by constructor represents output of operator
#The operator is added as a node to the default graph
hello = tf.constant('Hello tensorflow')

#Start a tf session
sess = tf.Session()
#Run the graph
print sess.run(hello)
