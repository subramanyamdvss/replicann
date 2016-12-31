


"""This is a neural network which is made just from input and output given by another neural network,,,,,,,,,,The results are 
TERRIBLE 9% accuracy""" 




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf



data_dir = '/tmp/tensorflow/mnist/input_data'

def model1():

  mnist = input_data.read_data_sets(data_dir, one_hot=True)
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = (tf.matmul(x, W) + b)
  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
  sess = tf.InteractiveSession()
  tf.initialize_all_variables().run()
  
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
  return W,b



W2,b2=model1()

def hthet1(x):
  
  y = tf.nn.relu(tf.matmul(x, W2) + b2)
  
  
  return y


def model2():
  # Import data
  mnist = input_data.read_data_sets(data_dir, one_hot=True)

  # Create the model
  x1 = tf.placeholder(tf.float32, [None, 784])
  W1 = tf.Variable(tf.zeros([784, 10]))
  b1 = tf.Variable(tf.zeros([10]))
  y1 = (tf.matmul(x1, W1) + b1)

  # Define loss and optimizer
  y_1 = tf.placeholder(tf.float32, [None, 10])
  l2 = tf.nn.l2_loss(y1 - y_1)
  train_step1 = tf.train.GradientDescentOptimizer(0.5).minimize(l2)
  sess1 = tf.InteractiveSession()
  tf.initialize_all_variables().run()
  # Train
 
  for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    y=hthet1(batch_xs)
    y=sess1.run(y)
    sess1.run(train_step1, feed_dict={x1: batch_xs, y_1:y })
  
  # Test trained model
    if i%100==0:
      correct_prediction1 = tf.equal(tf.argmax(y1, 1), tf.argmax(y_1, 1))
      accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
      print(sess1.run(accuracy1, feed_dict={x1: mnist.test.images,
                                      y_1: mnist.test.labels}))
model2()
