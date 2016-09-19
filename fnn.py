import tensorflow as tf
import numpy as np
from scipy.stats import norm


def fnn(x, output_dim):
    #weights and biases
    w1 = tf.Variable(tf.random_normal([10, 20], stddev=0.35), name="weights1")
    b1 = tf.Variable(tf.zeros([20]), name="biases1")

    w2 = tf.Variable(tf.random_normal([20, output_dim], stddev=0.35), name="weights2")
    b2 = tf.Variable(tf.zeros([20]), name="biases2")

    # nn operators
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    y2 = tf.nn.sigmoid(tf.matmul(y1,w2) + b2)
    return y2, [w1, w2]

# Defining the computational graph
x1 = tf.placeholder(tf.float32, shape=(1, 10)) 
y1, w1 = fnn(x1, 1)

# The second network has different weights and biases
x2 = tf.placeholder(tf.float32, shape=(1, 10)) 
y2, w2 = fnn(x2, 1)

# Initializing the session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    # Feeding and Fetching data
    theta1, theta2 = sess.run([w1, w2], {x1: np.random.random([1, 10]), x2: np.random.random([1, 10])})
    print(theta1)
    print(theta2)



# function for creating nn layers
def linear(x, out_dim, name, activation_fn=None):
    with tf.variable_scope(name):
        w = tf.get_variable(name='weights', shape=[x.get_shape()[1], out_dim], dtype=tf.float32, initializer=tf.random_normal_initializer())
        b = tf.get_variable(name='biases', shape=[out_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        out = tf.matmul(x, w) + b
        if activation_fn != None:
            out = activation_fn(out)
    return out, [w, b]


# Computational Graph
with tf.variable_scope("ffn") as scope:
    x1 = tf.placeholder(tf.float32, shape=(1, 10)) 
    y11, theta11 = linear(x1, 10, name="h", activation_fn=tf.nn.relu)
    y12, theta12 = linear(y1, 1, name="out", activation_fn=tf.nn.sigmoid)

    scope.reuse_variables()

    x2 = tf.placeholder(tf.float32, shape=(1, 10)) 
    y21, theta21 = linear(x2, 10, name="h", activation_fn=tf.nn.relu)
    y22, theta22 = linear(y1, 1, name="out", activation_fn=tf.nn.sigmoid)


# Initializing the session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    # Feeding and Fetching data
    theta1, theta2 = sess.run([theta12, theta22], {x1: np.random.random([1, 10]), x2: np.random.random([1, 10])})
    print(theta1[0])
    print(theta2[0])




