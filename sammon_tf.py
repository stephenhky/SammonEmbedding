import numpy as np
import tensorflow as tf

def sammon_embedding(Xmat, initYmat, alpha=0.3, nbsteps=500):
    N = Xmat.shape[0]
    d = Xmat.shape[1]

    # define X
    X = tf.placeholder('float')
    Xshape = tf.shape(X)

    # distance matrix for X
    sqX = tf.reduce_sum(X * X, 1)
    sqX = tf.reshape(sqX, [-1, 1])
    sqDX = sqX - 2 * tf.matmul(X, tf.transpose(X)) + tf.transpose(sqX)
    DX = tf.sqrt(sqDX)

    # distance matrix for Y
    Y = tf.Variable(initYmat, dtype='float')
    sqY = tf.reduce_sum(Y * Y, 1)
    sqY = tf.reshape(sqY, [-1, 1])
    sqDY = sqY - 2 * tf.matmul(Y, tf.transpose(Y)) + tf.transpose(sqY)
    DY = tf.sqrt(sqDY)

    # cost function
    Z = tf.reduce_sum(DX) * 0.5
    numerator = tf.reduce_sum(tf.divide(tf.square(DX - DY), DX + tf.diag(tf.ones(N)))) * 0.5
    cost = numerator / Z

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
    init = tf.global_variables_initializer()

    # Tensorflow session
    sess = tf.Session()
    sess.run(init)
    embedded_pts = sess.run()