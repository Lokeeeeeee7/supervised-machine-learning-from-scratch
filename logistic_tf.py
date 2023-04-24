"""
This file is for binary classification using TensorFlow
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from util import get_vehicle_data
from logistic_np import *

if __name__ == "__main__":
    np.random.seed(2018)
    tf.compat.v1.set_random_seed(2018)

    # Load data from file
    # Make sure that vehicles.dat is in data/
    train_x, train_y, test_x, test_y = get_vehicle_data()
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]

    #generate_unit_testcase(train_x.copy(), train_y.copy())
    # logistic_unit_test()

    # Normalize our data: choose one of the two methods before training
    #train_x, test_x = normalize_all_pixel(train_x, test_x)
    train_x, test_x = normalize_per_pixel(train_x, test_x)

    # Reshape our data
    # train_x: shape=(2400, 64, 64) -> shape=(2400, 64*64)
    # test_x: shape=(600, 64, 64) -> shape=(600, 64*64)
    train_x = reshape2D(train_x)
    test_x = reshape2D(test_x)

    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x)
    test_x = add_one(test_x)

    # [TODO 1.11] Create TF placeholders to feed train_x and train_y when training
    m, n = train_x.shape
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, n))
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

    # [TODO 1.12] Create weights (W) using TF variables
    w = tf.Variable(tf.ones([n, 1]), dtype=tf.float32)

    # [TODO 1.13] Create a feed-forward operator
    pred = tf.matmul(x, w)

    # [TODO 1.14] Write the cost function
    # cost = -tf.reduce_sum(train_y*tf.math.log(pred)+(1-train_y)*tf.math.log(1-pred))/num_train
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = pred, labels = y)

    # Define hyper-parameters and train-related parameters
    num_epoch = 1000
    learning_rate = 0.001
    # momentum_rate = 0.9

    # [TODO 1.15] Create an SGD optimizer
    SGD = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)


    # Some meta parameters
    epochs_to_draw = 100
    all_loss = []
    plt.ion()

    # Start training
    init = tf.compat.v1.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for e in range(num_epoch):
            tic = time.clock()
            # [TODO 1.16] Compute loss and update weights here
            sess.run(SGD, feed_dict={x: train_x, y: train_y})
            cal_cost = sess.run(cost, feed_dict={x: train_x, y: train_y})
            # Update weights...
            Weight = sess.run(w)

            all_loss.append(loss)

            if (e % epochs_to_draw == epochs_to_draw-1):
                plot_loss(all_loss)
                plt.show()
                plt.pause(0.1)
                print("Epoch %d: loss is %.5f" % (e+1, loss))
            toc = time.clock()
            print(toc-tic)

        y_hat = sess.run(pred, feed_dict={x: test_x})
        test(y_hat, test_y)
