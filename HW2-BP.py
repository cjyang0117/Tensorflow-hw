# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:44:40 2019

@author: harmo
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(2)
np.random.seed(2)

def add_layer(inputs, input_size, output_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([input_size, output_size]))
    threshold = tf.Variable(tf.zeros([1, output_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + threshold
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs, Weights, threshold

n_data = np.ones((100, 2))
x0 = np.random.normal(2*n_data, 1)
x1 = np.random.normal(-2*n_data, 1)
y0 = np.full((100, 1), 1)
y1 = np.full((100, 1), -1)

x_data = np.vstack([x0,x1])
y_data = np.vstack([y0,y1])
from sklearn.utils import shuffle
x_data, y_data = shuffle(x_data, y_data, random_state=0)
#tmp = np.full((200, 1), 0.00)
#for i in range(200): tmp[i] = x_data[i].sum()
#y_result = tmp
a_data = []
b_data = []
for i in range(200):
    if y_data[i] > 0:
        a_data.append(i)
    else:
        b_data.append(i) 
a = -1
b = 1
xs = tf.placeholder(tf.float32, x_data.shape)
ys = tf.placeholder(tf.float32, y_data.shape)
y_data_r = tf.placeholder(tf.float32)

learning_rate = tf.placeholder(tf.float32, shape=[])

hiddenLayer, weights_h, threshold_h = add_layer(xs, 2, 10, tf.tanh)
outputLayer, weights_o, threshold_o = add_layer(hiddenLayer, 10, 1)

error = tf.reduce_mean(tf.reduce_sum(tf.square(ys - outputLayer), reduction_indices=[1]))
regularizers = tf.nn.l2_loss(weights_h) + tf.nn.l2_loss(weights_o) + tf.nn.l2_loss(threshold_h) + tf.nn.l2_loss(threshold_o)
error = error + 0.01 * regularizers

accuracy = tf.metrics.accuracy(labels = ys, predictions = y_data_r)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

#plt.scatter(x_data[:, 0], x_data[:,1], c = y_data.transpose()[0], s = 80, lw = 0, cmap = 'RdYlGn')
#plt.show()
#plt.ion()

init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess = tf.Session()
sess.run(init)

lr=0.1
sess.run(train, feed_dict={xs: x_data, ys: y_data, learning_rate: lr})
tmp = sess.run([error, weights_h, weights_o, threshold_h, threshold_o], feed_dict={xs: x_data, ys: y_data, learning_rate: lr})
temp_loss, temp_weights_h, temp_weights_o, temp_threshold_h, temp_threshold_o = tmp
learning_times = 1
tmp = []
while (a < b):
    learning_times += 1 
    sess.run(train, feed_dict={xs: x_data, ys: y_data, learning_rate: lr})
    tmp = sess.run([error, outputLayer, weights_h, weights_o, threshold_h, threshold_o], feed_dict={xs: x_data, ys: y_data, learning_rate: lr})
    loss, pred_, wh, wo, th, to = tmp
    if(loss < temp_loss):
        lr = lr * 1.2
        a = min(pred_[a_data])
        b = max(pred_[b_data])
        temp_loss = loss 
        temp_weights_h = wh
        temp_weights_o = wo
        temp_threshold_h = th
        temp_threshold_o = to
        print('loss:', loss)        
    else:
        if (lr > 0.05):
            sess.run(weights_h.assign(temp_weights_h))
            sess.run(weights_o.assign(temp_weights_o))
            sess.run(threshold_h.assign(temp_threshold_h))
            sess.run(threshold_o.assign(temp_threshold_o))
            lr = lr * 0.7
        else:
            print("----------Claim Undesired Attractor----------")
            break               

tmp = [None] * 200
for i in range(len(a_data)):
    tmp[a_data[i]] = 1
for i in range(len(b_data)):
    tmp[b_data[i]] = -1
acura_ = sess.run(accuracy, feed_dict={ys: y_data, y_data_r: tmp})
sess.close();

plt.cla()
plt.scatter(x_data[:, 0], x_data[:, 1], c = tmp, s = 100, lw = 0, cmap = 'RdYlGn')
plt.text(1.5, -4, 'Accuracy=%.2f' % acura_[1], fontdict={'size': 20, 'color': 'red'})
plt.ioff()
plt.show()
print("learning_times: ", learning_times) 
print("Complete!")